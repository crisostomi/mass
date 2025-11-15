from copy import deepcopy

from hydra.utils import instantiate

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import wandb

from mass.merger.tsv import TaskSingularVectorsMerger
from mass.modules.mass_gate import MassGate
from mass.modules.encoder import ImageEncoder

from mass.utils.task_vectors import get_svd_dict
from mass.utils.fusion_bench_utils import get_attr, set_attr

from mass.utils.utils import compute_task_dict, get_routing_weights, pad_output
import logging

pylogger = logging.getLogger(__name__)


num_of_tasks_to_scaling_coeff = {
    1: 1.0,
    2: 0.4,
    3: 0.35,
}


class MassAlgorithm:

    _linear_layer_cls = (nn.Linear,)
    _image_encoder_cls = (ImageEncoder,)  

    def __init__(
        self,
        merger,
        base_merger,
        zeroshot_model,
        finetuned_models,
        dataset_names,
        routing_mode,
        layer_to_hook,
        max_num_tasks_to_select,
        device: str = "cuda",
        svd_path: str = None,
        debug: bool = False,
    ):
        """

        encoder: the model used to do the first pass of delta
        router:
        zeroshot_model:
        classification_heads: list of classification heads, one for each dataset
        """
        self.dataset_names = dataset_names
        self.routing_mode = routing_mode
        self.layer_to_hook = layer_to_hook
        self.max_num_tasks_to_select = max_num_tasks_to_select
        self.device = device
        self.debug = debug
        
        self.vision = isinstance(zeroshot_model, self._image_encoder_cls)

        self.merger = merger
        self.base_merger = base_merger

        task_dicts = {}
        for dataset in dataset_names:
            pylogger.info(f"State dict of pretrinaed {zeroshot_model.__class__.__name__}: {list(zeroshot_model.state_dict().keys())}")
            task_dicts[dataset] = compute_task_dict(
                zeroshot_model.state_dict(), finetuned_models[dataset].state_dict()
            )
            del finetuned_models[dataset]
            torch.cuda.empty_cache()

        self.svd_dict = get_svd_dict(
            task_dicts,
            self.dataset_names,
            svd_path,
        )

        del task_dicts

        self.zeroshot_model = zeroshot_model
        merged_encoder = self.base_merger.merge_from_svd_dict(
            zeroshot_model,
            self.svd_dict,
        )

        merged_encoder = self.merge(merged_encoder, in_place=True)

        self.model = MassInferenceWrapper(
            layer_to_hook,
            merged_encoder,
            zeroshot_model,
            self.svd_dict,
            self.merger,
        ).to(device)

    def merge(self, base_model, in_place=True):
        if in_place:
            model = base_model
        else:
            model = deepcopy(base_model)

        self._upscale_submodules(model, self.layer_to_hook, debug=self.debug)
        return model

    def _upscale_submodules(
        self,
        zeroshot_model: nn.Module,
        name: str = None,
        debug: bool = True,
        tqdm_desc: str = "Upscaling Linear Modules",
    ):
        """
        Upscales the submodules of the pretrained model by merging them with the corresponding submodules from the fine-tuned models.

        Args:
            zeroshot_model (nn.Module): The pretrained model.
            finetuned_models (List[nn.Module]): A list of fine-tuned models.
            tqdm_desc (str): Description for the tqdm progress bar.
        """
        if debug:
            pylogger.warning("Upscaling all linear layers. This might slow down the method quite a lot, should be used only for debug purposes. Requires Wandb integration.")
                
        for name, module in tqdm(
            tuple(zeroshot_model.named_modules()),
            tqdm_desc,
            leave=False,
            dynamic_ncols=True,
        ):
            if isinstance(module, self._linear_layer_cls) and debug:
                self._upscale_linear_layer(
                    zeroshot_model,
                    name,
                )
            elif name == self.layer_to_hook:
                self._upscale_linear_layer(
                    zeroshot_model,
                    name,
                )
    def _upscale_linear_layer(
        self,
        base_model: nn.Module,
        name: str,
    ):
        """
        Upscales the submodules of the pretrained model by merging them with the corresponding submodules from the fine-tuned models.

        Args:
            zeroshot_model (nn.Module): The pretrained model.
            finetuned_models (List[nn.Module]): A list of fine-tuned models.
            tqdm_desc (str): Description for the tqdm progress bar.
        """
        name_list = name.split(".")
        # pylogger.info(f"Layer name: {name}")
        module = get_attr(base_model, name_list)

        try:

            # pylogger.info(f"Creating MassGate for layer {name}")
            mass_gate = MassGate(
                name,
                module,
                get_routing_weights(self.svd_dict, name + ".weight"), # TODO: remove hardocoding for keys
                self.dataset_names,
                self.routing_mode,
                self.max_num_tasks_to_select,
                visual=self.vision,
                debug=self.debug,
            )
            mass_gate.to(self.device)
            # pylogger.info(f" MassGate created for layer {name}")
        except Exception as e:
            pylogger.error(f"❌ Error creating MassGate: {e}")
            return
        set_attr(base_model, name_list, mass_gate)
        # pylogger.info(f"Layer type: {type(get_attr(base_model, name_list))}")


class MassInferenceWrapper(nn.Module):
    def __init__(
        self,
        layer_to_hook: str,
        base_model,
        zeroshot_model: nn.Module,
        svd_dicts: dict,
        merger: TaskSingularVectorsMerger,
        debug: bool = False,
    ):
        super().__init__()
        self.base_model = base_model
        self.zeroshot_model = zeroshot_model
        self.svd_dicts = svd_dicts
        self.merger = merger

        self.layer_to_hook = layer_to_hook

        self.max_num_tvs_to_keep = 20
        self.cached_tvs = {}

        self.debug = debug

    def collect_output(self):
        mass = get_attr(self.base_model, self.layer_to_hook.split("."))
        output = mass.output
        mass.output = None
        return output
    
    def _process_dataset_groups(self, batch, dataset_group_to_samples, processing_fn):
        batch_size = batch.shape[0]
        sample_embeddings = [None] * batch_size

        for dataset_group, assigned_sample_idxs in dataset_group_to_samples.items():
            assigned_sample_idxs = torch.tensor(assigned_sample_idxs)
            merged_model = self._apply_tv(list(dataset_group))
            
            group_batch = batch[assigned_sample_idxs]
            merged_model.to(batch.device)
            
            group_output = processing_fn(merged_model, group_batch)

            for j, idx in enumerate(assigned_sample_idxs):
                sample_embeddings[idx] = group_output[j : j + 1]

        return sample_embeddings
    
    def embed_image(self, batch, classification_heads, num_classes):
        self.base_model(batch)
        
        selected_dataset_idxs, _, dataset_group_to_samples = self.collect_output()

        # needed to handle the difference in image/text encoders (see below)
        def process_group(merged_model, group_batch):
            return merged_model(group_batch)
        
        sample_embeddings = self._process_dataset_groups(batch, dataset_group_to_samples, process_group)
        sample_embeddings = torch.cat(sample_embeddings, dim=0)

        outputs = []

        for sample_routed_datasets, sample_embedding in zip(
            selected_dataset_idxs, sample_embeddings
        ):

            assert isinstance(
                sample_routed_datasets, (int, list, tuple)
            ), f"Unexpected type for routing indices: {type(sample_routed_datasets)}"

            # logits for each dataset the sample was routed to, so a tensor for each routed_dataset in len(sample_routed_datasets)
            candidate_logits = [
                classification_heads[j](sample_embedding.unsqueeze(0))
                for j in sample_routed_datasets
            ]
            # for each dataset, get the heads_selection_criteria score
            candidate_scores = [
                torch.max(logits).item()
                for logits in candidate_logits
            ] 
            # get the index of the best score among the datasets
            best_idx = candidate_scores.index(max(candidate_scores))
            # get the logits of the best dataset
            logits = candidate_logits[best_idx]

            outputs.append(logits)

        assert (
            num_classes is not None
        ), "Output classes not set. Use set_metrics() method to set them."

        return pad_output(outputs, num_classes)
    

    def generate(self, batch, max_length):
        self.base_model.generate(batch, max_length=max_length)

        _, _, dataset_group_to_samples = self.collect_output()

        def process_group(merged_model, group_batch):
            return merged_model.generate(group_batch, max_length=max_length)
        
        sample_embeddings = self._process_dataset_groups(batch, dataset_group_to_samples, process_group)

        max_len = max(t.size(1) for t in sample_embeddings if t is not None)

        # Try to read pad_token_id from the merged model config, else default to -100
        pad_token_id = getattr(getattr(self.base_model, "config", None), "pad_token_id", -100)
        pad_value = pad_token_id if isinstance(pad_token_id, int) else -100

        for i, t in enumerate(sample_embeddings):
            seq_len = t.size(1)
            if seq_len < max_len:
                # F.pad pads as (pad_left, pad_right) for last dimension
                sample_embeddings[i] = torch.nn.functional.pad(
                    t, (0, max_len - seq_len), value=pad_value
                )


        sample_embeddings = torch.cat(sample_embeddings, dim=0)

        return sample_embeddings

    @torch.no_grad()
    def _apply_tv(self, dataset_names):
        """Apply the aggregated task vector to the model."""

        dataset_combo = "_".join(dataset_names)

        if dataset_combo in self.cached_tvs:
            return self.cached_tvs[dataset_combo]

        if isinstance(self.merger, TaskSingularVectorsMerger):

            aggregated = self.merger.merge_from_svd_dict(
                self.zeroshot_model,
                {dataset_name: self.svd_dicts[dataset_name] for dataset_name in dataset_names},
            )

            if len(self.cached_tvs) > self.max_num_tvs_to_keep:
                self.flush_cache()

            self.cached_tvs[dataset_combo] = aggregated

            return aggregated

        else:
            raise NotImplementedError

    def flush_cache(self):
        self.cached_tvs = {}
        torch.cuda.empty_cache()
        
    @property
    def train_preprocess(self):
        return getattr(self.zeroshot_model, 'train_preprocess', None)

    @property
    def val_preprocess(self):
        return getattr(self.zeroshot_model, 'val_preprocess', None)

    # Logging

    def logging(self, logger, current_task):
        """Log statistics from all MassGate modules layer-wise"""
        layer_stats = {}
        layer_accuracy_stats = {}

        # Collect stats from all MassGate layers
        for layer_name, module in self.base_model.named_modules():
            if isinstance(module, MassGate):
                if not module.norms_to_log:
                    pylogger.warning(f"No norms to log for layer {layer_name}")
                    continue

                norms_array = np.array(module.norms_to_log)
                mean_coeffs = norms_array.mean(axis=0)
                std_coeffs = norms_array.std(axis=0)

                layer_stats[layer_name] = {
                    "mean_coeffs": mean_coeffs,
                    "std_coeffs": std_coeffs,
                    "dataset_names": list(module.dataset_names),
                }

                # Collect task accuracy data if available
                if module.layer_accuracy_to_log[module.name]:
                    layer_accuracy_stats[layer_name] = {
                        "predictions": module.layer_accuracy_to_log[module.name],
                        "dataset_names": list(module.dataset_names),
                    }

        if not layer_stats:
            pylogger.warning("No MassGate layers found with logging data")
            return

        # Log coefficient statistics
        for layer_name, stats in layer_stats.items():
            mean_coeffs = stats["mean_coeffs"]
            std_coeffs = stats["std_coeffs"]
            dataset_names = stats["dataset_names"]

            # Import here to avoid circular imports
            from mass.utils.plots import plot_interactive_coefficients_std

            fig_std = plot_interactive_coefficients_std(mean_coeffs, std_coeffs, dataset_names)

            logger.experiment.log(
                {
                    f"norms/{current_task}/{layer_name}": wandb.Plotly(fig_std),
                }
            )

            # pylogger.info(f"Logged coefficient statistics for MassGate layer: {layer_name}")

        # Log task accuracy statistics
        if layer_accuracy_stats:
            from mass.utils.plots import create_interactive_layer_task_accuracy_plot

            # Get dataset names from any layer (they should all be the same)
            dataset_names = next(iter(layer_accuracy_stats.values()))["dataset_names"]

            # Find the index of the current task
            if current_task in dataset_names:
                current_task_idx = dataset_names.index(current_task)

                # Create a single dict with all layers for the CURRENT task accuracy plot
                all_layer_predictions = {}
                for layer_name, accuracy_stats in layer_accuracy_stats.items():
                    all_layer_predictions[layer_name] = accuracy_stats["predictions"]

                if all_layer_predictions:
                    fig_accuracy = create_interactive_layer_task_accuracy_plot(
                        all_layer_predictions,
                        current_task_idx,
                        dataset_names,
                        title=f"Task Accuracy for {current_task} across all layers",
                    )

                    logger.experiment.log(
                        {
                            f"task_accuracy/{current_task}": wandb.Plotly(fig_accuracy),
                        }
                    )

                pylogger.info(f"Logged task accuracy statistics for current task: {current_task}")
            else:
                pylogger.warning(
                    f"Current task '{current_task}' not found in dataset names: {dataset_names}"
                )

        # Reset all logging stats after logging
        for layer_name, module in self.base_model.named_modules():
            if isinstance(module, MassGate):
                module.reset_to_log()
