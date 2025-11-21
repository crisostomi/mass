from copy import deepcopy
import copy
from typing import List, Optional
from collections import OrderedDict

from fastapi import routing
from hydra.utils import instantiate

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import wandb

from mass.merger.arithmetic_merger import TaskArithmeticMerger
from mass.merger.dummy_merger import DummyMerger
from mass.merger.no_red_tsv import TaskSingularVectorsMergerNoRedundancy
from mass.merger.principal_angles_merger import TaskSingularVectorsWithPrincipalAngles
from mass.merger.tsv import TaskSingularVectorsMerger
from mass.modules.mass_gate import MassGate
from mass.modules.encoder import ImageEncoder
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from mass.utils.task_vectors import get_svd_dict
from mass.utils.fusion_bench_utils import get_attr, set_attr

from mass.utils.utils import (
    compute_task_dict,
    get_routing_weights,
    get_routing_weights_from_finetuned,
    get_routing_weights_from_task_dict,
    get_routing_weights_whitened,
    pad_output,
    reconstruct_tv_from_svddict,
)
import logging

pylogger = logging.getLogger(__name__)



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
        use_finetuned: bool = False,
        whitened: bool = False,
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
        self.whitened = whitened
        self.use_finetuned = use_finetuned

        self.vision = isinstance(zeroshot_model, self._image_encoder_cls)

        self.merger = merger
        self.base_merger = base_merger

        if (
            not self.use_finetuned
        ):  # finetuned might be pre-merged models, that we suggest to use in case of very large and few models
            task_dicts = {}
            for dataset in dataset_names:
                task_dicts[dataset] = compute_task_dict(
                    zeroshot_model.state_dict(), finetuned_models[dataset].state_dict()
                )
                del finetuned_models[dataset]
                torch.cuda.empty_cache()

            self.zeroshot_model = zeroshot_model

            if (
                isinstance(self.base_merger, TaskSingularVectorsMerger)
                or isinstance(self.base_merger, TaskSingularVectorsMergerNoRedundancy) or isinstance(self.base_merger, TaskSingularVectorsWithPrincipalAngles)
            ) and isinstance(self.merger, TaskSingularVectorsMerger):
                self.svd_dict = get_svd_dict(
                    task_dicts,
                    self.dataset_names,
                    svd_path,
                )

                del task_dicts

                merged_encoder = self.base_merger.merge_from_svd_dict(
                    zeroshot_model,
                    self.svd_dict,
                )
            elif isinstance(self.base_merger, TaskArithmeticMerger) and isinstance(
                self.merger, TaskArithmeticMerger
            ):
                merged_encoder = self.base_merger.merge_from_task_dicts(
                    zeroshot_model,
                    task_dicts,
                )
                self.task_dicts = task_dicts
            elif isinstance(
                self.base_merger, TaskSingularVectorsMergerNoRedundancy
            ) and isinstance(self.merger, TaskArithmeticMerger):
                svd_dict = get_svd_dict(
                    task_dicts,
                    self.dataset_names,
                    svd_path,
                )
                merged_encoder = self.base_merger.merge_from_svd_dict(
                    zeroshot_model,
                    svd_dict,
                )
                self.task_dicts = task_dicts
            elif isinstance(self.base_merger, DummyMerger):
                if isinstance(self.merger, TaskArithmeticMerger):
                    self.task_dicts = task_dicts
                elif isinstance(self.merger, TaskSingularVectorsMerger):
                    self.svd_dict = get_svd_dict(
                        task_dicts,
                        self.dataset_names,
                        svd_path,
                    )

                    del task_dicts
            else:
                raise NotImplementedError(
                    f"Base Merger type {type(self.base_merger)} and Merger {type(self.merger)} not supported yet."
                )
        else:
            assert (
                routing_mode == "top1"
            ), f"When using finetuned models, only 'top1' routing is supported. Given mode: {routing_mode}"
            self.zeroshot_model = zeroshot_model
            self.finetuned = finetuned_models
            merged_encoder = copy.deepcopy(zeroshot_model)

        merged_encoder = self.merge(merged_encoder, in_place=True)
        
        if isinstance(self.base_merger, TaskSingularVectorsMergerNoRedundancy) and isinstance(self.merger, TaskArithmeticMerger):
            self.task_dicts = {task: reconstruct_tv_from_svddict(svd_dict) for task, svd_dict in svd_dict.items()}

        self.model = MassInferenceWrapper(
            layer_to_hook,
            merged_encoder,
            self.zeroshot_model,
            svd_dicts=(
                self.svd_dict
                if (
                    (
                        (
                            isinstance(self.base_merger, TaskSingularVectorsMerger)
                            or isinstance(
                                self.base_merger, TaskSingularVectorsMergerNoRedundancy
                            ) or isinstance(self.base_merger, TaskSingularVectorsWithPrincipalAngles)
                        )
                        and isinstance(self.merger, TaskSingularVectorsMerger)
                        or (
                            isinstance(self.base_merger, DummyMerger)
                            and isinstance(self.merger, TaskSingularVectorsMerger)
                        )
                    )
                    and not self.use_finetuned
                )
                else None
            ),
            task_dicts=(
                self.task_dicts
                if (
                    (
                        isinstance(self.merger, TaskArithmeticMerger)
                        and isinstance(self.base_merger, DummyMerger)
                    )
                    or isinstance(self.base_merger, TaskArithmeticMerger) or (isinstance(self.base_merger, TaskSingularVectorsMergerNoRedundancy) and isinstance(self.merger, TaskArithmeticMerger))
                )
                and not self.use_finetuned
                else None
            ),
            finetuned=self.finetuned if self.use_finetuned else None,
            merger=self.merger,
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
            pylogger.warning(
                "Upscaling all linear layers. This might slow down the method quite a lot, should be used only for debug purposes. Requires Wandb integration."
            )

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

    def get_routing_weights(self, name: str, dtype=torch.float32):
        if not self.use_finetuned:
            if isinstance(self.merger, TaskSingularVectorsMerger) or isinstance(
                self.merger, TaskSingularVectorsMergerNoRedundancy
            ):  #
                if not self.whitened:
                    return get_routing_weights(
                        self.svd_dict, name + ".weight"
                    )  # TODO: remove hardocoding for keys
                else:
                    return get_routing_weights_whitened(
                        self.svd_dict, name + ".weight"
                    )
            elif isinstance(self.merger, TaskArithmeticMerger):
                return get_routing_weights_from_task_dict(
                    self.task_dicts, name + ".weight"
                )
            else:
                raise NotImplementedError(
                    f"Merger type {type(self.merger)} not supported yet."
                )
        else:
            return get_routing_weights_from_finetuned(
                self.finetuned, self.zeroshot_model, name + ".weight", dtype=dtype
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
        module = get_attr(base_model, name_list)

        try:
            dtype = self.zeroshot_model.dtype if hasattr(self.zeroshot_model, "dtype") else torch.float32
            
            routing_weights = self.get_routing_weights(
                name, dtype=dtype
            )

            mass_gate = MassGate(
                name,
                module,
                routing_weights,
                self.dataset_names,
                self.routing_mode,
                self.max_num_tasks_to_select,
                visual=self.vision,
                debug=self.debug,
            )
            mass_gate.to(self.device)
        except Exception as e:
            pylogger.error(f"❌ Error creating MassGate: {e}")
            return
        set_attr(base_model, name_list, mass_gate)


class MassInferenceWrapper(nn.Module):
    def __init__(
        self,
        layer_to_hook: str,
        base_model,
        zeroshot_model: nn.Module,
        merger: TaskSingularVectorsMerger,
        svd_dicts=None,
        task_dicts=None,
        finetuned=None,
        debug: bool = False,
        device: str = "cuda",
    ):
        super().__init__()
        assert (
            sum(x is not None for x in [svd_dicts, task_dicts, finetuned]) == 1
        ), "Exactly one of `svd_dicts`, `task_dicts`, or `finetuned` must be non-None."

        self.base_model = base_model
        self.zeroshot_model = zeroshot_model
        self.svd_dicts = svd_dicts or None
        self.task_dicts = task_dicts or None
        self.finetuned = finetuned or None
        self.merger = merger

        self.layer_to_hook = layer_to_hook

        self.max_num_tvs_to_keep = 10
        self.cached_tvs = OrderedDict()

        self.debug = debug

        self.config = (
            self.zeroshot_model.config
            if hasattr(self.zeroshot_model, "config")
            else None
        )
        self.name_or_path = (
            self.zeroshot_model.name_or_path
            if hasattr(self.zeroshot_model, "name_or_path")
            else None
        )
        self.dtype = (
            self.zeroshot_model.dtype if hasattr(self.zeroshot_model, "dtype") else None
        )
        if self.finetuned is not None:
            self.zeroshot_model = None

        self.device = torch.device(device)  # Store the target device

    def to(self, device, *args, **kwargs):
        device_obj = torch.device(device)
        self.device = device_obj
        self.base_model.to(device_obj)
        if self.zeroshot_model is not None:
            self.zeroshot_model.to(device_obj)
        super().to(device, *args, **kwargs)

        return self

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

        def process_group(merged_model, group_batch):
            return merged_model(group_batch)

        sample_embeddings = self._process_dataset_groups(
            batch, dataset_group_to_samples, process_group
        )
        sample_embeddings = torch.cat(sample_embeddings, dim=0)

        outputs = []

        for sample_routed_datasets, sample_embedding in zip(
            selected_dataset_idxs, sample_embeddings
        ):

            assert isinstance(
                sample_routed_datasets, (int, list, tuple)
            ), f"Unexpected type for routing indices: {type(sample_routed_datasets)}"

            candidate_logits = [
                classification_heads[j](sample_embedding.unsqueeze(0))
                for j in sample_routed_datasets
            ]

            candidate_scores = [torch.max(logits).item() for logits in candidate_logits]
            best_idx = candidate_scores.index(max(candidate_scores))
            logits = candidate_logits[best_idx]

            outputs.append(logits)

        assert (
            num_classes is not None
        ), "Output classes not set. Use set_metrics() method to set them."

        return pad_output(outputs, num_classes)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        **kwargs,
    ):

        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        if position_ids is not None:
            position_ids = position_ids.to(self.device)

        self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )

        _, _, dataset_group_to_samples = self.collect_output()

        batch_size = input_ids.shape[0]
        final_logits = torch.zeros(
            batch_size,
            input_ids.shape[1],
            self.config.vocab_size,
            device=self.device,
            dtype=self.dtype,
        )

        for dataset_group, assigned_sample_idxs in dataset_group_to_samples.items():
            if not assigned_sample_idxs:
                continue

            merged_model = self._apply_tv(list(dataset_group))

            group_input_ids = input_ids[assigned_sample_idxs]
            group_attention_mask = (
                attention_mask[assigned_sample_idxs]
                if attention_mask is not None
                else None
            )
            group_position_ids = (
                position_ids[assigned_sample_idxs] if position_ids is not None else None
            )

            with torch.no_grad():
                group_outputs = merged_model(
                    input_ids=group_input_ids,
                    attention_mask=group_attention_mask,
                    position_ids=group_position_ids,
                )

            logits = group_outputs.logits

            if logits.ndim == 2:
                num_samples_in_group = group_input_ids.shape[0]
                sequence_length = group_input_ids.shape[1]
                logits = logits.view(num_samples_in_group, sequence_length, -1)

            target_vocab_size = final_logits.shape[-1]
            logits = logits[:, :, :target_vocab_size]

            final_logits[assigned_sample_idxs] = logits.to(final_logits.dtype)

        return CausalLMOutputWithPast(logits=final_logits)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        **generation_kwargs,
    ) -> torch.Tensor:

        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        _, _, dataset_group_to_samples = self.collect_output()

        batch_size = input_ids.shape[0]
        output_sequences = [None] * batch_size

        for dataset_group, assigned_sample_idxs in dataset_group_to_samples.items():
            if not assigned_sample_idxs:
                continue

            merged_model = self._apply_tv(list(dataset_group))

            group_input_ids = input_ids[assigned_sample_idxs]
            group_attention_mask = (
                attention_mask[assigned_sample_idxs]
                if attention_mask is not None
                else None
            )

            group_output = merged_model.generate(
                input_ids=group_input_ids,
                attention_mask=group_attention_mask,
                **generation_kwargs,
            )

            for i, original_idx in enumerate(assigned_sample_idxs):
                output_sequences[original_idx] = group_output[i]

        pad_token_id = self.config.pad_token_id or self.config.eos_token_id

        max_len = max(seq.size(0) for seq in output_sequences)

        final_output = torch.full(
            (batch_size, max_len), pad_token_id, dtype=torch.long, device=self.device
        )
        for i, seq in enumerate(output_sequences):
            final_output[i, : seq.size(0)] = seq

        return final_output

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """
        Delegates the call to the underlying zeroshot model, enabling KV caching.
        """
        return self.base_model.prepare_inputs_for_generation(*args, **kwargs)

    def tie_weights(self):
        """
        This method is called by the HFLM wrapper during initialization.
        We delegate the call to our underlying Hugging Face models to ensure
        their weights (e.g., embeddings and lm_head) are correctly tied.
        """
        if self.zeroshot_model is not None:
            self.zeroshot_model.tie_weights()
        self.base_model.tie_weights()

    def _apply_tv(self, dataset_names):
        """Apply the aggregated task vector to the model."""

        dataset_combo = "_".join(dataset_names)

        if self.finetuned is not None:
            aggregated = self.finetuned[dataset_names[0]].to(self.device)
        else:
            if dataset_combo in self.cached_tvs:
                return self.cached_tvs[dataset_combo]

            if isinstance(self.merger, TaskSingularVectorsMerger):

                aggregated = self.merger.merge_from_svd_dict(
                    self.zeroshot_model,
                    {
                        dataset_name: self.svd_dicts[dataset_name]
                        for dataset_name in dataset_names
                    },
                )
            elif isinstance(self.merger, TaskArithmeticMerger):

                aggregated = self.merger.merge_from_task_dicts(
                    self.zeroshot_model,
                    {
                        dataset_name: self.task_dicts[dataset_name]
                        for dataset_name in dataset_names
                    },
                )

            else:
                raise NotImplementedError

        if len(self.cached_tvs) > self.max_num_tvs_to_keep:
            pylogger.warning("Flushing TV cache to save memory...")
            self.flush_cache()
            self.cached_tvs[dataset_combo] = aggregated
        return aggregated

    def flush_cache(self):
        self.cached_tvs = {}
        torch.cuda.empty_cache()

    @property
    def train_preprocess(self):
        return getattr(self.zeroshot_model, "train_preprocess", None)

    @property
    def val_preprocess(self):
        return getattr(self.zeroshot_model, "val_preprocess", None)

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

            fig_std = plot_interactive_coefficients_std(
                mean_coeffs, std_coeffs, dataset_names
            )

            logger.experiment.log(
                {
                    f"norms/{current_task}/{layer_name}": wandb.Plotly(fig_std),
                }
            )

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

            else:
                pylogger.warning(
                    f"Current task '{current_task}' not found in dataset names: {dataset_names}"
                )

        # Reset all logging stats after logging
        for layer_name, module in self.base_model.named_modules():
            if isinstance(module, MassGate):
                module.reset_to_log()
