from copy import deepcopy

from hydra.utils import instantiate

import torch
import torch.nn as nn

from mass.merger.tsv import TaskSingularVectorsMerger
from mass.modules.mass_gate import MassGate

from mass.task_vectors.task_singular_vectors import get_svd_dict
from mass.utils.fusion_bench_utils import get_attr, set_attr

from mass.utils.utils import (
    compute_task_dict,
    get_routing_weights)
import logging

pylogger = logging.getLogger(__name__)


num_of_tasks_to_scaling_coeff = {
    1: 1.0,
    2: 0.4,
    3: 0.35,
}


class MassAlgorithm():
    
    _linear_layer_cls = (nn.Linear,)
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
        
        self.merger = merger
        self.base_merger = base_merger
        
        task_dicts = {}
        for dataset in dataset_names:
            task_dicts[dataset] = compute_task_dict(
                zeroshot_model.state_dict(), finetuned_models[dataset].state_dict()
            )
            torch.cuda.empty_cache()


        self.svd_dict = get_svd_dict(
            task_dicts, 
            self.dataset_names, 
            svd_path,
        )

        pylogger.info(f"SVD dict keys: {self.svd_dict['cola'].keys()}")

        del task_dicts
        
        self.zeroshot_model = zeroshot_model
        merged_encoder = self.base_merger.merge(zeroshot_model, {dataset: finetuned_models[dataset].state_dict() for dataset in dataset_names})
        
        finetuned_models_list = list(finetuned_models.values())
        del finetuned_models
        merged_encoder = self.merge(merged_encoder, finetuned_models_list, in_place=True)
        
        self.model = MassInferenceWrapper(
            layer_to_hook,
            merged_encoder,
            zeroshot_model,
            self.svd_dict,
            self.merger,
        ).to(device)
        
        pylogger.info(f"{type(get_attr(self.model.base_model, self.layer_to_hook.split('.')))}")
        
        
    
    def merge(self, base_model, finetuned_models, in_place=True):
        if in_place:
            model = base_model
        else:
            model = deepcopy(base_model)

        self._upscale_submodules(model, self.layer_to_hook)
        return model
    
    def _upscale_submodules(
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
        # TODO: do we need this still?
        # replace_attention_with_linear(zeroshot_model, finetuned_models)
        name_list = name.split(".")
        pylogger.info(f"Layer name {name}")
        module = get_attr(base_model, name_list)
        
        try:
            pylogger.info(f"Svd dict keys: {self.svd_dict.keys()}")
            # TODO: can we fix once for all this layer key mess
            pylogger.info(get_routing_weights(self.svd_dict, self.layer_to_hook + ".weight"))
            
            pylogger.info(f"Creating MassGate for layer {self.layer_to_hook}")
            mass_gate = MassGate(
                module,
                get_routing_weights(self.svd_dict, self.layer_to_hook + ".weight"),
                self.dataset_names,
                self.routing_mode, 
                self.max_num_tasks_to_select,
                token_selection="mean"
            )
            mass_gate.to(self.device)
        except Exception as e:
            pylogger.error(f"Error creating MassGate: {e}")
            return
        set_attr(base_model, name_list, mass_gate)
        pylogger.info(f"Layer type:{type(get_attr(base_model, name_list))}")



class MassInferenceWrapper(nn.Module):
    def __init__(
        self, 
        layer_to_hook: str,
        base_model,
        zeroshot_model: nn.Module,
        svd_dicts: dict,
        merger: TaskSingularVectorsMerger,
    ):
        super().__init__()
        self.base_model = base_model
        self.zeroshot_model = zeroshot_model
        self.svd_dicts = svd_dicts
        self.merger = merger
        
        self.layer_to_hook = layer_to_hook
        
        self.max_num_tvs_to_keep = 1
        self.cached_tvs = {}
        
        
    def collect_output(self):
        mass = get_attr(self.base_model, self.layer_to_hook.split("."))
        return mass.output
    
    def generate(self, batch, max_length):
        pylogger.info(f"Batch size: {batch.shape[0]}")
        self.base_model.generate(batch, max_length=max_length)
        
        _, _, dataset_group_to_samples = self.collect_output()

        batch_size = batch.shape[0]
        sample_embeddings = [None] * batch_size

        for dataset_group, assigned_sample_idxs in dataset_group_to_samples.items():

            assigned_sample_idxs = torch.tensor(
                assigned_sample_idxs
            )  # Ensure assigned_sample_idxs is also a tensor

            merged_model = self._apply_tv(list(dataset_group))

            # (num_samples_in_group, C, H, W)
            group_batch = batch[assigned_sample_idxs]

            # (num_samples_in_group, embedding_dim)
            merged_model.to(batch.device)
            group_output = merged_model.generate(group_batch, max_length=max_length)

            for j, idx in enumerate(assigned_sample_idxs):
                sample_embeddings[idx] = group_output[j : j + 1]

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
                {
                    dataset_name: self.svd_dicts[dataset_name]
                    for dataset_name in dataset_names
                }
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



    