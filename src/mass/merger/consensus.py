import copy
import logging
from typing import Dict, List
import torch
from mass.merger.merger import TaskVectorBasedMerger
from mass.modules.encoder import ImageEncoder
from mass.utils.utils import apply_dict_to_model, compute_task_dict, sum_task_dict, print_memory
from mass.utils.consensus_utils import (
    state_dict_to_vector,
    construct_consensus_mask,
    construct_tall_mask,
    apply_eval_mask,
)

pylogger = logging.getLogger(__name__)


class ConsensusMerger(TaskVectorBasedMerger):

    def __init__(
        self, model_name, prun_thre_k, load_mask, mask_location, optimal_alphas, device="cuda"
    ):
        super().__init__()

        self.model_name = model_name
        self.prun_thre_k = prun_thre_k
        self.load_mask = load_mask
        self.mask_location = mask_location
        self.optimal_alphas = optimal_alphas

    def merge(
        self,
        base_model: ImageEncoder,
        finetuned_models: Dict[str, ImageEncoder],
    ):

        cumulative_dict = {}
        eval_masks = None

        datasets = list(finetuned_models.keys())

        flattened_finetunings = torch.vstack(
            [state_dict_to_vector(check) for check in finetuned_models.values()]
        )
        flattened_pretrained = state_dict_to_vector(base_model.state_dict())

        # compute the task vector as {\theta_t - \theta_0}.
        tv_flattened_checkpoints = flattened_finetunings - flattened_pretrained
        merged_tv = tv_flattened_checkpoints.sum(dim=0)

        if self.load_mask:
            # load tall masks directly from storage

            consensus_mask = construct_consensus_mask(
                base_model.state_dict(),
                self.prun_thre_k,
                self.model_name,
                self.mask_location,
                datasets,
            )

            for dataset in datasets:
                cumulative_dict = sum_task_dict(
                    cumulative_dict,
                    compute_task_dict(
                        base_model.state_dict(), finetuned_models[dataset]  # .state_dict()
                    ),
                )
                del finetuned_models[dataset]  # Delete one model at a time
                torch.cuda.empty_cache()

            model_name = self.model_name
            num_tasks = len(datasets)

            if model_name in self.optimal_alphas and num_tasks in self.optimal_alphas[model_name]:
                coefficient = self.optimal_alphas[model_name][num_tasks]
                pylogger.info(
                    f"Using optimal coefficient {coefficient} for model {model_name} with {num_tasks} tasks"
                )
            else:
                coefficient = 1.0 / num_tasks
                pylogger.warning(
                    f"Warning: using default coefficient {coefficient} for model {model_name} with {num_tasks} tasks"
                )

            merged_encoder = apply_eval_mask(
                copy.deepcopy(cumulative_dict),
                copy.deepcopy(base_model),
                eval_mask=consensus_mask,
                coefficient=coefficient,
            )

            eval_masks = consensus_mask
            print_memory("after computing task dicts")
        else:
            eval_masks = construct_tall_mask(
                tv_flattened_checkpoints,
                flattened_finetunings,
                flattened_pretrained,
                merged_tv,
                base_model.state_dict(),
                datasets,
            )

            merged_encoder: ImageEncoder = copy.deepcopy(base_model)

            merged_encoder = apply_dict_to_model(
                cumulative_dict,
                merged_encoder,
                coefficient=1.0,
            )

        return merged_encoder, eval_masks
