import copy
import logging
from typing import Dict, List
import torch
from mass.merger.merger import TaskVectorBasedMerger
from mass.modules.encoder import ImageEncoder
from mass.utils.utils import apply_dict_to_model, compute_task_dict, sum_task_dict, print_memory

pylogger = logging.getLogger(__name__)


class TaskArithmeticMerger(TaskVectorBasedMerger):

    def __init__(self, optimal_alpha, device="cuda"):
        super().__init__()

        self.optimal_alpha = optimal_alpha

    def merge(
        self, base_model: ImageEncoder, finetuned_models: Dict[str, ImageEncoder]
    ) -> ImageEncoder:

        cumulative_dict = {}

        datasets = list(finetuned_models.keys())

        for dataset in datasets:
            cumulative_dict = sum_task_dict(
                cumulative_dict,
                compute_task_dict(
                    base_model.state_dict(), finetuned_models[dataset]  # .state_dict()
                ),
            )
            del finetuned_models[dataset]  # Delete one model at a time
            torch.cuda.empty_cache()

        print_memory("after computing task dicts")

        merged_encoder: ImageEncoder = copy.deepcopy(base_model)

        if len(datasets) not in self.optimal_alpha:
            coefficient = 0.7733 * len(datasets) ** (-0.5517)
        else:
            coefficient = self.optimal_alpha[len(datasets)]

        merged_encoder = apply_dict_to_model(
            cumulative_dict,
            merged_encoder,
            coefficient=coefficient,
        )

        return merged_encoder
