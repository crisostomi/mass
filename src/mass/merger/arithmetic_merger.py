import copy
import logging
from typing import Dict, List
import torch
from mass.merger.merger import TaskVectorBasedMerger
from mass.modules.encoder import ImageEncoder
from mass.utils.utils import apply_dict_to_model, compute_task_dict, sum_task_dict

pylogger = logging.getLogger(__name__)

class TaskArithmeticMerger(TaskVectorBasedMerger):

    def __init__(self, optimal_alpha, device="cuda"):
        super().__init__()

        self.optimal_alpha = optimal_alpha

    def merge(self, base_model: ImageEncoder, finetuned_models: Dict[str,ImageEncoder]) -> ImageEncoder:

        comulative_dict = {}
        
        base_model.cuda()

        datasets = list(finetuned_models.keys())
        pretrained_model = copy.deepcopy(base_model)

        for dataset in datasets:
            finetuned_models[dataset].cuda()
            comulative_dict = sum_task_dict(
                comulative_dict,
                compute_task_dict(base_model.state_dict(), finetuned_models[dataset].state_dict()),
            )
            del finetuned_models[dataset]  # Delete one model at a time
            torch.cuda.empty_cache()

        merged_encoder = apply_dict_to_model(
            comulative_dict, pretrained_model, coefficient=self.optimal_alpha
        )

        return merged_encoder
