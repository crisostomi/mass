import copy
import logging
from mass.merger.merger import TaskVectorBasedMerger
from mass.modules.encoder import ImageEncoder
from mass.utils.utils import (
    apply_dict_to_model,
    compute_task_dict,
    print_memory,
    sum_task_dict,
)
from mass.task_vectors.isotropic_merging import isotropic_sum

import torch

pylogger = logging.getLogger(__name__)


class IsotropicMerger(TaskVectorBasedMerger):

    def __init__(self, optimal_alphas, model_name):
        super().__init__()

        self.optimal_alphas = optimal_alphas

        self.model_name = model_name
        # self.svd_compress_factor = svd_compress_factor

    def merge(self, base_model, finetuned_models, device="cuda"):

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

        # svd_dict = get_svd_dict(
        #     task_dicts, datasets, self.svd_path, self.svd_compress_factor
        # )

        multi_task_vector = isotropic_sum(
            cumulative_dict=cumulative_dict,
            datasets=datasets,
            device=device,
        )

        model_name = self.model_name
        num_tasks = len(datasets)

        if (
            model_name in self.optimal_alphas
            and num_tasks in self.optimal_alphas[model_name]
        ):
            coefficient = self.optimal_alphas[model_name][num_tasks]
        else:
            coefficient = 1.0 / num_tasks
            pylogger.warning(
                f"Warning: using default coefficient {coefficient} for model {model_name} with {num_tasks} tasks"
            )

        merged_encoder: ImageEncoder = copy.deepcopy(base_model)

        merged_encoder = apply_dict_to_model(
            multi_task_vector,
            merged_encoder,
            coefficient=coefficient,
        )

        return merged_encoder
