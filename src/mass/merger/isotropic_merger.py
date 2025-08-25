import copy
from mass.merger.merger import TaskVectorBasedMerger
from mass.modules.encoder import ImageEncoder
from mass.utils.utils import apply_dict_to_model, compute_task_dict, print_memory
from mass.task_vectors.task_singular_vectors import (
    get_svd_dict,
    isotropic_sum,
    sum_svd_no_redundant_tasks_simple,
)

import torch


class IsotropicMerger(TaskVectorBasedMerger):

    def __init__(self, optimal_alphas, svd_path, svd_compress_factor):
        super().__init__()

        self.optimal_alphas = optimal_alphas
        self.svd_path = svd_path
        self.svd_compress_factor = svd_compress_factor

    def merge(self, base_model, finetuned_models):

        task_dicts = {}

        datasets = list(finetuned_models.keys())

        for dataset in datasets:
            task_dicts[dataset] = compute_task_dict(
                base_model.state_dict(), finetuned_models[dataset]
            )
            del finetuned_models[dataset]  # Delete one model at a time
            torch.cuda.empty_cache()

        print_memory("after computing task dicts")

        svd_dict = get_svd_dict(
            task_dicts, datasets, self.svd_path, self.svd_compress_factor
        )

        multi_task_vector = isotropic_sum(
            ref_state_dict=copy.deepcopy(base_model.state_dict()),
            svd_dict=svd_dict,
        )

        model_name = self.cfg.nn.module.encoder.model_name

        num_tasks = len(finetuned_models)

        if (
            model_name in self.optimal_alphas
            and num_tasks in self.optimal_alphas[model_name]
        ):
            coefficient = self.optimal_alphas[model_name][num_tasks]

        merged_encoder: ImageEncoder = copy.deepcopy(base_model)

        merged_encoder = apply_dict_to_model(
            multi_task_vector,
            merged_encoder,
            coefficient=coefficient,
        )

        return merged_encoder
