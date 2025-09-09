import copy
from mass.merger.merger import TaskVectorBasedMerger
from mass.modules.encoder import ImageEncoder
from mass.utils.utils import apply_dict_to_model, compute_task_dict, print_memory
from mass.task_vectors.task_singular_vectors import (
    get_svd_dict,
    sum_svd,
)

import torch


class TaskSingularVectorsMerger(TaskVectorBasedMerger):

    def __init__(self, svd_path, svd_compress_factor, non_matrix_params_aggregation):
        super().__init__()

        self.svd_path = svd_path
        self.svd_compress_factor = svd_compress_factor
        self.non_matrix_params_aggregation = non_matrix_params_aggregation

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

        multi_task_vector = sum_svd(
            ref_state_dict=copy.deepcopy(base_model.state_dict()),
            svd_dicts=svd_dict,
            non_matrix_params_aggregation=self.non_matrix_params_aggregation,
        )

        merged_encoder: ImageEncoder = copy.deepcopy(base_model)

        merged_encoder = apply_dict_to_model(
            multi_task_vector,
            merged_encoder,
        )

        return merged_encoder
