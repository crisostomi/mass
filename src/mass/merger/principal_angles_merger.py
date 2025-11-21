import copy
import logging
from mass.merger.merger import TaskVectorBasedMerger
from mass.modules.encoder import ImageEncoder
from mass.utils.utils import apply_dict_to_model, compute_task_dict, print_memory
from mass.utils.task_vectors import (
    get_svd_dict,
    sum_svd_no_redundant_tasks_simple,
    sum_svd_principal_angles,
)

import torch

pylogger = logging.getLogger(__name__)


class TaskSingularVectorsWithPrincipalAngles(TaskVectorBasedMerger):

    def __init__(self, svd_path, svd_compress_factor, principal_angle_threshold):
        super().__init__()

        self.svd_path = svd_path
        self.svd_compress_factor = svd_compress_factor
        self.principal_angle_threshold = principal_angle_threshold

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

        multi_task_vector = sum_svd_principal_angles(
            ref_state_dict=copy.deepcopy(base_model.state_dict()),
            svd_dict=svd_dict,
            principal_angle_threshold=self.principal_angle_threshold,
        )

        merged_encoder = copy.deepcopy(base_model)

        merged_encoder = apply_dict_to_model(
            multi_task_vector,
            merged_encoder,
        )

        return merged_encoder
    
    def merge_from_svd_dict(self, base_model, svd_dict):
        multi_task_vector = sum_svd_principal_angles(
            ref_state_dict=copy.deepcopy(base_model.state_dict()),
            svd_dict=svd_dict,
            principal_angle_threshold=self.principal_angle_threshold,
        )

        merged_encoder = copy.deepcopy(base_model)

        # pylogger.info(f"Applying multi-task vector to base model")
        merged_encoder = apply_dict_to_model(
            multi_task_vector,
            merged_encoder,
        )

        return merged_encoder
