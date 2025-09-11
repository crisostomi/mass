import copy
from mass.merger.merger import TaskVectorBasedMerger
from mass.modules.encoder import ImageEncoder
from mass.utils.utils import apply_dict_to_model, compute_task_dict, print_memory
from mass.task_vectors.task_singular_vectors import (
    get_svd_dict,
    sum_svd,
)

import torch


class DummyMerger(TaskVectorBasedMerger):

    def __init__(self):
        super().__init__()

    def merge(self, base_model, finetuned_models):

        return base_model
