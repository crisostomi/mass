import copy
from typing import Dict, List, OrderedDict, Union

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from tvp.task_vectors.aggregation import slerp, spherical_weighted_average

import logging

from tvp.task_vectors.task_singular_vectors import isotropic_sum, sum_svd

pylogger = logging.getLogger(__name__)


class Aggregator:
    def __init__(self, **kwargs):
        pass

    def __call__(self, task_vectors):
        return self.aggregate(task_vectors)

    def aggregate(self, task_vectors):
        pass


class SumAggregator(Aggregator):
    def __init__(self, mean=False, rescaling=1.0, **kwargs):
        super(SumAggregator, self).__init__(**kwargs)

        self.mean = mean
        self.rescaling = rescaling

    def aggregate(self, task_vectors):
        if isinstance(task_vectors, List):
            task_vectors = torch.stack(task_vectors)

        multi_task_vector = torch.sum(task_vectors, dim=0)

        if self.mean:
            multi_task_vector /= len(task_vectors)

        return multi_task_vector * self.rescaling


class WeightedAggregator(Aggregator):
    def __init__(self, zeroshot_model=None, normalize_coefficients=False, **kwargs):
        super(WeightedAggregator, self).__init__(**kwargs)
        self.zeroshot_model = copy.deepcopy(zeroshot_model)
        self.normalize_coefficients = normalize_coefficients
        self.tmp_model = copy.deepcopy(zeroshot_model)

    def aggregate(
        self,
        task_dicts: Union[Dict, List],
        scaling_coeff: float,
        device: str = "cuda",
    ):
        """

        task_dicts: either a dict {dataset: task_dict} or a list of task_dicts
        """

        self.tmp_model = copy.deepcopy(self.zeroshot_model)

        model_device = next(self.zeroshot_model.parameters()).device.type

        # Ensure that all task vectors are on the same device as the model
        assert (
            device == model_device
        ), "The target model and the task vector are not on the same device!"

        new_state_dict = copy.deepcopy(self.zeroshot_model.state_dict())

        # create multi task vector
        multi_task_vector = copy.deepcopy(task_dicts[0])

        for key in multi_task_vector:
            multi_task_vector[key] = torch.zeros_like(multi_task_vector[key]).to(device)

        for i, task_dict in enumerate(task_dicts):

            for key in task_dict:
                new_key = key.replace("encoder.", "")

                if new_key not in new_state_dict:
                    pylogger.warning(
                        f"[WARNING] Key {key} is present in the task vector but not in the model"
                    )
                    continue

                multi_task_vector[new_key] += task_dict[key].to(device)

        # apply
        for key in multi_task_vector:
            new_state_dict[key] += scaling_coeff * multi_task_vector[key]

        self.tmp_model.load_state_dict(
            new_state_dict, strict=False
        )  # Allow missing keys

        return self.tmp_model.cuda()


# class WeightedAggregatorWithSVD(Aggregator):

#     def __init__(self, target_model=None, **kwargs):
#         super(WeightedAggregatorWithSVD, self).__init__(**kwargs)

#     def aggregate(self, svd_dict):

#         return sum_svd(svd_dict)


class TaskSingularVectorAggregator(Aggregator):
    def __init__(
        self,
        zeroshot_model,
        normalize_coefficients=False,
    ):
        super().__init__()

        self.zeroshot_model = copy.deepcopy(zeroshot_model)
        self.tmp_model = copy.deepcopy(self.zeroshot_model)
        self.normalize_coefficients = normalize_coefficients

    def aggregate(self, svd_dicts, coefficients: torch.Tensor = None):

        self.tmp_model = copy.deepcopy(self.zeroshot_model)

        new_state_dict = copy.deepcopy(self.zeroshot_model.state_dict())

        delta_aggregated_state_dict = sum_svd(
            copy.deepcopy(self.zeroshot_model.state_dict()), svd_dicts
        )

        for key in delta_aggregated_state_dict:
            new_state_dict[key] += delta_aggregated_state_dict[key]

        self.tmp_model.load_state_dict(new_state_dict)

        return self.tmp_model.cuda()


class IsotropicAggregator(Aggregator):

    def __init__(self, zeroshot_model):
        super().__init__()

        self.zeroshot_model: OrderedDict = zeroshot_model
        self.tmp_model = copy.deepcopy(self.zeroshot_model)

    def aggregate(self, svd_dicts, coefficients: torch.Tensor):

        new_state_dict = copy.deepcopy(self.zeroshot_model.state_dict())

        delta_aggregated_state_dict = isotropic_sum(
            copy.deepcopy(self.zeroshot_model.state_dict()), svd_dicts
        )

        for key in delta_aggregated_state_dict:
            new_state_dict[key] += delta_aggregated_state_dict[key]

        self.tmp_model.load_state_dict(new_state_dict)

        return self.tmp_model.cuda()
