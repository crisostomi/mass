from typing import List, Tuple, Any
import torch
import abc

import logging

pylogger = logging.getLogger(__name__)


class AbstractRouter(torch.nn.Module, abc.ABC):
    def __init__(
        self,
        name,
        encoder,
        dataset_names,
        threshold,
        temperature,
        routing_mode: str,  # top1 or weighted
        max_num_tasks_to_select=2,
        routing_on: str = "residual",  # could also be 'sub_usage' or 'tv_usage'
        cache_dir=None,
        device=None,
        **kwargs,
    ):
        super().__init__()

        assert routing_mode in [
            "top1",
            "weighted",
            "topk",
        ], f"Invalid routing mode {routing_mode}"

        self.name = name
        self.device = device
        self.routing_on = routing_on
        self.routing_mode = routing_mode

        self.encoder = encoder

        self.middle_features = {}
        self.threshold = threshold
        self.temperature = temperature
        self.cache_dir = cache_dir

        self.dataset_names = dataset_names
        self.dataset_idx_to_name = {
            i: name for i, name in enumerate(self.dataset_names)
        }
        self.dataset_name_to_idx = {
            name: i for i, name in enumerate(self.dataset_names)
        }

        self.max_num_tasks_to_select = min(max_num_tasks_to_select, len(dataset_names))

    def forward(self, images: torch.Tensor) -> Tuple:
        """
        The overall forward pass of the router.
        Groups images based on selected task vectors.
        """

        dataset_coeffs = self._compute_tv_coefficients(images)

        # for each sample, select the datasets such that the router coeffs surpass the threshold (B, num_datasets)
        selected_dataset_idxs: List[List[int]] = self._filter_datasets(dataset_coeffs)

        # group images that share the same selected datasets, e.g. {('Cars', 'MNIST'): [0, 1, 4, 5], ('GTSRB',): [2, 3], ..}
        dataset_group_to_samples = self.group_images_by_selected_datasets(
            selected_dataset_idxs
        )

        return selected_dataset_idxs, dataset_coeffs, dataset_group_to_samples

    @abc.abstractmethod
    def _compute_logits(self, images):
        """
        Compute the routing weights based on intermediate features extracted from the images.
        This method must be implemented by subclasses.
        """
        pass

    def _compute_tv_coefficients(self, images):

        norms = self._compute_logits(images)

        # (B, num_datasets)
        tv_coefficients = self._logits_to_coefficients(norms)
        # pylogger.info(tv_coefficients)

        return tv_coefficients

    def _logits_to_coefficients(self, norms) -> torch.Tensor:
        """
        Transforms logits into probabilities.
        """

        if self.routing_mode == "top1":
            tv_coefficients = torch.zeros_like(norms)
            idx = torch.argmax(norms, dim=1)
            tv_coefficients[torch.arange(norms.shape[0]), idx] = 1.0
        elif self.routing_mode == "topk":

            mean = norms.mean(dim=1, keepdim=True)
            std = norms.std(dim=1, keepdim=True) + 1e-6
            standardized_norms = (norms - mean) / std
            tv_coefficients = torch.nn.functional.softmax(
                standardized_norms / self.temperature, dim=1
            )

        elif self.routing_mode == "weighted":
            tv_coefficients = torch.nn.functional.softmax(
                norms / self.temperature, dim=1
            )

        return tv_coefficients

    def predict_task_train(self, images):
        yield NotImplementedError(f"{self.name} router is not trainable")

    def predict_task(self, images) -> torch.Tensor:
        """
        Predict the task using a softmax on the computed routing weights.
        Returns the task probabilities.
        """
        norms = self._compute_logits(images)

        return self._logits_to_coefficients(norms)

    def _filter_datasets(self, tv_coefficients):
        selected_dataset_idxs = []

        for coeff in tv_coefficients:
            idxs = torch.where(coeff > self.threshold)[0].tolist()

            # only keep self.max_num_tasks_to_select tasks

            if len(idxs) > self.max_num_tasks_to_select and self.routing_mode == "topk":
                top_k = self.max_num_tasks_to_select
                topk_values, idxs = torch.topk(coeff, k=top_k)

                idxs = idxs.tolist()

            if not idxs:

                top_k = 1  # for now top 1, i.e. argmax

                pylogger.info("Using the argmax, no coefficients above threshold")
                topk_values, idxs = torch.topk(coeff, k=top_k)

                idxs = idxs.tolist()

            selected_dataset_idxs.append(idxs)

        return selected_dataset_idxs

    def group_images_by_selected_datasets(self, selected_dataset_idxs: List[List[int]]):
        """
        Group images that share the same selected datasets to be processed with the same task vector combination for efficiency
        """
        # Map from dataset group to samples
        dataset_group_to_samples = {}

        for sample_idx, selected_dataset_idxs_for_sample in enumerate(
            selected_dataset_idxs
        ):

            # get the names of the dataset group selected for the current sample, .e.g. ('Cars', 'MNIST')
            sample_selected_datasets = tuple(
                [
                    self.dataset_idx_to_name[idx]
                    for idx in selected_dataset_idxs_for_sample
                ]
            )

            # add the current sample to those assigned to this dataset group
            dataset_group_to_samples.setdefault(sample_selected_datasets, []).append(
                sample_idx
            )

        return dataset_group_to_samples

    def logging(self, logger, current_task):
        yield NotImplementedError(f"{self.name} router is not loggable")
