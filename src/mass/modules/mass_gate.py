from collections import defaultdict
from typing import List, Optional
import torch
import numpy as np
import wandb

import torch.nn as nn

from mass.utils.routing_methods import (
    compute_residual_norm,
)
from mass.utils.plots import plot_interactive_coefficients_std

import logging

pylogger = logging.getLogger(__name__)


class MassGate(nn.Module):
    def __init__(
        self,
        name,
        module,
        routing_weights,
        dataset_names,
        routing_mode,
        max_num_tasks_to_select,
        temperature: int = 1,
        threshold: float = 0.2,
        debug: Optional[bool] = False,
        visual: bool = True,  # cls or mean
    ):
        super().__init__()

        self.name = name
        self.module = module
        self.routing_mode = routing_mode
        self.threshold = threshold
        self.temperature = temperature
        self.dataset_names = dataset_names
        self.max_num_tasks_to_select = min(max_num_tasks_to_select, len(dataset_names))

        v, s, cov = routing_weights

        self.register_buffer("routing_weights", v)
        self.register_buffer("routing_singular_values", s)
        self.register_buffer("covariance", cov)

        self.debug = debug

        self.select_token = lambda x: (x.mean(dim=1) if not visual else x[0, :, :])

        self.dataset_idx_to_name = {i: name for i, name in enumerate(dataset_names)}

        self.output = None

        self.layer_residuals_to_log = defaultdict(list)
        self.layer_accuracy_to_log = defaultdict(list)
        self.layer_impact_log = defaultdict(list)
        self.norms_to_log = []

    def forward(self, x: torch.Tensor, bsz: int = None):
        """
        The overall forward pass of the router.
        Groups images based on selected task vectors.
        """
        dataset_coeffs = self._compute_tv_coefficients(x, bsz=bsz)  # (B, num_datasets)

        # for each sample, select the datasets such that the router coeffs surpass the threshold (B, num_datasets)
        selected_dataset_idxs: List[List[int]] = self._filter_datasets(dataset_coeffs)

        # group images that share the same selected datasets, e.g. {('Cars', 'MNIST'): [0, 1, 4, 5], ('GTSRB',): [2, 3], ..}
        dataset_group_to_samples = self.group_images_by_selected_datasets(
            selected_dataset_idxs
        )
        self.output = selected_dataset_idxs, dataset_coeffs, dataset_group_to_samples

        return self.module(x)

    def _compute_tv_coefficients(self, images, bsz: int = None) -> torch.Tensor:

        norms = self._compute_logits(images, bsz=bsz)  # (B, num_datasets)

        tv_coefficients = self._logits_to_coefficients(norms)

        # Log task predictions if debug is enabled
        if self.debug:
            task_predictions = torch.argmax(tv_coefficients, dim=1)  # (B,)
            self.layer_accuracy_to_log[self.name].append(task_predictions.cpu())

        return tv_coefficients

    def _compute_logits(self, x, bsz: int = None) -> torch.Tensor:
        # pylogger.info(f"Layer name {self.name}, input shape {x.shape}")

        if bsz and len(x.shape) == 2:
            patches = x.shape[0] // bsz
            x = x.view(patches, bsz, *x.shape[1:])

        pylogger.info(f"Token shape prior selection {x.shape}")
        x = self.select_token(x)
        pylogger.info(f"Layer name {self.name}, selected token shape {x.shape}")

        norms = compute_residual_norm(
            x,
            v=self.routing_weights,
            s=self.routing_singular_values,
            cov=self.covariance,
            norm="l2" if self.covariance is None else "mahalanobis",
        )

        # logging stuff
        if self.debug:
            if isinstance(self.norms_to_log, np.ndarray):
                # Convert back to list if it was converted to array during logging
                self.norms_to_log = self.norms_to_log.tolist()
            self.norms_to_log.append((norms.mean(dim=0)).cpu().numpy())

        return -norms

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
        else:
            raise NotImplementedError(
                f"Routing mode {self.routing_mode} is not implemented"
            )

        return tv_coefficients

    def _filter_datasets(self, tv_coefficients):
        selected_dataset_idxs = []

        for coeff in tv_coefficients:
            idxs = torch.where(coeff > self.threshold)[0].tolist()

            if len(idxs) > self.max_num_tasks_to_select and self.routing_mode == "topk":
                top_k = self.max_num_tasks_to_select
                _, idxs = torch.topk(coeff, k=top_k)

                idxs = idxs.tolist()

            if not idxs:

                top_k = 1  # for now top 1, i.e. argmax
                _, idxs = torch.topk(coeff, k=top_k)

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

    @property
    def weight(self):
        return self.module.weight

    def reset_to_log(self):
        self.norms_to_log = []
        self.layer_residuals_to_log = defaultdict(list)
        self.layer_accuracy_to_log = defaultdict(list)
        self.layer_impact_log = defaultdict(list)
