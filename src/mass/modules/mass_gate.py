from collections import defaultdict
from typing import List, Optional
import torch
import numpy as np
import wandb

import torch.nn as nn

from mass.utils.routing_methods import (
    compute_residual_norm,
)
from mass.utils.utils import (
    get_hook_fn,
    get_hook_fn_impact,
    get_routing_weights,
    is_supported_layer,
    router_key_from_layer,
    svd_key_from_layer,
    from_router_to_svd_dict_key,
)

from mass.utils.plots import (
    plot_interactive_coefficients_std,
    create_interactive_layer_task_residual_plot,
    create_interactive_layer_task_accuracy_plot,
    create_interactive_layer_impact_bar_chart,
)

import logging

pylogger = logging.getLogger(__name__)


class MassGate(nn.Module):
    def __init__(
        self,
        module,
        routing_weights,
        dataset_names,
        routing_mode,
        max_num_tasks_to_select,
        temperature: int = 1,
        threshold: float = 0.2,
        debug_residuals: Optional[bool] = False,
        debug_layer_impact: Optional[bool] = False,
        token_selection="mean",  # cls or mean
    ):
        super().__init__()
        
        self.module = module
        self.routing_mode = routing_mode
        self.threshold = threshold
        self.temperature = temperature
        self.dataset_names = dataset_names
        self.max_num_tasks_to_select = min(max_num_tasks_to_select, len(dataset_names))

        v, s, u = routing_weights
        
        self.register_buffer("routing_weights", v)
        self.register_buffer("routing_singular_values", s)
        self.register_buffer("routing_left_weights", u)

        self.debug_residuals = debug_residuals
        self.debug_layer_impact = debug_layer_impact

        # TODO: check if it works with LLMs
        self.select_token = lambda x: (
            x[0, :] if token_selection == "cls" else x.mean(dim=0)
        )  # CLS token or mean pooling = 'cls'
        
        self.dataset_idx_to_name = {
            i: name for i, name in enumerate(dataset_names)
        }

        self.output = None

        self.layer_residuals_to_log = defaultdict(list)
        self.layer_accuracy_to_log = defaultdict(list)
        self.layer_impact_log = defaultdict(list)
        self.norms_to_log = []

            
    def forward(self, x: torch.Tensor):
        """
        The overall forward pass of the router.
        Groups images based on selected task vectors.
        """

        pylogger.info(f"MassGate forward, input shape: {x.shape}")
        dataset_coeffs = self._compute_tv_coefficients(x)

        pylogger.info(f"Dataset coefficients: {dataset_coeffs}")
        # for each sample, select the datasets such that the router coeffs surpass the threshold (B, num_datasets)
        selected_dataset_idxs: List[List[int]] = self._filter_datasets(dataset_coeffs)

        # group images that share the same selected datasets, e.g. {('Cars', 'MNIST'): [0, 1, 4, 5], ('GTSRB',): [2, 3], ..}
        dataset_group_to_samples = self.group_images_by_selected_datasets(
            selected_dataset_idxs
        )

        pylogger.info(f"Dataset group to samples: {dataset_group_to_samples}")
        self.output = selected_dataset_idxs, dataset_coeffs, dataset_group_to_samples
        return self.module(x)

    def _compute_tv_coefficients(self, images):

        norms = self._compute_logits(images)

        tv_coefficients = self._logits_to_coefficients(norms)

        return tv_coefficients
    
    def _compute_logits(self, x) -> torch.Tensor:
        x = self.select_token(x)

        norms = compute_residual_norm(
            x, v=self.routing_weights, s=self.routing_singular_values, 
        )

        # logging stuff
        # if self.debug_residuals:
        #     self.log_layer_residuals()

        # self.norms_to_log.append((norms.mean(dim=0)).cpu().numpy())

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
            raise NotImplementedError(f"Routing mode {self.routing_mode} is not implemented")

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

                pylogger.info("Using the argmax, no coefficients above threshold")
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
    
    # Logging functions
    
    # def log_layer_residuals(self):

    #     for layer_key, features in self.middle_features.items():
    #         try:
    #             x_layer = features[0].to(self.device)
    #             v, s, _ = get_routing_weights(
    #                 self.svd_dicts,
    #                 layer=from_router_to_svd_dict_key(layer_key),
    #                 get_sigma=True,
    #                 get_u=False,
    #             )

    #             residual = compute_residual_norm(x_layer, v=v, s=s, norm=self.norm)
    #             layer_pred_task = torch.argmin(residual, dim=1)  # (B, )
    #             avg_vector = residual.mean(dim=0).cpu().numpy()

    #             self.layer_residuals_to_log[layer_key].append(avg_vector)
    #             self.layer_accuracy_to_log[layer_key].append(layer_pred_task)

    #         except Exception as e:
    #             pylogger.warning(
    #                 f"Skipping logging for layer {layer_key} due to error: {e}"
    #             )

    # def logging(self, logger, current_task):
    #     self.norms_to_log = np.array(self.norms_to_log)

    #     mean_coeffs = self.norms_to_log.mean(axis=0)
    #     std_coeffs = self.norms_to_log.std(axis=0)

    #     dataset_names = list(self.dataset_names)

    #     fig_std = plot_interactive_coefficients_std(
    #         mean_coeffs, std_coeffs, dataset_names
    #     )

    #     logger.experiment.log(
    #         {
    #             f"norms/{current_task}": wandb.Plotly(fig_std),
    #         }
    #     )

    #     if self.debug_residuals:
    #         fig = create_interactive_layer_task_residual_plot(
    #             self.layer_residuals_to_log, dataset_names
    #         )

    #         logger.experiment.log(
    #             {f"average_residuals/{current_task}": wandb.Plotly(fig)}
    #         )

    #         fig = create_interactive_layer_task_accuracy_plot(
    #             self.layer_accuracy_to_log,
    #             dataset_names.index(current_task),
    #             dataset_names,
    #         )

    #         logger.experiment.log(
    #             {f"layer_task_accuracy/{current_task}": wandb.Plotly(fig)}
    #         )

    #     if self.debug_layer_impact:

    #         fig = create_interactive_layer_impact_bar_chart(self.layer_impact_log)

    #         logger.experiment.log({f"layer_impact/{current_task}": wandb.Plotly(fig)})

    #     self.reset_log_stats()

    # def reset_log_stats(self):
    #     self.norms_to_log = []
    #     self.layer_residuals_to_log = defaultdict(list)
    #     self.layer_accuracy_to_log = defaultdict(list)
    #     self.layer_impact_log = defaultdict(list)

