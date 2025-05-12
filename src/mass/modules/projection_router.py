from collections import defaultdict
import torch
import numpy as np
import wandb

from mass.modules.router import AbstractRouter
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


class ProjectionRouter(AbstractRouter):
    def __init__(
        self,
        name,
        model_name,
        encoder,
        dataset_names,
        svd_dict,
        layer_to_hook,  # e.g. "attn"
        layer_num_to_hook,  # e.g. <= 11 for ViT-B-(32,16) <= 23 for ViT-L-14
        hook_type,
        threshold,
        temperature,
        routing_mode: str,
        norm,
        routing_weights,
        debug_residuals,
        debug_layer_impact,
        openclip_cachedir=None,
        max_num_tasks_to_select=2,
        routing_on="residual",
        cache_dir=None,
        keep_lang=False,
        device=None,
        token_selection="mean",  # cls or mean
        **kwargs,
    ):
        super().__init__(
            name=name,
            model_name=model_name,
            dataset_names=dataset_names,
            encoder=encoder,
            threshold=threshold,
            temperature=temperature,
            routing_mode=routing_mode,
            openclip_cachedir=openclip_cachedir,
            max_num_tasks_to_select=max_num_tasks_to_select,
            routing_on=routing_on,
            cache_dir=cache_dir,
            keep_lang=keep_lang,
            device=device,
        )

        self.layer = layer_to_hook
        self.layer_num = layer_num_to_hook
        self.hook = hook_type

        if self.layer == "all":
            self.layer = ["attn", "mlp"]
        if self.layer_num == "all":
            self.layer_num = list(range(24)) if model_name == "ViT-L-14" else list(range(12))
        if isinstance(self.layer, str):
            self.layer = [self.layer]
        if isinstance(self.layer_num, int):
            self.layer_num = [self.layer_num]

        self.svd_key = [
            svd_key_from_layer(layer_to_hook, layer_num_to_hook)
            for layer_to_hook in self.layer
            for layer_num_to_hook in self.layer_num
        ]
        self.feature_key = [
            router_key_from_layer(layer_to_hook, layer_num_to_hook)
            for layer_to_hook in self.layer
            for layer_num_to_hook in self.layer_num
        ]

        for svd_key in self.svd_key:
            if routing_weights is None:
                routing_weights, sigma, u = get_routing_weights(
                    svd_dict, svd_key, get_sigma=True, get_u=True
                )
            # TODO: fix this whe use the use_fixed_compress_bl_bla
            # else:
            #     routing_weights, sigma, u = routing_weights

            self.register_buffer(
                f"routing_weights_{svd_key.replace('in_proj_weight', '').replace('c_fc.weight', '').replace('.', '')}",
                routing_weights,
            )
            self.register_buffer(
                f"routing_singular_values_{svd_key.replace('in_proj_weight', '').replace('c_fc.weight', '').replace('.', '')}",
                sigma,
            )
            self.register_buffer(
                f"routing_left_weights_{svd_key.replace('in_proj_weight', '').replace('c_fc.weight', '').replace('.', '')}",
                u,
            )

        self.debug_residuals = debug_residuals
        self.debug_layer_impact = debug_layer_impact

        self.select_token = lambda x: (
            x[0, :] if token_selection == "cls" else x.mean(dim=0)
        )  # CLS token or mean pooling = 'cls'

        if self.debug_residuals:
            self.svd_dicts = svd_dict

        self.norm = norm

        self.layer_residuals_to_log = defaultdict(list)
        self.layer_accuracy_to_log = defaultdict(list)
        self.layer_impact_log = defaultdict(list)
        self.norms_to_log = []

        hooked = False
        for name, module in self.named_modules():

            if not is_supported_layer(name):
                continue

            if name in self.feature_key or self.debug_residuals:
                hooked = True
                pylogger.info(f"Registering hook for {name}")
                module.register_forward_hook(get_hook_fn(self, name, self.hook))

            if self.debug_layer_impact:
                pylogger.info(f"Registering hook for {name} for impact logging")
                module.register_forward_hook(get_hook_fn_impact(self, name))

        assert hooked, f"Layer {layer_to_hook} not found in model."

    def _compute_logits(self, images) -> torch.Tensor:
        with torch.no_grad():
            _ = self.encoder(images)
        # (L, B, D)

        norms = None
        for feature_key in self.feature_key:
            x = self.middle_features[feature_key].to(self.device)
            x = self.select_token(x)

            # dynamically grab the buffers you registered in __init__
            # print("*" * 20)
            # print(f"routing_weights_{feature_key.replace('.', '')}")
            # print("*" * 20)
            v = getattr(
                self, f"routing_weights_{feature_key.replace('transformer','').replace('.', '')}"
            )
            s = getattr(
                self,
                f"routing_singular_values_{feature_key.replace('transformer','').replace('.', '')}",
            )

            # compute this layer’s residual norm
            this_norm = compute_residual_norm(x, v=v, s=s, norm=self.norm)
            # on first iter, create norms tensor of same shape as this_norm
            if norms is None:
                norms = this_norm
            else:
                norms = norms + this_norm

        #     norms += compute_residual_norm(
        #     x, v=self.routing_weights, s=self.routing_singular_values, norm=self.norm
        # )

        # logging stuff
        if self.debug_residuals:
            self.log_layer_residuals()

        self.norms_to_log.append((norms.mean(dim=0)).cpu().numpy())

        return -norms

    def log_layer_residuals(self):

        for layer_key, features in self.middle_features.items():
            try:
                x_layer = features[0].to(self.device)
                v, s, _ = get_routing_weights(
                    self.svd_dicts,
                    layer=from_router_to_svd_dict_key(layer_key),
                    get_sigma=True,
                    get_u=False,
                )

                residual = compute_residual_norm(x_layer, v=v, s=s, norm=self.norm)
                layer_pred_task = torch.argmin(residual, dim=1)  # (B, )
                avg_vector = residual.mean(dim=0).cpu().numpy()

                self.layer_residuals_to_log[layer_key].append(avg_vector)
                self.layer_accuracy_to_log[layer_key].append(layer_pred_task)

            except Exception as e:
                pylogger.warning(f"Skipping logging for layer {layer_key} due to error: {e}")

    def logging(self, logger, current_task):
        self.norms_to_log = np.array(self.norms_to_log)

        mean_coeffs = self.norms_to_log.mean(axis=0)
        std_coeffs = self.norms_to_log.std(axis=0)

        dataset_names = list(self.dataset_names)

        fig_std = plot_interactive_coefficients_std(mean_coeffs, std_coeffs, dataset_names)

        logger.experiment.log(
            {
                f"norms/{current_task}": wandb.Plotly(fig_std),
            }
        )

        if self.debug_residuals:
            fig = create_interactive_layer_task_residual_plot(
                self.layer_residuals_to_log, dataset_names
            )

            logger.experiment.log({f"average_residuals/{current_task}": wandb.Plotly(fig)})

            fig = create_interactive_layer_task_accuracy_plot(
                self.layer_accuracy_to_log,
                dataset_names.index(current_task),
                dataset_names,
            )

            logger.experiment.log({f"layer_task_accuracy/{current_task}": wandb.Plotly(fig)})

        if self.debug_layer_impact:

            fig = create_interactive_layer_impact_bar_chart(self.layer_impact_log)

            logger.experiment.log({f"layer_impact/{current_task}": wandb.Plotly(fig)})

        self.reset_log_stats()

    def reset_log_stats(self):
        self.norms_to_log = []
        self.layer_residuals_to_log = defaultdict(list)
        self.layer_accuracy_to_log = defaultdict(list)
        self.layer_impact_log = defaultdict(list)


class BatchedProjectionRouter(ProjectionRouter):
    def __init__(
        self,
        name,
        model_name,
        encoder,
        dataset_names,
        svd_dict,
        layer_to_hook,
        layer_num_to_hook,
        hook_type,
        threshold,
        temperature,
        routing_mode,
        norm,
        routing_weights,
        debug_residuals,
        debug_layer_impact,
        openclip_cachedir=None,
        max_num_tasks_to_select=2,
        routing_on="residual",
        cache_dir=None,
        keep_lang=False,
        device=None,
        token_selection="mean",
        **kwargs,
    ):
        super().__init__(
            name=name,
            model_name=model_name,
            encoder=encoder,
            dataset_names=dataset_names,
            svd_dict=svd_dict,
            layer_to_hook=layer_to_hook,
            layer_num_to_hook=layer_num_to_hook,
            hook_type=hook_type,
            threshold=threshold,
            temperature=temperature,
            routing_mode=routing_mode,
            norm=norm,
            routing_weights=routing_weights,
            debug_residuals=debug_residuals,
            debug_layer_impact=debug_layer_impact,
            openclip_cachedir=openclip_cachedir,
            max_num_tasks_to_select=max_num_tasks_to_select,
            routing_on=routing_on,
            cache_dir=cache_dir,
            keep_lang=keep_lang,
            device=device,
            token_selection=token_selection,
            **kwargs,
        )

    def _compute_logits(self, images) -> torch.Tensor:
        with torch.no_grad():
            _ = self.encoder(images)

        x = self.middle_features[self.feature_key].to(self.device)
        x = self.select_token(x)

        if x.dim() == 2:
            B, D = x.shape  # Handling the batch dimension correctly
        else:
            L, B, D = x.shape  # Expected shape (L, B, D)

        norms = compute_residual_norm(
            x, v=self.routing_weights, s=self.routing_singular_values, norm=self.norm
        )

        # logging stuff
        if self.debug_residuals:
            self.log_layer_residuals()

        self.norms_to_log.append((norms.mean(dim=0)).cpu().numpy())

        norms = -norms.mean(dim=0)

        if norms.dim() == 1:
            norms = norms.unsqueeze(0)  # Ensure it's 2D before repeating

        return norms.repeat(B, 1)
