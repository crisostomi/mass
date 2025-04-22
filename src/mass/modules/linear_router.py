import copy
import torch
import torch.nn.functional as F
import torch.nn as nn

from mass.modules.router import AbstractRouter

import logging

pylogger = logging.getLogger(__name__)


class LinearRouter(AbstractRouter):
    def __init__(
        self,
        name,
        model_name,
        encoder,
        dataset_names,
        threshold,
        temperature,
        embedding_dims,
        hidden_dim,
        dropout_prob,
        routing_mode: str,  # top1 or weighted
        cfg,
        openclip_cachedir=None,
        routing_on="residual",
        cache_dir=None,
        keep_lang=False,
        device=None,
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
            routing_on=routing_on,
            cache_dir=cache_dir,
            keep_lang=keep_lang,
            device=device,
        )

        self.mlp_router = torch.nn.Sequential(
            nn.Linear(embedding_dims, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_dim, len(dataset_names)),
        )

    def parameters(self):
        return self.mlp_router.parameters()

    def predict_task_train(self, images) -> torch.Tensor:
        with torch.no_grad():
            embedding = self.encoder(images)

        embedding = embedding.detach().requires_grad_(True)

        logits = self.mlp_router(embedding)
        return logits

    def predict_task(self, images):
        return self._compute_logits(images)

    def _compute_logits(self, images) -> torch.Tensor:
        with torch.no_grad():
            embedding = self.encoder(images)
        logits = self.mlp_router(embedding)
        return logits

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        return self.mlp_router.state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )

    def load_state_dict(self, state_dict, strict):
        self.mlp_router.load_state_dict(state_dict, strict)
