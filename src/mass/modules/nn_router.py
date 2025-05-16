import copy

from tqdm import tqdm

from torch import Tensor
import torch

import logging
from mass.data.datasets.common import maybe_dictionarize
from mass.data.datasets.registry import get_dataset
from mass.data.datasets.templates import get_dataset_to_label
from mass.modules.router import AbstractRouter
from mass.utils.utils import apply_dict_to_model

pylogger = logging.getLogger(__name__)


class NNRouter(AbstractRouter):
    def __init__(
        self,
        name,
        model_name,
        encoder,
        dataset_names,
        threshold,
        temperature,
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

        self.nn_head = None

        nn_heads = NNhead.build(
            self.encoder,
            cfg.eval_datasets,
            cfg.nn.module,
            cfg.nn.data.data_path,
            "cuda",
        )
        self._initialize_nnhead(nn_heads)

    def _initialize_nnhead(self, nn_heads):
        """
        Initialize the NNhead with the task vectors.
        """
        pylogger.info("Initializing NNhead.")

        if not isinstance(nn_heads, NNhead):
            raise ValueError(f"Expected NNhead, got {type(nn_heads)}")

        self.nn_head = nn_heads

    def _compute_logits(self, images):
        """
        Here we use the image embeddings (from the model's encode_image)
        and pass them through the NNhead to obtain a logit per task.
        """
        assert self.nn_head is not None, "NNhead not initialized."

        with torch.no_grad():
            features = self.encoder(images)  # shape: (B, D)

            logits = self.nn_head(features)  # shape: (B, tasks)

        return logits


class NNhead(torch.nn.Module):
    def __init__(
        self,
        normalize: bool,
        buffer_size: int,
        weights: torch.Tensor,
        method: str = "set_distance",
    ):
        super().__init__()
        self.normalize = normalize
        self.buffer_size = buffer_size
        self.method = method

        self.num_tasks = weights.shape[0] // buffer_size

        task_embeddings = weights.reshape(self.num_tasks, buffer_size, -1)
        task_embeddings /= task_embeddings.norm(dim=-1, keepdim=True)

        if method == "prototype":
            task_embeddings = task_embeddings.mean(dim=1)
            task_embeddings /= task_embeddings.norm(dim=-1, keepdim=True)

        self.register_buffer("task_embeddings", task_embeddings)

    def forward(self, inputs: Tensor) -> Tensor:
        """

        :param inputs: (B, D) tensor of image embeddings
        """
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)

        if self.method == "prototype":
            # task_embeddings shape: (num_tasks, D)
            # Output shape: (B, num_tasks)
            dists = torch.cdist(
                inputs, self.task_embeddings, p=2
            )  # shape: (B, num_tasks)
            return -dists

        elif self.method == "set_distance":
            # task_embeddings shape: (num_tasks, buffer_size, D)

            # We want the min distance from each input to each task's set of embeddings
            num_tasks, buff_size, D = self.task_embeddings.shape

            # Flatten all embeddings
            flattened_emb = self.task_embeddings.view(
                num_tasks * buff_size, D
            )  # (num_tasks*buffer_size, D)

            # Compute pairwise Euclidian distances: (B, D) vs (num_tasks*buffer_size, D) -> (B, num_tasks*buffer_size)
            dists = torch.cdist(inputs, flattened_emb, p=2)

            # Reshape so we can get a distance per task per input
            dists = dists.view(-1, num_tasks, buff_size)  # (B, num_tasks, buffer_size)

            # Minimum distance to each task's embedding set
            min_dists, _ = dists.min(dim=-1)  # (B, num_tasks)

            return -min_dists

        else:
            raise ValueError(f"Unknown method: {self.method}")

    @classmethod
    def build(cls, encoder, task_names, cfg, data_location, device):
        DATASET_TO_LABEL = get_dataset_to_label(task_names)

        # sort task_names by task label
        task_names = sorted(task_names, key=lambda x: DATASET_TO_LABEL[x])

        encoder.eval()
        encoder.to(device)

        pylogger.info("Building nearest neighbor classification head.")
        with torch.no_grad():
            weights = []

            for task_name in tqdm(task_names):

                dataset = get_dataset(
                    task_name, encoder.train_preprocess, location=data_location
                )
                batch = maybe_dictionarize(
                    next(iter(dataset.train_loader)), cfg.x_key, cfg.y_key
                )
                x = batch[cfg.x_key]

                encodings = encoder(x.to(device))  # Encode images with CLIP

                weights.append(encodings)

        # Stack the weights into a single tensor
        # (T*B, D)
        nn_weights = torch.cat(weights, dim=0).detach().to(device)

        classification_head = cls(
            normalize=True, weights=nn_weights, buffer_size=x.shape[0]
        )

        return classification_head
