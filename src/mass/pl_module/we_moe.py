import functools

from hydra.utils import instantiate
import logging
from typing import Any, Generic, List, cast  # noqa: F401

import lightning.fabric.wrappers
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

from mass.modules.encoder import ImageEncoder
from mass.modules.we_moe import WeightEnsemblingMoE
from mass.pl_module.image_multihead_classifier import MultiHeadImageClassifier

from open_clip import CLIP

pylogger = logging.getLogger(__name__)


class WeightEnsemblingMoEAlgorithm(MultiHeadImageClassifier):
    """
    Algorithm for fusing models using Weight Ensembling Mixture of Experts (MoE).

    This class provides methods for constructing the MoE model, performing test-time adaptation,
    and running the fusion process.

    Attributes:
        _fabric (L.Fabric): The fabric for distributed training.
        modelpool (ModelPool): The pool of models to be fused.
    """

    def __init__(
        self,
        pretrained_model,
        finetuned_models,
        classification_heads,
        checkpoint=False,
        save_checkpoint=False,
        router_hidden_layers=2,
        init_lambda=0.3,
        batch_reduce=True,
        use_grad_accumulate=True,
        model_path: str = None,
        **kwargs: Any,
    ):

        pylogger.info(
            "Fusing models using WeightEnsembling Mixture of Experts modules."
        )
        self.aggregator = instantiate(
            self.hparams.aggregator, zeroshot_model=pretrained_model.cuda()
        )
        moe_model = self.construct_moe_model(pretrained_model, finetuned_models)

        if self.use_checkpoint:
            pylogger.info(
                f"load checkpoint from {self.checkpoint_path}, test-time adaptation will be skipped."
            )
            self.load_checkpoint(moe_model, self.checkpoint_path)
        else:
            moe_model = self.test_time_adaptation(moe_model)
            if self.write_checkpoint:
                pylogger.info(f"save checkpoint to {self.save_checkpoint_path}")
                self.save_checkpoint(moe_model, self.save_checkpoint_path)

            if lightning.fabric.wrappers.is_wrapped(moe_model):
                moe_model = lightning.fabric.wrappers._unwrap_objects(moe_model)

        moe_model.batch_reduce = False
        super().__init__(moe_model, classification_heads)

    def load_checkpoint(self, model: Any, checkpoint: Any):
        """
        Load the checkpoint file.

        Args:
            model: The model to load the checkpoint into.
            checkpoint: The path to the checkpoint file.
        """
        state = {"model": model}
        self._fabric.load(checkpoint, state)

    def save_checkpoint(self, model: Any, checkpoint: Any):
        """
        Save the checkpoint file.

        Args:
            model: The model to save the checkpoint from.
            checkpoint: The path to the checkpoint file.
        """
        self._fabric.save(checkpoint, {"model": model})

    def construct_moe_model(self, pretrained_model: ImageEncoder, finetuned_models: List[ImageEncoder]) -> WeightEnsemblingMoE:
        """
        Construct the Mixture of Experts (MoE) model using the models in the model pool.

        Returns:
            WeightEnsemblingMoE: The constructed MoE model.
        """

        # Merge the models using task arithmetic
        moe_model = self.aggregator.merge(
            pretrained_model,
            finetuned_models
        )

        # Up-scale MLP modules
        # TODO: what are these models? (i mean which classes)
        base_encoder = pretrained_model.model.visual
        moe_encoder = moe_model.model.visual
        expert_encoders = [m.model.visual for m in expert_models]

        # TODO: iterate over named_modules same problem of SMILE?
        num_layers = len(base_encoder.layers)
        for layer_idx in range(num_layers):
            base_mlp = base_encoder.layers[layer_idx].mlp
            expert_mlps = [e.layers[layer_idx].mlp for e in expert_encoders]

            moe_encoder.layers[layer_idx].mlp = WeightEnsemblingMoE(
                hidden_size=base_encoder.config.hidden_size,
                base_model=base_mlp,
                expert_models=expert_mlps,
                init_lambda=self.config.init_lambda,
                batch_first=True,  # For open_clip models this is False
                router_hidden_layers=self.config.router_hidden_layers,
                batch_reduce=self.config.batch_reduce,
            )

        return moe_model

    def on_test_time_adaptation_start(self):
        """
        Load the CLIP processor and construct the zero-shot classification head for each task.
        """
        self.setup_zero_shot_classification_head()

    @functools.cache
    def get_shuffled_test_loader_iter(self, tta_dataset: str) -> Iterator:
        """
        Get an iterator for the shuffled test data loader.

        Args:
            tta_dataset (str): The name of the test-time adaptation dataset.

        Returns:
            Iterator: An iterator for the shuffled test data loader.
        """
        dataset = self.modelpool.load_test_dataset(tta_dataset)
        dataset = CLIPDataset(dataset, processor=self.clip_processor)
        log.info("get_shuffled_test_loader_iter")
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
        loader = self.fabric.setup_dataloaders(loader)
        return iter(InfiniteDataLoader(loader))

    def compute_logits(self, module: Any, batch: Any, task: Any) -> Tensor:
        """
        Compute the logits for the given batch and task.

        Args:
            module: The model module.
            batch: The input batch.
            task: The task name.

        Returns:
            Tensor: The computed logits.
        """
        images, _ = batch
        text_embeds = self.zeroshot_weights[task]

        image_embeds = module(images)[1]
        image_embeds = self.visual_projection(image_embeds)

        # Normalize embeddings
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

        # Cosine similarity
        logits_per_text = (
            torch.matmul(text_embeds, image_embeds.t()) * self.logit_scale_exp
        )
        logits_per_image = logits_per_text.t()

        return logits_per_image

    def test_time_adaptation(self, module: WeightEnsemblingMoE) -> WeightEnsemblingMoE:
        """
        Perform test-time adaptation for the given module.

        Args:
            module (WeightEnsemblingMoE): The MoE module to adapt.

        Returns:
            WeightEnsemblingMoE: The adapted MoE module.
        """
        self.on_test_time_adaptation_start()

        # configure optimizer
        if self.config.optimizer == "adam":
            optimizer = torch.optim.Adam(
                [p for p in module.parameters() if p.requires_grad], lr=self.config.lr
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")

        if self._fabric is not None:
            module, optimizer = self._fabric.setup(module, optimizer)

        module.train()

        if self.config.get("fast_dev_run", False):
            pylogger.info("Running fast_dev_run, only one step")
            pbar = tqdm(
                range(1),
                "Test-time adaptation",
                dynamic_ncols=True,
            )
        else:
            pbar = tqdm(
                range(self.config.max_steps),
                "Test-time adaptation",
                dynamic_ncols=True,
            )
        for step_idx in pbar:
            if self.config.use_grad_accumulate:
                for task in self.modelpool.model_names:
                    with self.profile("data time"):
                        batch = next(self.get_shuffled_test_loader_iter(task))
                    with self.profile("forward pass"):
                        logits = self.compute_logits(module, batch, task)
                        assert (
                            logits.dim() == 2
                        ), f"Expected logits to be 2D, got {logits.dim()}"
                        loss = entropy_loss(logits)
                    # .backward() accumulates when .zero_grad() wasn't called
                    # this can save memory
                    with self.profile("backward pass"):
                        self._fabric.backward(loss, retain_graph=True)
            else:
                loss = 0
                for task in self.modelpool.model_names:
                    with self.profile("data time"):
                        batch = next(self.get_shuffled_test_loader_iter(task))
                    with self.profile("forward pass"):
                        logits = self.compute_logits(module, batch, task)
                        assert (
                            logits.dim() == 2
                        ), f"Expected logits to be 2D, got {logits.dim()}"
                        loss = loss + entropy_loss(logits)
                with self.profile("backward pass"):
                    self._fabric.backward(loss, retain_graph=True)

            with self.profile("optimizer step"):
                optimizer.step()
                optimizer.zero_grad()

        return module
