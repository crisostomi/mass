import logging
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torchmetrics
from torch.optim import Optimizer

from nn_core.common import PROJECT_ROOT
from nn_core.model_logging import NNLogger

from mass.data.datamodule import MetaData
from mass.data.datasets.common import maybe_dictionarize
from mass.utils.utils import torch_load, torch_save

pylogger = logging.getLogger(__name__)


class ImageClassifier(pl.LightningModule):
    logger: NNLogger

    def __init__(
        self, encoder, classifier, metadata: Optional[MetaData] = None, *args, **kwargs
    ) -> None:
        super().__init__()

        # Populate self.hparams with args and kwargs automagically!
        # We want to skip metadata since it is saved separately by the NNCheckpointIO object.
        # Be careful when modifying this instruction. If in doubt, don't do it :]
        self.save_hyperparameters(logger=False, ignore=("metadata",))

        self.metadata = metadata
        self.num_classes = classifier.out_features

        metric = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.num_classes, top_k=1
        )
        self.train_acc = metric.clone()
        self.val_acc = metric.clone()
        self.test_acc = metric.clone()

        self.encoder = encoder
        self.classification_head = classifier

        self.log_fn = lambda metric, val: self.log(
            metric, val, on_step=False, on_epoch=True
        )

        self.finetuning_accuracy = None

    def set_encoder(self, encoder: torch.nn.Module):
        """Set the encoder of the model.

        Args:
            encoder (torch.nn.Module): The new encoder to set.
        """
        self.encoder = encoder

    def set_head(self, head: torch.nn.Module):
        """Set the classification head of the model.

        Args:
            head (torch.nn.Module): The new classification head to set.
        """
        self.classification_head = head

    def set_metrics(self, num_classes):

        self.num_classes = num_classes

        metric = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, top_k=1
        )

        self.train_acc = metric.clone()
        self.val_acc = metric.clone()
        self.test_acc = metric.clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Method for the forward pass.

        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.

        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.
        """
        embeddings = self.encoder(x)

        logits = self.classification_head(embeddings)

        return logits

    def _step(self, batch: Dict[str, torch.Tensor], split: str) -> Mapping[str, Any]:
        batch = maybe_dictionarize(batch, self.hparams.x_key, self.hparams.y_key)

        x = batch[self.hparams.x_key]
        gt_y = batch[self.hparams.y_key]

        logits = self(x)

        loss = F.cross_entropy(logits, gt_y)
        preds = torch.softmax(logits, dim=-1)

        metrics = getattr(self, f"{split}_acc")
        metrics.update(preds, gt_y)

        self.log_fn(f"acc/{split}/{self.task_name}", metrics)
        self.log_fn(f"loss/{split}/{self.task_name}", loss)

        return {"logits": logits.detach(), "loss": loss}

    def freeze_head(self):
        self.classification_head.weight.requires_grad_(False)
        self.classification_head.bias.requires_grad_(False)

    def training_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        return self._step(batch=batch, split="train")

    def validation_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        return self._step(batch=batch, split="val")

    def test_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        return self._step(batch=batch, split="test")

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Tuple[Sequence[Optimizer], Sequence[Any]]]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Return:
            Any of these 6 options.
            - Single optimizer.
            - List or Tuple - List of optimizers.
            - Two lists - The first list has multiple optimizers, the second a list of LR schedulers (or lr_dict).
            - Dictionary, with an 'optimizer' key, and (optionally) a 'lr_scheduler'
              key whose value is a single LR scheduler or lr_dict.
            - Tuple of dictionaries as described, with an optional 'frequency' key.
            - None - Fit will run without any optimizer.
        """
        opt = hydra.utils.instantiate(self.hparams.optimizer, params=self.parameters())
        if "lr_scheduler" not in self.hparams:
            return [opt]
        scheduler = hydra.utils.instantiate(self.hparams.lr_scheduler, optimizer=opt)
        return [opt], [scheduler]

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f"Saving image classifier to {filename}")
        torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f"Loading image classifier from {filename}")
        return torch_load(filename)

    def set_task(self, task_name):
        self.task_name = task_name

    def set_finetuning_accuracy(self, finetuning_accuracy):
        self.finetuning_accuracy = finetuning_accuracy

    def on_test_epoch_end(self):

        if self.finetuning_accuracy is not None:
            accuracy = (
                self.trainer.callback_metrics[f"acc/test/{self.task_name}"].cpu().item()
            )

            normalized_acc = accuracy / self.finetuning_accuracy

            self.log_fn(f"normalized_acc/test/{self.task_name}", normalized_acc)
