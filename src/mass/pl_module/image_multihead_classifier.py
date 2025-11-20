import pytorch_lightning as pl
import torch
from typing import Any, Dict, Mapping, Optional

import torch.nn.functional as F
import torchmetrics

from mass.data.datamodule import MetaData
from mass.data.dataset import maybe_dictionarize
from mass.utils.utils import torch_load, torch_save


class MultiHeadImageClassifier(pl.LightningModule):
    def __init__(
        self,
        moe_model,
        classification_heads,
        custom_logger: Optional[Any] = None,
    ):
        super().__init__()

        self.moe_model = moe_model
        if self.moe_model is not None:
            self.train_preprocess = self.moe_model.train_preprocess
            self.val_preprocess = self.moe_model.val_preprocess

        self.classification_heads = torch.nn.ModuleList(classification_heads)

        # Name and accuracy of the current task -- only used for logging
        self.current_task = None
        self.finetuning_accuracy = None
        self.custom_logger = custom_logger

        self.log_fn = lambda metric, val: self.log(
            metric, val, on_step=False, on_epoch=True
        )
        self.freeze_head()

    def freeze_head(self):
        for idx in range(len(self.classification_heads)):
            self.classification_heads[idx].weight.requires_grad_(False)
            self.classification_heads[idx].bias.requires_grad_(False)

    def forward(self, inputs):
        return self.moe_model.embed_image(
            inputs, self.classification_heads, self.num_classes
        )

    def _step(self, batch: Dict[str, torch.Tensor], split: str) -> Mapping[str, Any]:
        batch = maybe_dictionarize(batch, "x", "y")

        x = batch["x"]
        gt_y = batch["y"]

        logits = self(x)
        loss = F.cross_entropy(logits, gt_y)
        preds = torch.softmax(logits, dim=-1)

        metrics = getattr(self, f"{split}_acc")
        metrics.update(preds, gt_y)

        self.log_fn(f"acc/{split}/{self.task_name}", metrics)
        self.log_fn(f"loss/{split}/{self.task_name}", loss)

        return {"logits": logits.detach(), "loss": loss}

    def on_test_epoch_end(self):

        if hasattr(self.moe_model, "logging"):
            self.moe_model.logging(self.custom_logger, self.task_name)

        accuracy = (
            self.trainer.callback_metrics[f"acc/test/{self.task_name}"].cpu().item()
        )

        normalized_acc = accuracy / self.finetuning_accuracy

        self.log_fn(f"normalized_acc/test/{self.task_name}", normalized_acc)

    def training_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        return self._step(batch=batch, split="train")

    def validation_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        return self._step(batch=batch, split="val")

    def test_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        return self._step(batch=batch, split="test")

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

    def set_metrics(self, num_classes):

        self.num_classes = num_classes

        metric = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, top_k=1
        )

        self.train_acc = metric.clone()
        self.val_acc = metric.clone()
        self.test_acc = metric.clone()
