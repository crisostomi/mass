import pytorch_lightning as pl
import torch
from typing import Any, Dict, Mapping, Optional

import torch.nn.functional as F

from mass.data.datamodule import MetaData
from mass.utils.utils import torch_load, torch_save
from mass.data.datasets.common import maybe_dictionarize


class MultiHeadImageClassifier(pl.LightningModule):
    def __init__(
        self,
        encoder,
        classification_heads,
        metadata: Optional[MetaData] = None,
        *args,
        **kwargs,
    ):
        """ """
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=("metadata",))

        self.encoder = encoder
        if self.encoder is not None:
            self.train_preprocess = self.encoder.train_preprocess
            self.val_preprocess = self.encoder.val_preprocess

        self.classification_heads = torch.nn.ModuleList(classification_heads)

        # Name and accuracy of the current task -- only used for logging
        self.current_task = None
        self.finetuning_accuracy = None

        self.log_fn = lambda metric, val: self.log(
            metric, val, on_step=False, on_epoch=True
        )
        self.freeze_head()

    def freeze_head(self):
        for idx in range(len(self.classification_heads)):
            self.classification_heads[idx].weight.requires_grad_(False)
            self.classification_heads[idx].bias.requires_grad_(False)

    def forward(self, inputs, head_idx):
        features = self.encoder(inputs)
        outputs = self.classification_heads[head_idx](features)
        return outputs

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

    def on_test_epoch_end(self):

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

    def __call__(self, inputs, head_idx):
        return self.forward(inputs, head_idx)

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
