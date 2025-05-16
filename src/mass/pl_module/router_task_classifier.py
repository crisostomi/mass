from typing import Any, Dict, Mapping

import hydra
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics

from mass.data.datasets.common import maybe_dictionarize
from mass.utils.utils import torch_load, torch_save


class RouterTaskClassifier(pl.LightningModule):
    def __init__(self, router, **kwargs):
        super().__init__()
        self.save_hyperparameters(
            logger=False, ignore=("metadata", "encoder", "router")
        )

        self.router = router

    def set_metrics(self, num_classes):

        self.output_classes = num_classes

        metric = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, top_k=1
        )

        self.train_acc = metric.clone()
        self.val_acc = metric.clone()
        self.test_acc = metric.clone()

    def forward(self, images, split="test"):
        if split == "train":
            return self.router.predict_task_train(images)
        else:
            return self.router.predict_task(images)

    def parameters(self):
        return self.router.parameters()

    def __call__(self, images, split="test"):
        return self.forward(images, split=split)

    def _step(self, batch: Dict[str, torch.Tensor], split: str) -> Mapping[str, Any]:
        batch = maybe_dictionarize(batch, self.hparams.x_key, self.hparams.y_key)

        x = batch[self.hparams.x_key]
        gt_y = batch[self.hparams.y_key]

        logits = self(x, split)
        loss = F.cross_entropy(logits, gt_y)
        preds = torch.softmax(logits, dim=-1)

        metrics = getattr(self, f"{split}_acc")
        metrics.update(preds, gt_y)

        self.log_dict(
            {
                f"acc/{split}": metrics,
                f"loss/{split}": loss,
            },
            on_epoch=True,
        )

        return {"logits": logits.detach(), "loss": loss}

    def training_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        return self._step(batch=batch, split="train")

    def validation_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        return self._step(batch=batch, split="val")

    def test_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        return self._step(batch=batch, split="test")

    def save(self, filename):
        print(f"Saving image classifier to {filename}")
        torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f"Loading image classifier from {filename}")
        return torch_load(filename)

    def configure_optimizers(
        self,
    ):
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

    def state_dict(self):
        return {"router": self.router.state_dict()}

    def load_state_dict(self, state_dict, strict=True):
        if "router" in state_dict:
            self.router.load_state_dict(state_dict["router"], strict=strict)
        else:
            raise KeyError("Expected 'router' key in state_dict")

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["state_dict"] = self.state_dict()

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.load_state_dict(checkpoint["state_dict"])

    def new_load_from_checkpoint(self, path):
        checkpoint = torch.load(path, weights_only=False)
        self.on_load_checkpoint(checkpoint)
