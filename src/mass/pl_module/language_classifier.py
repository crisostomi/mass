from typing import (
    Any,
    List,
    Optional,
)

import pytorch_lightning as pl
import torchmetrics

import logging

pylogger = logging.getLogger(__name__)

from mass.data.language.glue_evaluation import evaluate_accuracy, evaluate_spearman_rho

CLASSIFICATION_TASKS = [
    "cola",
    "mnli",
    "mrpc",
    "qnli",
    "qqp",
    "rte",
    "sst2",
]
REGRESSION_TASKS = ["stsb", "glue-stsb"]


def get_task_config_name(task_name: str) -> str:
    """
    Returns the appropriate task configuration name based on the task.

    Args:
        task_name: Name of the task (e.g., 'cola', 'stsb')

    Returns:
        Configuration name ('lang_classification' or 'lang_regression')
    """
    if task_name in REGRESSION_TASKS:
        return "lang_regression"
    else:
        return "lang_classification"


class LanguageTester(pl.LightningModule):

    def __init__(self, moe_model, tokenizer, custom_logger: Optional[Any] = None):
        super().__init__()
        self.moe_model = moe_model
        self.tokenizer = tokenizer
        self.custom_logger = custom_logger

        self.log_fn = lambda metric, val: self.log(
            metric, val, on_step=False, on_epoch=True
        )

    def set_task_name(self, task_name):
        self.task_name = task_name

    def _step(self, batch, split: str):
        raise NotImplementedError

    def training_step(self, batch: Any, batch_idx: int):
        return self._step(batch=batch, split="train")

    def validation_step(self, batch: Any, batch_idx: int):
        return self._step(batch=batch, split="val")

    def test_step(self, batch: Any, batch_idx: int):
        return self._step(batch=batch, split="test")

    def on_test_epoch_end(self):

        if hasattr(self.moe_model, "flush_cache"):
            self.moe_model.flush_cache()
            pylogger.info("Flushed MoE cache at the end of testing.")

        if hasattr(self.moe_model, "logging"):
            self.moe_model.logging(self.custom_logger, self.task_name)

        accuracy = (
            self.trainer.callback_metrics[f"acc/test/{self.task_name}"].cpu().item()
        )

        normalized_acc = accuracy / self.finetuning_accuracy

        self.log_fn(f"normalized_acc/test/{self.task_name}", normalized_acc)

    def __call__(self, inputs):
        return self.forward(inputs)

    def set_task(self, task_name):
        self.task_name = task_name

    def set_finetuning_accuracy(self, finetuning_accuracy):
        self.finetuning_accuracy = finetuning_accuracy

    def set_metrics(self, num_classes=None):
        """
        Set up averaging metrics since accuracy is already computed in evaluate_accuracy.
        We use MeanMetric to average the accuracy values across batches.
        """
        self.output_classes = num_classes

        # Use MeanMetric to average accuracy values across batches
        self.train_acc = torchmetrics.MeanMetric()
        self.val_acc = torchmetrics.MeanMetric()
        self.test_acc = torchmetrics.MeanMetric()


class SentenceClassification(LanguageTester):
    def _step(self, batch, split: str):

        logits, acc = evaluate_accuracy(self.moe_model, batch, self.tokenizer)

        # Update the MeanMetric with the batch accuracy
        metrics = getattr(self, f"{split}_acc")
        metrics.update(acc)  # MeanMetric.update() takes a single value

        running_acc = metrics.compute()
        self.log_fn(f"acc/{split}/{self.task_name}", running_acc)

        return {"logits": logits.detach()}


class Regression(LanguageTester):
    def _step(self, batch, split: str):
        logits, acc = evaluate_spearman_rho(self.moe_model, batch, self.tokenizer)

        # Update the MeanMetric with the batch accuracy
        metrics = getattr(self, f"{split}_acc")
        metrics.update(acc)  # MeanMetric.update() takes a single value

        running_acc = metrics.compute()
        self.log_fn(f"acc/{split}/{self.task_name}", running_acc)

        return {"logits": logits.detach()}
