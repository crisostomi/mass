from typing import (
    Any,
    List,
)

import pytorch_lightning as pl
import torchmetrics

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

class LanguageTester(pl.LightningModule):

    def __init__(self, moe_model, tokenizer):
        super().__init__()
        self.moe_model = moe_model
        self.tokenizer = tokenizer
        
        self.log_fn = lambda metric, val: self.log(
            metric, val, on_step=False, on_epoch=True
        )
        
    def _step(self, batch, split: str):
        raise NotImplementedError
        
    def training_step(self, batch: Any, batch_idx: int):
        return self._step(batch=batch, split="train")

    def validation_step(self, batch: Any, batch_idx: int):
        return self._step(batch=batch, split="val")

    def test_step(self, batch: Any, batch_idx: int):
        return self._step(batch=batch, split="test")

    def on_test_epoch_end(self):
        # TODO: add normalised accuracy
        pass

    def __call__(self, inputs):
        return self.forward(inputs)
    
    def set_task(self, task_name):
        self.task_name = task_name
        
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

        logits, acc = evaluate_accuracy(self.moe_model.model, batch, self.tokenizer)
        
        # Update the MeanMetric with the batch accuracy
        metrics = getattr(self, f"{split}_acc")
        metrics.update(acc)  # MeanMetric.update() takes a single value
    
        running_acc = metrics.compute()
        self.log_fn(f"acc/{split}/{self.task_name}", running_acc)

        return {"logits": logits.detach()}
    
    
class Regression(LanguageTester):
    def _step(self, batch, split: str):
        logits, acc = evaluate_spearman_rho(self.moe_model.model, batch, self.tokenizer)
        
        # Update the MeanMetric with the batch accuracy
        metrics = getattr(self, f"{split}_acc")
        metrics.update(acc)  # MeanMetric.update() takes a single value
    
        running_acc = metrics.compute()
        self.log_fn(f"spearman/{split}/{self.task_name}", running_acc)

        return {"logits": logits.detach()}
        

    