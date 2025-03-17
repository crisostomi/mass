import copy
import numpy as np
from hydra.utils import instantiate
import pytorch_lightning as pl
import wandb
import torch

import tvp
from mass.pl_module.image_multihead_classifier import MultiHeadImageClassifier

import torchmetrics
from mass.task_vectors.aggregator import (
    IsotropicAggregator,
    TaskSingularVectorAggregator,
    WeightedAggregator,
)
from mass.utils.plots import (
    plot_interactive_coefficients_barchart,
    plot_interactive_curve,
    plot_interactive_coefficients_std,
)
from mass.utils.utils import (
    apply_dict_to_model,
    reconstruct_tv_from_svddict,
    pad_output,
)
import logging

pylogger = logging.getLogger(__name__)


num_of_tasks_to_scaling_coeff = {
    1: 1.0,
    2: 0.4,
    3: 0.35,
}


class MASS(MultiHeadImageClassifier):
    def __init__(
        self, router, encoder, zeroshot_model, classification_heads, svd_dicts, **kwargs
    ):
        """

        encoder: the model used to do the first pass of MASS
        router:
        zeroshot_model:
        classification_heads: list of classification heads, one for each dataset
        """
        super().__init__(
            encoder=encoder, classification_heads=classification_heads, **kwargs
        )

        self.router = router
        self.svd_dicts = svd_dicts
        self.output_classes = None

        self.aggregator = instantiate(
            self.hparams.aggregator, zeroshot_model=zeroshot_model.cuda()
        )
        self.heads_selection_critieria = (
            torch.mean if self.hparams.heads_selection_method == "avg" else torch.max
        )

        self.dataset_names = list(svd_dicts.keys())
        self.dataset_idx_to_name = {
            i: name for i, name in enumerate(self.dataset_names)
        }
        self.dataset_name_to_idx = {
            name: i for i, name in enumerate(self.dataset_names)
        }

        self.task_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=len(self.dataset_names), top_k=1
        )

        self.cached_tvs = {}

        self.coeffs_to_log = []
        self.task_act_to_log = {}

        self.max_num_tvs_to_keep = 6

    def set_metrics(self, num_classes):

        self.output_classes = num_classes

        metric = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, top_k=1
        )

        self.train_acc = metric.clone()
        self.val_acc = metric.clone()
        self.test_acc = metric.clone()

    @torch.no_grad()
    def forward(self, images: torch.Tensor):

        selected_dataset_idxs, dataset_coeffs, dataset_group_to_samples = self.router(
            images
        )

        # log coefficients
        self.coeffs_to_log.append(dataset_coeffs.mean(dim=0).cpu().numpy())
        # log task accuracy

        pred_tasks = torch.max(dataset_coeffs, dim=1)[1]
        gt_tasks = torch.full_like(pred_tasks, self.dataset_name_to_idx[self.task_name])

        task_acc = self.task_accuracy(pred_tasks, gt_tasks)
        self.log_fn(f"task_accuracy/{self.task_name}", task_acc)

        # log task activation
        active_tasks = torch.sum(dataset_coeffs > self.router.threshold, dim=1)

        for active_num in active_tasks:
            active = int(active_num.item())
            if active not in self.task_act_to_log:
                self.task_act_to_log[active] = 0
            self.task_act_to_log[active] += 1

        # task survival
        correct_task_coeffs = dataset_coeffs[
            :, self.dataset_name_to_idx[self.task_name]
        ]
        task_survival_count = torch.mean(
            (correct_task_coeffs > self.router.threshold).float()
        ).item()
        self.log_fn(f"task_survival/{self.task_name}", task_survival_count)

        batch_size = images.shape[0]
        sample_embeddings = [None] * batch_size

        for dataset_group, assigned_sample_idxs in dataset_group_to_samples.items():
            dataset_group_idxs = torch.tensor(
                [
                    self.dataset_name_to_idx[dataset_name]
                    for dataset_name in dataset_group
                ]
            )  # Convert to a PyTorch tensor

            assigned_sample_idxs = torch.tensor(
                assigned_sample_idxs
            )  # Ensure assigned_sample_idxs is also a tensor

            merged_model = self._apply_tv(
                list(dataset_group),
                coefficients=dataset_coeffs[
                    assigned_sample_idxs[:, None], dataset_group_idxs
                ].mean(dim=0),
            )

            # (num_samples_in_group, C, H, W)
            group_images = images[assigned_sample_idxs]

            # (num_samples_in_group, embedding_dim)
            group_output = merged_model(group_images)

            for j, idx in enumerate(assigned_sample_idxs):
                sample_embeddings[idx] = group_output[j : j + 1]

        sample_embeddings = torch.cat(sample_embeddings, dim=0)

        outputs = []

        for sample_routed_datasets, sample_embedding in zip(
            selected_dataset_idxs, sample_embeddings
        ):

            assert isinstance(
                sample_routed_datasets, (int, list, tuple)
            ), f"Unexpected type for routing indices: {type(sample_routed_datasets)}"

            # logits for each dataset the sample was routed to, so a tensor for each routed_dataset in len(sample_routed_datasets)
            candidate_logits = [
                self.classification_heads[j](sample_embedding.unsqueeze(0))
                for j in sample_routed_datasets
            ]
            # for each dataset, get the heads_selection_criteria score
            candidate_scores = [
                self.heads_selection_critieria(logits).item()
                for logits in candidate_logits
            ]  ## try with the max mean (trained with contrastive loss)
            # get the index of the best score among the datasets
            best_idx = candidate_scores.index(max(candidate_scores))
            # get the logits of the best dataset
            logits = candidate_logits[best_idx]

            outputs.append(logits)

        assert (
            self.output_classes is not None
        ), "Output classes not set. Use set_metrics() method to set them."

        return pad_output(outputs, self.output_classes)

    @torch.no_grad()
    def _apply_tv(self, dataset_names, coefficients):
        """Apply the aggregated task vector to the model."""

        dataset_combo = "_".join(dataset_names)

        if dataset_combo in self.cached_tvs:

            return self.cached_tvs[dataset_combo]

        pylogger.info(f"Reconstructing task vector for {dataset_combo}")

        if isinstance(self.aggregator, WeightedAggregator):
            single_scaling_coeff = num_of_tasks_to_scaling_coeff[len(dataset_names)]

            tvs = []

            for dataset_name in dataset_names:

                # eventually offload the cache to the RAM
                tv = reconstruct_tv_from_svddict(
                    self.svd_dicts[dataset_name], device="cpu"
                )

                tvs.append(tv)

            aggregated = self.aggregator.aggregate(tvs, single_scaling_coeff)

            pylogger.info(f"Storing dataset combo: {dataset_combo}")
            self.cached_tvs[dataset_combo] = aggregated

            return aggregated

        elif isinstance(self.aggregator, TaskSingularVectorAggregator) or isinstance(
            self.aggregator, IsotropicAggregator
        ):

            single_scaling_coeff = 1.0

            aggregated = self.aggregator.aggregate(
                {
                    dataset_name: self.svd_dicts[dataset_name]
                    for dataset_name in dataset_names
                },
                coefficients,
            )

            self.cached_tvs[dataset_combo] = aggregated

            return aggregated

        else:
            raise NotImplementedError

    def __call__(self, images):
        return self.forward(images)

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        if len(self.cached_tvs) > self.max_num_tvs_to_keep:
            self.flush_cache()

        return super().on_test_batch_end(outputs, batch, batch_idx, dataloader_idx)

    def flush_cache(self):
        for k, v in self.cached_tvs.items():
            del v

        self.cached_tvs = {}

    def on_test_epoch_end(self):

        self.flush_cache()

        self.cached_tvs = {}

        self.plot_coeffs()
        self.plot_task_activation()

        # router stats logging
        self.router.logging(self.trainer.logger, self.task_name)

        self.reset_log_stats()

        return super().on_test_epoch_end()

    def plot_task_activation(self):

        x_values = sorted(self.task_act_to_log.keys())
        total_samples = sum(self.task_act_to_log.values())

        y_values = [self.task_act_to_log[x] / total_samples for x in x_values]

        fig = plot_interactive_coefficients_barchart(
            y_values,
            x_values,
            title="Activated tasks",
            x_label="Number of Activated Tasks",
            y_label="Percentage of Samples",
        )

        self.trainer.logger.experiment.log(
            {f"activated_tasks/{self.task_name}": wandb.Plotly(fig)}
        )

    def plot_coeffs(self):
        self.coeffs_to_log = np.array(self.coeffs_to_log)

        mean_coeffs = self.coeffs_to_log.mean(axis=0)
        std_coeffs = self.coeffs_to_log.std(axis=0)

        fig_std = plot_interactive_coefficients_std(
            mean_coeffs, std_coeffs, self.dataset_names
        )

        std_table = wandb.Table(columns=["Dataset", "Std Dev"])
        for dataset, std in zip(self.dataset_names, std_coeffs):
            std_table.add_data(dataset, std)

        self.trainer.logger.experiment.log(
            {
                f"coefficients/{self.task_name}": wandb.Plotly(fig_std),
                f"coefficients_std/{self.task_name}_table": std_table,
            }
        )

    def reset_log_stats(self):
        self.coeffs_to_log = []
        self.task_act_to_log = {}
