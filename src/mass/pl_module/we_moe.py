from ast import Dict
import copy
import functools

from hydra.utils import instantiate
import logging
from typing import Any, Generic, List, cast, Union  # noqa: F401

import lightning.fabric.wrappers
from omegaconf import OmegaConf
import torch
import torch.optim
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader
import torchmetrics
from tqdm.autonotebook import tqdm
from nn_core.common import PROJECT_ROOT

from mass.merger.arithmetic_merger import TaskArithmeticMerger
from mass.modules.encoder import ImageEncoder
from mass.modules.smile_gates import ExpertNotTrainedError
from mass.modules.we_moe import WeightEnsemblingMoE
from mass.pl_module.image_multihead_classifier import MultiHeadImageClassifier

from mass.utils.fusion_bench_utils import (
    InfiniteDataLoader,
    get_attr,
    get_device,
    set_attr,
)

from open_clip import CLIP

from mass.utils.io_utils import get_classification_heads
from mass.utils.utils import pad_unbatched_output, print_params_summary

pylogger = logging.getLogger(__name__)


def entropy_loss(logits: Tensor) -> Tensor:
    """
    Compute the entropy loss of a set of logits.

    Args:
        logits (Tensor): The logits to compute the entropy loss of.

    Returns:
        Tensor: The entropy loss of the logits.
    """
    probs = torch.softmax(logits, dim=-1)
    return -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()


class WeightEnsemblingMoEAlgorithm:
    _mlp_class = (nn.Sequential,)

    def __init__(
        self,
        zeroshot_model,
        finetuned_models,
        dataset_names,
        merger,
        optimizer,
        save_checkpoint_path,
        checkpoint=False,
        save_checkpoint=False,
        router_hidden_layers=2,
        init_lambda=0.3,
        max_steps=1000,
        batch_size=16,
        batch_reduce=True,
        use_grad_accumulate=True,
        model_path=None,
        device="cuda",
        encoder_name=None,
        ckpt_path=None,
        openclip_cachedir=None,
    ):

        # Store configuration parameters
        self.dataset_names = dataset_names
        self.merger = merger
        self.optimizer_config = optimizer
        self.save_checkpoint_path = save_checkpoint_path
        self.checkpoint = checkpoint
        self.save_checkpoint = save_checkpoint
        self.router_hidden_layers = router_hidden_layers
        self.init_lambda = init_lambda
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.batch_reduce = batch_reduce
        self.use_grad_accumulate = use_grad_accumulate
        self.model_path = model_path
        self.device = device

        self.task_to_index = {task: i for i, task in enumerate(dataset_names)}

        self.zeroshot_model = zeroshot_model
        self.finetuned_models = finetuned_models
        self.classification_heads = get_classification_heads(
            self.dataset_names, encoder_name, ckpt_path, openclip_cachedir
        )

        pylogger.info(
            "Fusing models using WeightEnsembling Mixture of Experts modules."
        )

        moe_model = self.construct_moe_model(zeroshot_model, finetuned_models)

        print_params_summary(moe_model)

        if self.checkpoint:
            pylogger.info(
                f"load checkpoint from {self.save_checkpoint_path}, test-time adaptation will be skipped."
            )
            self.load_checkpoint(moe_model, self.save_checkpoint_path)
        else:
            moe_model = self.test_time_adaptation(moe_model)
            if self.save_checkpoint:
                pylogger.info(f"save checkpoint to {self.save_checkpoint_path}")

                torch.save({"model": moe_model}, self.save_checkpoint_path)

        # Store the final model
        self.model = WeMoEInferenceWrapper(
            moe_model, zeroshot_model, dataset_names, device=device
        )

    def load_checkpoint(self, model: Any, checkpoint_path: str):
        """
        Load the checkpoint file.

        Args:
            model: The model to load the checkpoint into.
            checkpoint_path: The path to the checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint["model"])

    def _upscale_linear_layer(
        self,
        pretrained_model: ImageEncoder,
        moe_model,
        finetuned_models,
        name: str,
    ):
        """
        Upscale a linear layer by merging it with the corresponding layers from the fine-tuned models.

        Args:
            pretrained_model (nn.Module): The pretrained model.
            finetuned_models (List[nn.Module]): A list of fine-tuned models.
            name (str): The name of the linear layer to upscale.
        """

        name_list = name.split(".")
        try:
            module = get_attr(pretrained_model, name_list)
        except AttributeError as e:
            pylogger.warning(
                f"Failed to get attribute {name} from pretrained model: {e}"
            )
            exit(1)

        original_device = get_device(module)
        module = module.to(self.device, non_blocking=True)
        experts = [
            get_attr(m, name_list).to(self.device, non_blocking=True)
            for m in finetuned_models.values()
        ]
        try:
            moe_linear = WeightEnsemblingMoE(
                hidden_size=module[0].in_features,
                base_model=module,
                expert_models=experts,
                init_lambda=self.init_lambda,
                batch_first=False,  # For open_clip models this is False
                router_hidden_layers=self.router_hidden_layers,
                batch_reduce=self.batch_reduce,
            )
            moe_linear = moe_linear.to(original_device, non_blocking=True)
            # pylogger.info(f"Successfully upscaled layer: {name}")
        except ExpertNotTrainedError:
            pylogger.info(f"skip {name} because the experts are not trained.")
            exit(1)
        except Exception as e:
            pylogger.error(f"Failed to upscale layer {name}: {e}")
            exit(1)
        set_attr(moe_model, name_list, moe_linear)

    def construct_moe_model(
        self,
        pretrained_model: ImageEncoder,
        finetuned_models,
        tqdm_desc: str = "Creating WeMoE",
    ):
        """
        Construct the Mixture of Experts (MoE) model using the models in the model pool.

        Returns:
            WeightEnsemblingMoE: The constructed MoE model.
        """

        # Merge the models using task arithmetic
        moe_model = self.merger.merge(
            pretrained_model, copy.deepcopy(finetuned_models)
        ).requires_grad_(False)

        # Up-scale MLP modules
        for name, module in tqdm(
            tuple(pretrained_model.named_modules()),
            tqdm_desc,
            leave=False,
            dynamic_ncols=True,
        ):
            if isinstance(module, self._mlp_class):
                self._upscale_linear_layer(
                    pretrained_model, moe_model, finetuned_models, name
                )

        return moe_model

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
        image_embeds = module(images.cuda())

        text_embeds = self.classification_heads[self.task_to_index[task]].cuda()
        logits_per_text = text_embeds(image_embeds)
        return logits_per_text

    @functools.cache
    def get_infinite_dataloader(self, task):
        dataset_cfg = OmegaConf.load(PROJECT_ROOT / "conf" / "dataset" / f"{task}.yaml")

        dataset = instantiate(
            dataset_cfg,
            preprocess_fn=self.zeroshot_model.val_preprocess,
            batch_size=self.batch_size,
        )
        return iter(InfiniteDataLoader(dataset.test_loader))

    def test_time_adaptation(self, module: WeightEnsemblingMoE) -> WeightEnsemblingMoE:
        """
        Perform test-time adaptation for the given module.

        Args:
            module (WeightEnsemblingMoE): The MoE module to adapt.

        Returns:
            WeightEnsemblingMoE: The adapted MoE module.
        """

        # Configure optimizer - handle both partial and regular configs
        trainable_params = [p for p in module.parameters() if p.requires_grad]

        optimizer: torch.optim.Optimizer
        if callable(self.optimizer_config):
            optimizer = cast(
                torch.optim.Optimizer, self.optimizer_config(params=trainable_params)
            )
        else:
            optimizer = cast(
                torch.optim.Optimizer,
                instantiate(
                    self.optimizer_config,
                    params=trainable_params,
                ),
            )

        print_params_summary(module)
        module.cuda()
        module.train()

        pbar = tqdm(
            range(self.max_steps),
            "Test-time adaptation",
            dynamic_ncols=True,
        )
        for step_idx in pbar:
            if self.use_grad_accumulate:
                for task in self.dataset_names:
                    batch = next(
                        self.get_infinite_dataloader(task)
                    )  # Use cached iterator
                    logits = self.compute_logits(module, batch, task)
                    assert (
                        logits.dim() == 2
                    ), f"Expected logits to be 2D, got {logits.dim()}"
                    loss = entropy_loss(logits)
                    # .backward() accumulates when .zero_grad() wasn't called
                    # this can save memory
                    loss.backward(retain_graph=True)
            else:
                total_loss = None
                for task in self.dataset_names:

                    batch = next(
                        self.get_infinite_dataloader(task)
                    )  # Use cached iterator

                    logits = self.compute_logits(module, batch, task)
                    assert (
                        logits.dim() == 2
                    ), f"Expected logits to be 2D, got {logits.dim()}"

                    task_loss = entropy_loss(logits)
                    if total_loss is None:
                        total_loss = task_loss
                    else:
                        total_loss = total_loss + task_loss

                if total_loss is not None:
                    total_loss.backward(retain_graph=True)

            optimizer.step()
            optimizer.zero_grad()

        return module


class WeMoEInferenceWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        zeroshot_model: nn.Module,
        dataset_names: List[str],
        device: str = "cuda",
    ):
        super().__init__()
        self.model = model
        self.zeroshot_model = zeroshot_model
        self.dataset_names = dataset_names
        self.device = device

        self.task_to_index = {task: i for i, task in enumerate(dataset_names)}

    def collect_votes(self, bsz, device):
        votes = []

        for name, module in self.model.named_modules():
            if isinstance(module, WeightEnsemblingMoE) and hasattr(
                module, "last_selected_experts"
            ):
                if module.last_selected_experts is None:
                    pylogger.warning(f"Module {name} has no last selected experts")
                else:
                    votes.append(module.last_selected_experts)

        if votes is not None:
            votes = torch.stack(votes)
            majority_vote = torch.mode(votes, dim=0).values
        else:
            majority_vote = torch.zeros(bsz, dtype=torch.long, device=device)
        return majority_vote

    def embed_image(self, batch, classification_heads, num_classes):

        features = self.model(batch)

        majority_vote = self.collect_votes(batch.size(0), batch.device)

        head_groups = self.group_samples_by_selected_head(majority_vote)

        all_outputs = [None] * batch.size(0)

        for head_idx, sample_indices in head_groups.items():

            group_features = features[sample_indices]
            group_output = classification_heads[head_idx](group_features)

            for i, sample_idx in enumerate(sample_indices):
                all_outputs[sample_idx] = group_output[i]

        return pad_unbatched_output(all_outputs, num_classes)

    @property
    def train_preprocess(self):
        return getattr(self.zeroshot_model, "train_preprocess", None)

    @property
    def val_preprocess(self):
        return getattr(self.zeroshot_model, "val_preprocess", None)

    def generate(self, batch, max_length):
        return self.model.generate(batch, max_length=max_length)

    def group_samples_by_selected_head(self, selected_heads: torch.Tensor):
        """
        Group samples that share the same selected head to be processed together for efficiency

        Args:
            selected_heads: Tensor of shape (batch_size,) containing head indices for each sample

        Returns:
            Dict mapping head_idx to list of sample indices
        """
        head_group_to_samples = {}

        for sample_idx, head_idx in enumerate(selected_heads.cpu().numpy()):
            head_idx = int(head_idx)
            head_group_to_samples.setdefault(head_idx, []).append(sample_idx)

        return head_group_to_samples
