import logging
import os
from copy import deepcopy
import re
from typing import (
    Any,
    List,
)

import torch
from torch import Tensor, nn
from tqdm.auto import tqdm
import pytorch_lightning as pl

from mass.modules.smile_gates import (
    ExpertNotTrainedError,
    SmileMoELinear,
)
from mass.utils.fusion_bench_utils import (
    get_attr,
    get_device,
    replace_attention_with_linear,
    set_attr,
    simple_average,
)


pylogger = logging.getLogger(__name__)


class SmileUpscalingAlgorithm():
    # _linear_layer_cls = (nn.Linear,) avoid hard coding
    _linear_layer_cls = (nn.Linear,)
    _upscaled_layer_cls = (SmileMoELinear,)

    def __init__(
        self,
        zeroshot_model,
        finetuned_models,
        oracle_mode: bool = False,
        device: str = "cuda",
        full_matrices: bool = True,
        gate_k: int = 256,
        k: int = 256,
        top_k: int = 1,
        routing_use_diff: bool = True,
        average_experts: bool = False,
        model_path: str = None,
        **kwargs: Any,
    ):
        """
        Initialize the SmileUpscalingAlgorithm.

        Args:
            device (str): The device to perform the computation on.
            upscaling_accelerator (str): The device to perform the SVD computation on.
            full_matrices (bool): Whether to compute the full-sized U and V matrices.
            gate_k (int): The number of singular values to keep for the gate.
            k (int): The number of singular values to keep for the experts.
            top_k (int): The number of top experts to select.
            routing_use_diff (bool): Whether to use weight differences for routing.
            average_experts (bool): Whether to average the experts.
            model_path (str): The path to save/load the model.
            **kwargs: Additional arguments.
        """

        self.merge_device = device
        self.full_matrices = full_matrices
        self.gate_k = gate_k
        self.k = k
        self.top_k = top_k
        self.routing_use_diff = routing_use_diff
        self.average_experts = average_experts
        self.model_path = model_path
        self.upscaling_accelerator = kwargs.pop("upscaling_accelerator", None)
        self.upscaled_layers = set()  
        self.oracle_mode = oracle_mode

        for key, value in kwargs.items():
            pylogger.warning(f"Unrecognized argument: {key}")
            setattr(self, key, value)

        if model_path is not None and os.path.exists(model_path):
            pylogger.info(f"Loading model from {model_path}")
            model = torch.load(model_path)
        else:
            # finetuned_models should already be a list of model objects
            pylogger.info(f"Received finetuned_models as {type(finetuned_models)}")
            if isinstance(finetuned_models, dict):
                pylogger.info("Converting finetuned_models dict to list...")
                finetuned_models_list = list(finetuned_models.values())
            else:
                pylogger.info(f"Using finetuned_models as list")
                finetuned_models_list = finetuned_models
            pylogger.info("Creating SMILE model...")
            model = self.merge(zeroshot_model, finetuned_models_list)

        # self.pylogger.info_profile_summary()  # TODO: implement profiling
        if model_path is not None:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            pylogger.info(f"Saving model to {model_path}")
            torch.save(model, model_path)
            
        self.model = model
       
    def merge(
        self,
        zeroshot_model: nn.Module,
        finetuned_models: List[nn.Module],
        in_place: bool = True,
    ) -> nn.Module:
        """
        Merges the pretrained model with the fine-tuned models to create an upscaled model.

        Args:
            zeroshot_model (nn.Module): The pretrained model.
            finetuned_models (List[nn.Module]): A list of fine-tuned models.
            in_place (bool): If True, modifies the pretrained model in place. Otherwise, creates a copy.

        Returns:
            nn.Module: The merged model.
        """
        if in_place:
            model = zeroshot_model
        else:
            model = deepcopy(zeroshot_model)

        self._upscale_submodules(model, finetuned_models)
        return model

    def _upscale_linear_layer(
        self,
        zeroshot_model,
        finetuned_models,
        name: str,
    ):
        """
        Upscale a linear layer by merging it with the corresponding layers from the fine-tuned models.

        Args:
            zeroshot_model (nn.Module): The pretrained model.
            finetuned_models (List[nn.Module]): A list of fine-tuned models.
            name (str): The name of the linear layer to upscale.
        """

        name_list = name.split(".")
        pylogger.info(f"Layer name {name}")
        try:
            module = get_attr(zeroshot_model, name_list)
        except AttributeError as e:
            pylogger.warning(
                f"Failed to get attribute {name} from pretrained model: {e}"
            )
            set_attr(zeroshot_model, name_list, None)
            self.upscaled_layers.discard(name)  # Remove from upscaled layers set
            return

        pylogger.info(f"Upscaling layer {name} of type {type(module)}")
        original_device = get_device(module)
        module = module.to(self.merge_device, non_blocking=True)
        experts = [
            get_attr(m, name_list).to(self.merge_device, non_blocking=True)
            for m in finetuned_models
        ]
        try:
            moe_linear = SmileMoELinear(
                module,
                experts,
                gate_k=self.gate_k,
                k=self.k,
                top_k=self.top_k,
                routing_use_diff=self.routing_use_diff,
                full_matrices=self.full_matrices,
                upscaling_accelerator=self.upscaling_accelerator,
                name=name,
            )
            moe_linear = moe_linear.to(original_device, non_blocking=True)
            pylogger.info(f"Successfully upscaled layer: {name}")
        except ExpertNotTrainedError:
            pylogger.info(f"skip {name} because the experts are not trained.")
            self.upscaled_layers.discard(name)  # Remove from upscaled layers set
            return
        except Exception as e:
            pylogger.error(f"Failed to upscale layer {name}: {e}")
            self.upscaled_layers.discard(name)  # Remove from upscaled layers set
            return
        set_attr(zeroshot_model, name_list, moe_linear)
        # remove the original module from fine-tuned models to save memory
        for m in finetuned_models:
            set_attr(m, name_list, None)

    def _average_experts(self, pretarined_model, finetuned_models, name: str):
        """
        Average the experts for a given layer.

        Args:
            pretarined_model (nn.Module): The pretrained model.
            finetuned_models (List[nn.Module]): A list of fine-tuned models.
            name (str): The name of the layer to average.
        """
        name_list = name.split(".")
        experts = [get_attr(m, name_list) for m in finetuned_models]
        averaged_module = simple_average(experts)
        set_attr(pretarined_model, name_list, averaged_module)

    def _upscale_submodules(
        self,
        zeroshot_model: nn.Module,
        finetuned_models: List[nn.Module],
        tqdm_desc: str = "Upscaling Linear Modules",
    ):
        """
        Upscales the submodules of the pretrained model by merging them with the corresponding submodules from the fine-tuned models.

        Args:
            zeroshot_model (nn.Module): The pretrained model.
            finetuned_models (List[nn.Module]): A list of fine-tuned models.
            tqdm_desc (str): Description for the tqdm progress bar.
        """
        # TODO: do we need this still?
        replace_attention_with_linear(zeroshot_model, finetuned_models)

        for name, module in tqdm(
            tuple(zeroshot_model.named_modules()),
            tqdm_desc,
            leave=False,
            dynamic_ncols=True,
        ):
            if isinstance(module, self._linear_layer_cls):
                pylogger.info(f"Upscaling linear layer: {name}")
                self._upscale_linear_layer(
                    zeroshot_model=zeroshot_model,
                    finetuned_models=finetuned_models,
                    name=name,
                )
            elif self.average_experts and len(tuple(module.named_modules())) == 1:
                pylogger.info(f"Averaging experts for leaf module: {name}")
                # if the module is a leaf module, we perform a parameter average
                self._average_experts(zeroshot_model, finetuned_models, name)
                

    # def set_metrics(self, num_classes):

    #     self.output_classes = num_classes

    #     metric = torchmetrics.Accuracy(
    #         task="multiclass", num_classes=num_classes, top_k=1
    #     )

    #     self.train_acc = metric.clone()
    #     self.val_acc = metric.clone()
    #     self.test_acc = metric.clone()

    # def __call__(self, inputs):
    #     if self.oracle_mode:
    #         return self.forward_oracle(inputs)
    #     return self.forward(inputs)

    # def forward(self, inputs):
    #     votes = []
    #     collected_layers = set()

    #     features = self.encoder(inputs)

    #     for name, module in self.encoder.named_modules():
    #         pylogger.debug(f"Module {name}")
    #         if isinstance(module, SmileMoELinear) and hasattr(
    #             module, "last_selected_experts"
    #         ):
    #             collected_layers.add(name)
    #             if module.last_selected_experts is None:
    #                 pylogger.warning(f"Module {name} has no last selected experts")
    #             else:
    #                 votes.append(module.last_selected_experts)

    #     if votes:
    #         votes = torch.stack(votes)
    #         majority_vote = torch.mode(votes, dim=0).values
    #     else:
    #         majority_vote = torch.zeros(
    #             inputs.size(0), dtype=torch.long, device=inputs.device
    #         )

    #     head_groups = self.group_samples_by_selected_head(majority_vote)

    #     all_outputs = [None] * inputs.size(0)

    #     for head_idx, sample_indices in head_groups.items():

    #         group_features = features[sample_indices]

    #         group_output = self.classification_heads[head_idx](group_features)

    #         for i, sample_idx in enumerate(sample_indices):
    #             all_outputs[sample_idx] = group_output[i]

    #     return pad_unbatched_output(all_outputs, self.output_classes)

    # def forward_oracle(self, inputs):
    #     pylogger.warning("Using oracle forward pass")
    #     features = self.encoder(inputs)
    #     return self.classification_heads[self.head_idx](features)

    # def group_samples_by_selected_head(self, selected_heads: torch.Tensor):
    #     """
    #     Group samples that share the same selected head to be processed together for efficiency

    #     Args:
    #         selected_heads: Tensor of shape (batch_size,) containing head indices for each sample

    #     Returns:
    #         Dict mapping head_idx to list of sample indices
    #     """
    #     head_group_to_samples = {}

    #     for sample_idx, head_idx in enumerate(selected_heads.cpu().numpy()):
    #         head_idx = int(head_idx)
    #         head_group_to_samples.setdefault(head_idx, []).append(sample_idx)

    #     return head_group_to_samples
