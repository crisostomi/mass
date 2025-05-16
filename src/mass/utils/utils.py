from collections import OrderedDict
import copy
import logging
import os
import pickle
import psutil
import json
import zipfile
from pathlib import Path
from typing import Dict, List

import hydra
import numpy as np
import torch
from omegaconf import ListConfig
from pytorch_lightning import Callback

pylogger = logging.getLogger(__name__)


def print_memory(context):
    process = psutil.Process(os.getpid())
    pylogger.warning(
        f"{context} -- memory in MB: { process.memory_info().rss / 1024**2}",
    )


def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)

    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)

    return _lr_adjuster


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]


def add_normalized_accuracy(results, finetuning_accuracies):
    for dataset_name, metrics in results.items():
        if dataset_name in finetuning_accuracies:
            normalized_acc = (
                metrics[0][f"acc/test/{dataset_name}"]
                / finetuning_accuracies[dataset_name]
            )
            results[dataset_name][0][
                f"acc/test_normalized/{dataset_name}"
            ] = normalized_acc

    return results


def get_finetuning_accuracies(path):
    with open(path, "rb") as f:
        finetuning_accuracies = json.load(f)
    return finetuning_accuracies


def compute_avg_accuracy(results) -> Dict:
    total_acc = 0
    total_normalized_acc = 0
    count = 0

    for dataset_name, metrics in results.items():
        for m in metrics:
            total_acc += m[f"acc/test/{dataset_name}"]
            total_normalized_acc += m[f"normalized_acc/test/{dataset_name}"]
            count += 1

    average_acc = total_acc / count if count > 0 else 0
    average_normalized_acc = total_normalized_acc / count if count > 0 else 0

    return {
        "acc/test/avg": average_acc,
        "normalized_acc/test/avg": average_normalized_acc,
    }


def torch_load_old(save_path, device=None):
    with open(save_path, "rb") as f:
        classifier = pickle.load(f)
    if device is not None:
        classifier = classifier.to(device)
    return classifier


def torch_save(model, save_path):
    if os.path.dirname(save_path) != "":
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.cpu(), save_path)


def torch_load(save_path, device=None):
    model = torch.load(save_path, weights_only=False)
    if device is not None:
        model = model.to(device)
    return model


def get_logits(inputs, classifier):
    assert callable(classifier)
    if hasattr(classifier, "to"):
        classifier = classifier.to(inputs.device)
    return classifier(inputs)


def get_probs(inputs, classifier):
    if hasattr(classifier, "predict_proba"):
        probs = classifier.predict_proba(inputs.detach().cpu().numpy())
        return torch.from_numpy(probs)
    logits = get_logits(inputs, classifier)
    return logits.softmax(dim=1)


def print_params_summary(model: torch.nn.Module):
    print(
        f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}, ({sum(p.numel() for p in model.parameters() if p.requires_grad) / sum(p.numel() for p in model.parameters()) * 100}%)"
    )


def print_mask_summary(model: torch.nn.Module, mask_mode):
    pct_masked = {}

    for name, module in model.named_modules():
        if not hasattr(module, "weight"):
            continue

        pct_masked[name] = (
            round(1.0 - torch.mean(module.weight_mask).item(), 2)
            if hasattr(module, "weight_mask")
            else 0.0
        )

    print("Percentage of masked weights in each layer:")
    from rich import pretty

    pretty.pprint(pct_masked)

    # TODO remove this when one of the two (basically equivalent) methods, when a choice has been taken
    if mask_mode == "by_layer":
        print(
            f"% of masked weights across the entire network: {round(torch.as_tensor(data=list(pct_masked.values()), dtype=torch.float32).mean().item(), 2)}"
        )
    elif mask_mode == "by_nn":
        print(
            f"Sum percentage of masked weights across the entire network: {round(torch.as_tensor(data=list(pct_masked.values()), dtype=torch.float32).mean().item(), 2)}"
        )


class LabelSmoothing(torch.nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def build_callbacks(cfg: ListConfig, *args: Callback, verbose=False) -> List[Callback]:
    """Instantiate the callbacks given their configuration.

    Args:
        cfg: a list of callbacks instantiable configuration
        *args: a list of extra callbacks already instantiated

    Returns:
        the complete list of callbacks to use
    """
    callbacks: List[Callback] = list(args)

    for callback in cfg:
        if verbose:
            pylogger.info(f"Adding callback <{callback['_target_'].split('.')[-1]}>")
        callbacks.append(hydra.utils.instantiate(callback, _recursive_=False))

    return callbacks


def pad_output(outputs, output_classes):
    """
    Trims a list of output tensors to match the specified number of output classes.

    Args:
        outputs (list of torch.Tensor): List of tensors with shape (batch_size, num_classes).
        output_classes (int): The fixed number of classes to retain in each tensor.

    Returns:
        torch.Tensor: Concatenated tensor with shape (batch_size, output_classes).
    """
    trimmed_outputs = []

    for out in outputs:
        num_classes = out.shape[1]

        if num_classes > output_classes:
            out = out[:, :output_classes]  # Trim exceeding classes

        elif num_classes < output_classes:
            pad_size = output_classes - num_classes
            pad = torch.zeros(
                (out.shape[0], pad_size), device=out.device, dtype=out.dtype
            )
            out = torch.cat([out, pad], dim=1)  # Pad with zeros if necessary

        trimmed_outputs.append(out)

    return torch.cat(trimmed_outputs, dim=0)


def get_hook_fn_impact(model, name):
    """
    Hook function to capture both input and output for impact logging.
    It extracts the token embeddings from both the input and output
    and computes the average L2 norm difference across all tokens.
    """

    def hook_fn(module, input, output):
        # Extract the main tensor from input and output (handle tuple cases)
        inp = input[0] if isinstance(input, tuple) else input
        out = output[0] if isinstance(output, tuple) else output

        # Assuming shape (B, seq_len, hidden); compute per-token L2 norm
        diff = torch.norm(out - inp, p=2, dim=-1)  # Shape: (B, seq_len)

        # Compute the mean impact over all tokens
        avg_diff = diff.mean(dim=1)  # Shape: (B,)

        # Log the results
        model.layer_impact_log[name].append(avg_diff.detach().cpu().numpy())

    return hook_fn


def get_hook_fn(model, name, input_or_output="input"):
    """
    Register a hook to store intermediate features.
    """

    def hook_fn_output(module, input, output):
        if isinstance(output, torch.Tensor):
            model.middle_features[name] = output.cpu().detach()
        elif isinstance(output, tuple):
            model.middle_features[name] = output[0].cpu().detach()

    def hook_fn_input(module, input, output):
        if isinstance(input, torch.Tensor):
            model.middle_features[name] = input.cpu().detach()
        elif isinstance(input, tuple):
            model.middle_features[name] = input[0].cpu().detach()

    hook_fn = hook_fn_output if input_or_output == "output" else hook_fn_input

    return hook_fn


def reconstruct_tv_from_svddict(svd_dict, device="cuda"):
    with torch.no_grad():
        tv = {
            key: (
                (
                    svd_dict[key]["u"]
                    @ torch.diag_embed(svd_dict[key]["s"])
                    @ svd_dict[key]["v"]
                ).to(device)
                if "u" in svd_dict[key]
                else svd_dict[key]["dim1"].to(device)
            )
            for key in svd_dict.keys()
        }

    return tv


def state_dict_to_vector(state_dict, remove_keys=[]):
    shared_state_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in shared_state_dict:
            del shared_state_dict[key]
    sorted_shared_state_dict = OrderedDict(sorted(shared_state_dict.items()))
    return torch.nn.utils.parameters_to_vector(
        [value.reshape(-1) for key, value in sorted_shared_state_dict.items()]
    )


def vector_to_state_dict(vector, state_dict, remove_keys=[]):
    # create a reference dict to define the order of the vector
    reference_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in reference_dict:
            del reference_dict[key]
    sorted_reference_dict = OrderedDict(sorted(reference_dict.items()))

    # create a shared state dict using the refence dict
    torch.nn.utils.vector_to_parameters(vector, sorted_reference_dict.values())

    # add back the encoder and decoder embedding weights.
    if "transformer.shared.weight" in sorted_reference_dict:
        for key in remove_keys:
            sorted_reference_dict[key] = sorted_reference_dict[
                "transformer.shared.weight"
            ]
    return sorted_reference_dict


def apply_dict_to_model(task_vector_dict, model, coefficient: float = 1.0):
    """
    Applies a task vector dictionary to a model. The ressulting model is the deep copy of the input model
    on the GPU with the task vector applied to the weights.
    """
    with torch.no_grad():
        model.cuda()
        new_state_dict = (
            model.state_dict()
        )  # Get model's state_dict (reference, not a copy)

        for key, value in task_vector_dict.items():
            new_key = key.replace("encoder.", "")
            if new_key not in new_state_dict:
                pylogger.warning(
                    f"Key {new_key} is present in the task vector but not in the model"
                )
                continue
            else:
                new_state_dict[new_key] += coefficient * value.cuda()  # Update weight

        model.load_state_dict(new_state_dict, strict=False)  # Load updated parameters
    return model.cuda()


def is_matrix(layer):
    return len(layer.shape) == 2


def get_routing_weights(svd_dict, layer, get_sigma=False, get_u=False):
    """
    Returns the right singular vectors
    """

    vs = []
    sigma = []
    us = []

    for dt in svd_dict.keys():
        layer_key = layer.replace("encoder.", "")

        vs.append(svd_dict[dt][layer_key]["v"].cuda())
        sigma.append(svd_dict[dt][layer_key]["s"].cuda())
        us.append(svd_dict[dt][layer_key]["u"].cuda())

    return (
        torch.stack(vs),
        torch.stack(sigma) if get_sigma else None,
        torch.stack(us) if get_u else None,
    )


def is_supported_layer(layer_key: str) -> bool:
    """
    Check if layer_key contains 'mlp' or 'attn' and 'resblocks.'
    """

    return (
        ("resblocks." in layer_key)
        and (("attn" in layer_key) or ("mlp" in layer_key))
        and not ("ln" in layer_key)
        and not ("gelu" in layer_key)
        and not ("c_proj" in layer_key)
        and not ("c_fc" in layer_key)
    )


def router_key_from_layer(key, index):
    return f"encoder.model.visual.transformer.resblocks.{index}.{key}"

def _router_key_from_layer(key, index):
    return f"encoder.model.visual.resblocks.{index}.{key}"


def _router_key_from_layer(key, index):
    return f"encoder.model.visual.resblocks.{index}.{key}"


def _router_key_from_layer(key, index):
    return f"encoder.model.visual.resblocks.{index}.{key}"


def svd_key_from_layer(key, index):
    base = _router_key_from_layer(key, index)
    if "attn" in key:
        return base + ".in_proj_weight"
    elif "mlp" in key:
        return base + ".c_fc.weight"


def from_router_to_svd_dict_key(key):
    key = key.replace("model.encoder.", "")
    if "attn" in key:
        return key + ".in_proj_weight"
    if "mlp" in key:
        return key + ".c_fc.weight"


@torch.no_grad()
def compute_task_dict(pretrained, finetuned):
    new_state_dict = OrderedDict()

    for key in pretrained:
        if pretrained[key].dtype in [torch.int64, torch.uint8]:
            pylogger.info(f"Skipping key {key}")
            continue

        difference = finetuned[key].cuda() - pretrained[key].cuda()
        new_state_dict[key] = difference

    return new_state_dict


def unzip_all_in_folder(folder_path):
    """
    Unzips all .zip files in the given folder.

    Args:
        folder_path (str): The path to the folder containing zip files.

    Returns:
        None
    """
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory.")
        return

    for file in os.listdir(folder_path):
        if file.endswith(".zip"):  # Check if the file is a ZIP archive
            zip_path = os.path.join(folder_path, file)

            # Remove all extensions from the filename
            folder_name = file.split(".")[0]
            extract_path = os.path.join(folder_path, folder_name)

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_path)  # Extract files

            print(f"Extracted: {zip_path} → {extract_path}")
