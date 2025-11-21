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

import tqdm

import hydra
import numpy as np
import torch
from omegaconf import ListConfig
from pytorch_lightning import Callback

from mass.utils.task_vectors import compute_svd_and_compress

pylogger = logging.getLogger(__name__)


def print_memory(context: str):
    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / 1024**2

    log_message = f"{context} -- System RAM: {ram_mb:.2f} MB"

    if torch.cuda.is_available():
        gpu_allocated_mb = torch.cuda.memory_allocated() / 1024**2
        gpu_peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        _gpu_free_bytes, gpu_total_bytes = torch.cuda.mem_get_info()
        gpu_total_mb = gpu_total_bytes / 1024**2

        log_message += (
            f" | GPU Memory: "
            f"{gpu_allocated_mb:.2f} MB (Allocated) / "
            f"{gpu_peak_mb:.2f} MB (Peak) / "
            f"{gpu_total_mb:.2f} MB (Total)"
        )

    pylogger.warning(log_message)


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
    pylogger.info(
        f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}, ({sum(p.numel() for p in model.parameters() if p.requires_grad) / sum(p.numel() for p in model.parameters()) * 100}%)"
    )
    pylogger.info(
        f"Total number of parameters: {sum(p.numel() for p in model.parameters())}"
    )


def return_params_summary(model: torch.nn.Module) -> Dict[str, int]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    trainable_percentage = (
        (trainable_params / total_params * 100) if total_params > 0 else 0
    )

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "trainable_percentage": trainable_percentage,
    }


def print_parameters_increase(
    model_before: torch.nn.Module, model_after: torch.nn.Module
):
    params_before = sum(p.numel() for p in model_before.parameters())
    params_after = sum(p.numel() for p in model_after.parameters())
    increase = params_after - params_before
    pylogger.info(
        f"Parameters before: {params_before}, after: {params_after}, increase: {increase} ({increase / params_before * 100:.2f}%)"
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


# TODO: unify with the below
def pad_unbatched_output(outputs, output_classes):
    """
    Trims a list of unbatched output tensors to match the specified number of output classes,
    then stacks them into a batch.

    Args:
        outputs (list of torch.Tensor): List of tensors with shape (num_classes,) - one per sample.
        output_classes (int): The fixed number of classes to retain in each tensor.

    Returns:
        torch.Tensor: Stacked tensor with shape (batch_size, output_classes).
    """
    trimmed_outputs = []

    for out in outputs:
        num_classes = out.shape[0]

        if num_classes > output_classes:
            out = out[:output_classes]

        elif num_classes < output_classes:
            pad_size = output_classes - num_classes
            pad = torch.zeros(pad_size, device=out.device, dtype=out.dtype)
            out = torch.cat([out, pad], dim=0)

        trimmed_outputs.append(out)

    return torch.stack(trimmed_outputs, dim=0)


# TODO: unify with the above
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


def apply_dict_to_model(task_vector_dict, model, coefficient: float = 1.0):
    """
    Applies a task vector dictionary to a model. The resulting model is the deep copy of the input model
    on the GPU with the task vector applied to the weights.
    """
    with torch.no_grad():
        model.cuda()
        new_state_dict = (
            model.state_dict()
        )  # Get model's state_dict (reference, not a copy)

        for key, value in task_vector_dict.items():
            # new_key = key.replace("encoder.", "")
            if key not in new_state_dict:
                pylogger.warning(
                    f"Key {key} is present in the task vector but not in the model"
                )
                continue
            else:
                new_state_dict[key] += coefficient * value.cuda()  # Update weight

        model.load_state_dict(new_state_dict, strict=False)  # Load updated parameters
    return model.cuda()


def apply_dict_to_dict(task_vector_dict, model_dict, coefficient: float = 1.0):
    """
    Applies a task vector dictionary to a model. The resulting model is the deep copy of the input model
    on the GPU with the task vector applied to the weights.
    """
    with torch.no_grad():
        for key, value in task_vector_dict.items():
            # new_key = key.replace("encoder.", "")
            if key not in model_dict:
                pylogger.warning(
                    f"Key {key} is present in the task vector but not in the model"
                )
                continue
            else:
                model_dict[key] += coefficient * value.cuda()  # Update weight

    return model_dict


def sum_task_dict(task_vector_dict_1, task_vector_dict_2):
    """
    Sums two task vector dictionaries. It sums task_vector_dict_2 into task_vector_dict_1.
    """

    for key, value in task_vector_dict_2.items():
        if key in task_vector_dict_1:
            task_vector_dict_1[key] += value.cuda()  # TODO: remove
        else:
            task_vector_dict_1[key] = value.cuda()  # TODO: remove
    return task_vector_dict_1


def is_matrix(layer):
    return len(layer.shape) == 2


def is_matrix_dict(layer):
    return isinstance(layer, dict) and "u" in layer


def get_routing_weights_from_finetuned(
    finetuned, zeroshot, layer, device="cuda", dtype=torch.float32
):
    """
    Computes SVD components and returns them on the specified device and with the specified dtype.
    """
    vs = []
    sigma = []

    for task in finetuned.keys():
        if layer not in finetuned[task].state_dict():
            raise KeyError(
                f"Layer '{layer}' not found in finetuned model for key '{task}'."
            )

        layer_tensor = (
            finetuned[task].state_dict()[layer] - zeroshot.state_dict()[layer]
        )

        if not is_matrix(layer_tensor):
            pylogger.warning(f"Layer '{layer}' in task '{task}' is not a matrix.")
            continue

        with torch.no_grad():
            _, s, v = compute_svd_and_compress(
                layer_tensor.to(device), 1 / len(finetuned)
            )

        vs.append(v.to(device=device, dtype=dtype))
        sigma.append(s.to(device=device, dtype=dtype))

    return (
        torch.stack(vs) if vs else None,
        torch.stack(sigma) if sigma else None,
        None
    )


def get_routing_weights_from_task_dict(task_dict, layer):
    vs = []
    sigma = []

    for task in task_dict.keys():
        if layer not in task_dict[task]:
            raise KeyError(f"Layer '{layer}' not found in task dict for key '{task}'.")

        layer_tensor = task_dict[task][layer]

        if not is_matrix(layer_tensor):
            pylogger.warning(f"Layer '{layer}' in task '{task}' is not a matrix.")
            continue

        with torch.no_grad():
            _, s, v = compute_svd_and_compress(layer_tensor, 1 / len(task_dict))

        vs.append(v.to("cuda"))
        sigma.append(s.to("cuda"))

    return (
        torch.stack(vs) if vs else None,
        torch.stack(sigma) if sigma else None,
        None
    )


def get_routing_weights(svd_dict, layer, get_sigma=False, get_u=False):
    """
    Returns the right singular vectors.

    Args:
        svd_dict (dict): Dictionary containing SVD components.
        layer (str): Layer name to retrieve weights for.
        get_sigma (bool): Whether to return singular values.
        get_u (bool): Whether to return left singular vectors.

    Returns:
        tuple: Stacked right singular vectors, singular values (if requested), and left singular vectors (if requested).
    """
    vs = []
    sigma = []

    for dt in svd_dict.keys():
        if layer not in svd_dict[dt]:
            raise KeyError(
                f"Layer '{layer}' not found in SVD dictionary for key '{dt}'."
            )

        layer_data = svd_dict[dt][layer]
        if not all(k in layer_data for k in ["v", "s", "u"]):
            raise KeyError(
                f"Missing keys in SVD data for layer '{layer}' under key '{dt}'."
            )

        vs.append(layer_data["v"].to("cuda"))
        sigma.append(layer_data["s"].to("cuda"))


    return (
        torch.stack(vs) if vs else None,
        torch.stack(sigma) if get_sigma and sigma else None,
        None
    )
    
def get_routing_weights_whitened(svd_dict, layer, get_sigma=False, get_u=False, whitened=False):
    """
    Returns the right singular vectors.

    Args:
        svd_dict (dict): Dictionary containing SVD components.
        layer (str): Layer name to retrieve weights for.
        get_sigma (bool): Whether to return singular values.
        get_u (bool): Whether to return left singular vectors.

    Returns:
        tuple: Stacked right singular vectors, singular values (if requested), and left singular vectors (if requested).
    """
    vs = []
    sigma = []

    for dt in svd_dict.keys():
        if layer not in svd_dict[dt]:
            raise KeyError(
                f"Layer '{layer}' not found in SVD dictionary for key '{dt}'."
            )

        layer_data = svd_dict[dt][layer]
        if not all(k in layer_data for k in ["v", "s", "u"]):
            raise KeyError(
                f"Missing keys in SVD data for layer '{layer}' under key '{dt}'."
            )

        vs.append(layer_data["v"].to("cuda"))
        sigma.append(layer_data["s"].to("cuda"))
        
    mean = []
    for i,v in enumerate(vs):
        mean.append(torch.mean(v, dim=1, keepdim=True))
        print(f"mean[i].shape: {mean[i].shape}")
        vs[i] = v - mean[i]
        print(f"vs[i].shape: {vs[i].shape}")
    print(len(vs))
    Vs = torch.cat(vs, dim=0)
    print(f"Vs.shape: {Vs.shape}")
    cov = torch.cov(Vs.T)
    print(f"cov.shape: {cov.shape}")

    
    return (
        torch.stack(vs) if vs else None,
        torch.stack(sigma) if get_sigma and sigma else None,
        torch.cholesky_inverse(cov)
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


def svd_key_from_layer(key, index):
    base = router_key_from_layer(key, index)
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

    for key in tqdm.tqdm(pretrained, desc="Computing task dict"):
        if "embed_tokens" in key:
            pylogger.info(f"Skipping key {key}")
            continue
        if "lm_head" in key:
            pylogger.info(f"Skipping key {key}")
            continue
        if pretrained[key].dtype in [torch.int64, torch.uint8]:
            pylogger.info(f"Skipping key {key}")
            continue

        difference = finetuned[key] - pretrained[key]
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


def is_all_zeros(tensor: torch.Tensor | List[torch.Tensor]) -> bool:
    """
    Check if a tensor or a list of tensors are all zeros.

    Args:
        tensor (Tensor | List[Tensor]): A tensor or a list of tensors.

    Returns:
        bool: True if all elements are zeros, False otherwise.
    """
    if isinstance(tensor, torch.Tensor):
        return torch.allclose(tensor, torch.zeros_like(tensor))
    else:
        return all(is_all_zeros(t) for t in tensor)
