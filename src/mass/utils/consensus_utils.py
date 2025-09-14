import copy
import logging
import os
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Optional
from mass.utils.utils import apply_dict_to_model
from pathlib import Path

pylogger = logging.getLogger(__name__)

import torch


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
            sorted_reference_dict[key] = sorted_reference_dict["transformer.shared.weight"]
    return sorted_reference_dict


def generate_task_masks(
    tv_flattened_checkpoints: torch.Tensor,
    flattened_finetunings: torch.Tensor,
    flattened_pretrained: torch.Tensor,
    multi_tv: Optional[torch.Tensor] = None,
    tall_mask_lambda: float = 1.0,
) -> torch.Tensor:
    """
    Generate task-specific TALL masks
    TALL masks are generated as: mask_t = |theta_0 - theta_t| > |theta_mt - theta_t| * lambda

    Args:
        tv_flattened_checkpoints: individual task vectors
        flattened_finetunings: individual theta_t (fine-tuned weights)
        flattened_pretrained: theta_0 (pre-trained weight)
        tv: multi-task vector
        tall_mask_lambda: hyper-parameter lambda for generating TALL masks
    Returns:
        final_mask: generated TALL masks with the given lambda, in shape (n_task, n_parameter)
    """

    print(f"Generating TALL masks.")

    if multi_tv is None:
        multi_tv = tv_flattened_checkpoints.sum(0)

    flat_multi_task_model = flattened_pretrained + multi_tv

    original_shape = flattened_finetunings.shape

    # generate masks by comparing the l1 distance between |theta_0 - theta_t| and |theta_mt - theta_t|
    diff_pt_ft = (flattened_pretrained - flattened_finetunings).abs()
    diff_multi_ft = (flat_multi_task_model - flattened_finetunings).abs()
    # compare the l1 distance, scaled with hyper-parameter lambda
    mask = diff_pt_ft > diff_multi_ft * tall_mask_lambda

    final_mask = (
        mask.squeeze() if original_shape == tv_flattened_checkpoints.squeeze().shape else mask
    )

    print(
        f"Average sparsity for the mask with tall_mask_lambda of {tall_mask_lambda}: {final_mask.float().mean():.4f}"
    )

    return final_mask


def construct_tall_mask(
    tv_flattened_checkpoints: torch.Tensor,
    flattened_finetunings: torch.Tensor,
    flattened_pretrained: torch.Tensor,
    merged_tv: torch.Tensor,
    pretrained_model_checkpoint: torch.Tensor,
    dataset_names: List[str],
):
    """
    Construct TALL masks for all tasks for each lambda, and store in dictionary

    Args:
        tv_flattened_checkpoints: individual task vectors
        flattened_finetunings: individual theta_t (fine-tuned weights)
        flattened_pretrained: theta_0 (pre-trained weight)
        merged_tv: multi-task vector
        pretrained_model_checkpoint: pre-trained weight as state dictionary
        remove_keys: the keys to be removed when converting between dictionary and vector
    Returns:
        tall_masks: constructed TALL masks in dictionary format of {lambda: {task: mask}}
    """
    tall_masks = {}
    for tall_mask_lambda in [0.2, 0.3, 0.4, 0.5, 0.6]:
        # generate tall masks for each lambda
        masks_at_scale = generate_task_masks(
            tv_flattened_checkpoints,
            flattened_finetunings,
            flattened_pretrained,
            tall_mask_lambda=tall_mask_lambda,
            multi_tv=merged_tv,
        )
        # convert vectors to dictionary
        masks_at_scale = [
            vector_to_state_dict(mask, pretrained_model_checkpoint) for mask in masks_at_scale
        ]
        # store the masks with {dataset: mask}
        tall_masks[tall_mask_lambda] = {
            key: value for key, value in zip(dataset_names, masks_at_scale)
        }
    return tall_masks


def load_tall_mask(pretrained_model_checkpoint, model_name, mask_location, dataset_names):
    """Loads TALL masks from disk, unpack and transform to state dictionaries."""

    try:
        # if config.method.use_ties:
        #     print("==== Loading TALL Masks built with TIES ====")
        #     tall_masks = torch.load(
        #         os.path.join(
        #             mask_location,
        #             config.model,
        #             f"TALL_mask_{config.num_tasks}task_use_ties.npy",
        #         )
        #     )
        # else:
        print("==== Loading TALL Masks built with Task Arithmetic ====")
        dir_path = Path(mask_location) / model_name
        npy_path = dir_path / f"TALL_mask_{len(dataset_names)}task.npy"

        raw = np.load(
            npy_path,
            allow_pickle=True,
        )
    except:
        raise Exception("TALL Masks are not constructed yet.")

    # Case 1: 0-D object array holding a dict -> unwrap with .item()
    if isinstance(raw, np.ndarray) and raw.dtype == object and raw.shape == ():
        obj = raw.item()
        if not isinstance(obj, dict):
            raise TypeError(f"Expected a dict in {npy_path.name}, got {type(obj)}")
        tall_masks = obj

    # Case 2: object array / list-like aligned with dataset_names
    elif isinstance(raw, np.ndarray) and raw.dtype == object:
        if raw.size != len(dataset_names):
            raise ValueError(
                f"Mask count ({raw.size}) != number of datasets ({len(dataset_names)})."
            )
        tall_masks = {ds: raw[i] for i, ds in enumerate(dataset_names)}

    # Case 3: numeric array aligned with dataset_names along axis 0
    elif isinstance(raw, np.ndarray):
        if raw.shape[0] != len(dataset_names):
            raise ValueError(
                f"First dimension of array ({raw.shape[0]}) != number of datasets ({len(dataset_names)})."
            )
        tall_masks = {ds: raw[i] for i, ds in enumerate(dataset_names)}
    else:
        raise TypeError(f"Unsupported data type loaded from {npy_path.name}: {type(raw)}")

    # unpack masks and convert back to torch tensors
    tall_masks = {k: torch.from_numpy(np.unpackbits(v)) for k, v in tall_masks.items()}

    # convert vectors to dictionaries
    tall_masks = {
        dataset: vector_to_state_dict(mask, pretrained_model_checkpoint)
        for dataset, mask in tall_masks.items()
    }

    return tall_masks


def construct_consensus_mask(
    pretrained_model_checkpoint,
    prun_thre_k,
    model_name,
    mask_location,
    dataset_names,
    remove_keys=[],
):
    """
    Generate consensus mask by filtering out least-used parameters

    Args:
        pretrained_model_checkpoint: pretrained_checkpoint as state dictionary
        prun_thre_k: weight-pruning threhold, stands for the least number of activated tasks for a parameter to be preserved from pruning
                if prun_thre_k is set to 2: remove both catastrophic and selfish weights;
                if prun_thre_k is set to 1: remove only catastrophic weights;
                if prun_thre_k is set to 0: remove no weights -> reduce to TA or TIES
                if prun_thre_k is set to > num_tasks: remove all weights -> reduce to zero-shot
    Returns:
        consensus_mask_vector: constructed consensus mask as vector (boolean in shape (n_parameter, ))
    """

    print("==== Generating Consensus Mask ====")
    # load TALL masks (in shape (n_task, n_parameter))
    tall_masks = load_tall_mask(
        pretrained_model_checkpoint, model_name, mask_location, dataset_names
    )
    tall_masks = list(tall_masks.values())

    # generate consensus masks
    consensus_mask = copy.deepcopy(tall_masks[0])
    for key, value in consensus_mask.items():
        consensus_mask[key] = torch.zeros_like(value)
        # count for each parameter, the tasks it has been activated for
        for mask in tall_masks:
            consensus_mask[key] = consensus_mask[key] + mask[key].float()
        # filter out the least-activated parameters based on given threshold
        consensus_mask[key] = consensus_mask[key].float() >= prun_thre_k
    # consensus_mask_vector = state_dict_to_vector(consensus_mask, remove_keys=remove_keys)

    return consensus_mask  # consensus_mask_vector


def find_optimal_mask(
    val_metrics,
    eval_masks,
    dataset_names,
    load_mask,
    mask_location,
    model_name,
    save_masks=True,
    use_normalized_acc=False,  # set True to select on normalized accuracy
):
    """
    Finds the optimal mask for each dataset/task by maximizing a chosen validation metric.

    Args:
        val_metrics: dict[lambda] -> dict[dataset] -> (list[dict] | dict) of metrics
        eval_masks:  dict[lambda] -> dict[dataset] -> state_dict-like mask
        dataset_names: list of dataset names to consider
        load_mask: bool, whether we're loading masks instead of saving them
        mask_location: base directory for saving masks
        model_name: subdirectory name for saving masks
        save_masks: whether to save the selected masks as packed bits
        use_normalized_acc: if True, use "normalized_acc/val/<ds>" instead of "acc/val/<ds>"

    Returns:
        best_masks_for_test: dict[dataset] -> mask(state_dict)
        best_val_metrics:    dict[dataset] -> original metrics payload for the chosen lambda
    """

    def _extract_metric(metrics_payload, ds_name, prefer_normalized):
        """
        metrics_payload: either a dict of metrics or a 1-length list containing that dict.
        Returns a float metric value.
        """
        # Unwrap possible list
        if isinstance(metrics_payload, list):
            if len(metrics_payload) == 0:
                raise ValueError(f"Empty metrics list for dataset '{ds_name}'.")
            metrics_dict = metrics_payload[-1]  # take the last one if multiple
        elif isinstance(metrics_payload, dict):
            metrics_dict = metrics_payload
        else:
            raise TypeError(f"Unexpected metrics type for '{ds_name}': {type(metrics_payload)}")

        # Choose key
        main_prefix = "normalized_acc/val/" if prefer_normalized else "acc/val/"
        exact_key = f"{main_prefix}{ds_name}"

        # Fast path: exact key present
        if exact_key in metrics_dict:
            return float(metrics_dict[exact_key])

        # Fallback: find a key that contains both the prefix and the dataset name
        for k, v in metrics_dict.items():
            if main_prefix in k and ds_name in k:
                return float(v)

        # Last resort: any acc/val key with dataset name
        for k, v in metrics_dict.items():
            if "acc/val/" in k and ds_name in k:
                return float(v)

        raise KeyError(
            f"Could not find an accuracy key for dataset '{ds_name}'. "
            f"Looked for '{exact_key}' or similar in keys: {list(metrics_dict.keys())}"
        )

    # transpose into: per-dataset -> per-lambda -> numeric score
    transposed_dict = {}
    for lam, per_ds in val_metrics.items():
        # normalize lambda key to float for consistent indexing
        try:
            lam_f = float(lam)
        except Exception:
            lam_f = lam  # leave as-is if truly non-numeric
        for ds_name, metrics_payload in per_ds.items():
            score = _extract_metric(metrics_payload, ds_name, use_normalized_acc)
            transposed_dict.setdefault(ds_name, {})[lam_f] = score

    # for each dataset, pick the lambda with the highest score
    max_subkeys = {
        ds_name: max(lam_to_score, key=lam_to_score.get)
        for ds_name, lam_to_score in transposed_dict.items()
    }

    # select masks and collect metrics for the winning lambda per dataset
    best_masks_for_test = {}
    best_masks_for_test_vector = {}
    best_val_metrics = {}

    for ds in dataset_names:
        if ds not in max_subkeys:
            raise KeyError(f"Dataset '{ds}' not found in validation metrics.")

        best_lambda = max_subkeys[ds]

        # handle possible string-vs-float key mismatch in eval_masks/val_metrics
        def _getitem_by_lambda(dct, lam_key):
            if lam_key in dct:
                return dct[lam_key]
            try:
                lf = float(lam_key)
                if lf in dct:
                    return dct[lf]
            except Exception:
                pass
            ls = str(lam_key)
            if ls in dct:
                return dct[ls]
            raise KeyError(f"Lambda '{lam_key}' not found in dict keys: {list(dct.keys())}")

        # select the mask based on the selected lambda
        best_masks_for_test[ds] = _getitem_by_lambda(eval_masks, best_lambda)[ds]

        # vectorize mask
        best_masks_for_test_vector[ds] = state_dict_to_vector(
            best_masks_for_test[ds], remove_keys=[]
        )

        # keep the original metrics payload for reference
        best_val_metrics[ds] = _getitem_by_lambda(val_metrics, best_lambda)[ds]

        print(
            f"Best lambda for {ds} is {best_lambda} "
            f"(val {'normalized ' if use_normalized_acc else ''}acc="
            f"{transposed_dict[ds][best_lambda]:.6f})"
        )

    # optionally save packed masks
    if save_masks and not load_mask:
        # ensure directory exists
        save_dir = os.path.join(mask_location, model_name)
        os.makedirs(save_dir, exist_ok=True)

        # np.packbits expects uint8/bool; cast defensively
        packed = {
            k: np.packbits(np.asarray(v, dtype=np.uint8))
            for k, v in best_masks_for_test_vector.items()
        }

        mask_name = f"TALL_mask_{len(dataset_names)}task.npy"
        masks_file_path = os.path.join(save_dir, mask_name)
        print(f"Saving best masks to {masks_file_path}")
        np.save(masks_file_path, packed)
        print(f"Saved best masks to {masks_file_path}")
        del best_masks_for_test_vector  # free memory

    return best_masks_for_test, best_val_metrics


def apply_eval_mask(task_vector_dict, model, eval_mask, coefficient: float = 1.0):
    """
    Applies a tall mask to a model. The resulting model is the deep copy of the input model
    on the GPU with the task vector applied to the weights.
    """
    sparse_task_vector = copy.deepcopy(task_vector_dict)
    # apply mask to sparsify the task vectors with Hadamard product
    sparse_task_vector = {
        k: sparse_task_vector[k] * eval_mask[k].bool().cpu() for k in eval_mask.keys()
    }
    # reconstruct theta_t^
    image_encoder = apply_dict_to_model(sparse_task_vector, model, coefficient=coefficient)
    return image_encoder
