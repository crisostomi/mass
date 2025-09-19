import copy
from hmac import new
import os
from typing import Tuple

import torch
from tqdm import tqdm
import numpy as np
from pathlib import Path
import logging
import copy
import os
import torch
import logging
from tqdm import tqdm
from typing import Tuple

from mass.utils.utils import is_matrix

pylogger = logging.getLogger(__name__)

@torch.no_grad()
def sum_svd(
    ref_state_dict, svd_dicts, device="cuda", non_matrix_params_aggregation="base_model", silent=False
):
    """
    Takes the (SVD) for each vector in the task_vectors, and concatenate the low-rank matrices.
    If the vector is not a 2D tensor or is "text_projection", it computes the mean of the vectors.
    Computation of the SVD is performed also for the second operation.

    :param ref_state_dict: The reference state dictionary of the model.
    :param svd_dicts: A dictionary containing the SVD decompositions for each dataset.
    :param non_matrix_params_aggregation: The aggregation method for non-matrix parameters. Valid values are 'mean' or 'base_model'.

    :return: The aggregated model state dictionary.
    """

    aggregated_model_dict = ref_state_dict
    layer_names = list(aggregated_model_dict.keys())

    datasets = list(svd_dicts.keys())

    for layer_name in (tqdm(layer_names, desc="Summing SVD") if not silent else layer_names):
        is_matrix = aggregated_model_dict[layer_name].dim() == 2
        # TODO: modified
        # new_key = layer_name.replace(".transformer", "")
        new_key = layer_name
        offset = 0

        for i, dataset in enumerate(datasets):

            if "text_projection" in layer_name:
                continue

            if is_matrix:

                delta_layer_svd = svd_dicts[dataset][new_key]

                u, s, v = (
                    delta_layer_svd["u"],
                    delta_layer_svd["s"],
                    delta_layer_svd["v"],
                )
                u, s, v = u.to(device), s.to(device), v.to(device)

                if i == 0:
                    total_rank = sum(
                        svd_dicts[d][new_key]["s"].shape[0] for d in datasets
                    )
                    sum_u = torch.zeros(u.shape[0], total_rank, device=device)
                    sum_s = torch.zeros(total_rank, device=device)
                    sum_v = torch.zeros(total_rank, v.shape[1], device=device)

                # reduced_index_s = int(s.shape[0] * sv_reduction)#
                rank_i = s.shape[0]

                # select only the first reduced_index_s columns of u and place them
                sum_u[:, offset : offset + rank_i] = u
                sum_s[offset : offset + rank_i] = s
                sum_v[offset : offset + rank_i, :] = v

                offset += rank_i

            # layer is not a matrix, compute the mean
            else:
                delta_layer = svd_dicts[datasets[i]][new_key]["dim1"].to(device)

                if non_matrix_params_aggregation == "mean":

                    if i == 0:
                        aggregated_model_dict[layer_name] = delta_layer
                    else:
                        aggregated_model_dict[layer_name] += (
                            delta_layer - aggregated_model_dict[layer_name]
                        ) / (i + 1)

                else:  # keep the weights of the base model

                    aggregated_model_dict[layer_name] = torch.zeros_like(delta_layer)

        # aggregation step
        # text_projection is ignored and vectors were already aggregated
        if "text_projection" in layer_name or not is_matrix:
            continue

        u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
        u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)

        aggregated_model_dict[layer_name] = torch.linalg.multi_dot(
            (
                u_u,
                v_u,
                torch.diag(sum_s),
                u_v,
                v_v,
            )
        ).to(device)

    return aggregated_model_dict


def compute_svd_and_compress(
    matrix, compress_ratio
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the Singular Value Decomposition (SVD) of a given matrix and compresses it by reducing the number of singular values.

    Args:
        matrix (torch.Tensor): The input matrix to decompose.
        compress_ratio (float): The fraction of singular values to retain (0 < compress_ratio <= 1).

    Returns:
        tuple: A tuple containing:
            - u (torch.Tensor): The left singular vectors of the reduced SVD.
            - s (torch.Tensor): The reduced singular values.
            - v (torch.Tensor): The right singular vectors of the reduced SVD.
    """
    u, s, v = torch.linalg.svd(matrix, full_matrices=False)

    reduced_index_s = int(s.shape[0] * compress_ratio)

    return u[:, :reduced_index_s], s[:reduced_index_s], v[:reduced_index_s, :]


def compress_tv(task_dicts, compress_rate: float, compress_ratio_per_task=None):
    """
    Compress task vectors using Singular Value Decomposition (SVD).

    Args:
        task_dicts (dict): A dictionary where keys are dataset names and values are task dicts.
        compress_rate (float): The fraction of singular values to keep for compression.
        compress_ratio_per_task (dict, optional): Specific compression ratios per dataset.

    Returns:
        dict: A dictionary with the same structure as `task_dicts`, but with each layer matrix
              replaced by its compressed SVD components (u, s, v) if the layer is 2-dimensional.
              If the layer is not 2-dimensional, it is stored as is under the key "dim1".
    """
    with torch.no_grad():
        svd_dict = {}

        for dataset, task_dict in tqdm(
            task_dicts.items(), desc="Computing and compressing SVD"
        ):
            svd_dict[dataset] = {}

            for key, layer in task_dict.items():
                # Remove ".transformer" from the key but keep the layer
                # TODO: check here
                # new_key = key.replace(".transformer", "")
                new_key = key

                if is_matrix(layer):
                    # Use dataset-specific compression ratio if provided
                    current_compress_rate = (
                        compress_ratio_per_task.get(dataset, compress_rate)
                        if compress_ratio_per_task
                        else compress_rate
                    )

                    u, s, v = compute_svd_and_compress(layer, current_compress_rate)
                    svd_dict[dataset][new_key] = {
                        "u": u.detach().cpu(),
                        "s": s.detach().cpu(),
                        "v": v.detach().cpu(),
                    }
                else:
                    svd_dict[dataset][new_key] = {"dim1": layer.detach().cpu()}

        return svd_dict


def get_svd_dict(
    task_dicts,
    datasets,
    svd_path: str,
    compression_factor: float = None,
    compress_ratio_per_task: dict = None,
):
    """
    Retrieves the SVD dictionary from disk if available, otherwise computes it from scratch and saves it.

    Args:
        task_dicts: The list (or dict) of NonLinearTaskVector objects.
        datasets: The datasets for which the SVD dictionary is built.
        svd_path (str): The file path where the SVD dictionary is stored.
        compression_factor (float, optional): Compression factor to use. Defaults to len(datasets) if not provided.

    Returns:
        dict: The SVD dictionary.
    """

    pylogger.info(f"Datasets: {datasets}")
    compression_factor = compression_factor or len(datasets)
    compression_ratio = 1 / compression_factor
    pylogger.info(f"Using compression ratio: {compression_ratio:.4f}")
    pylogger.info(f"Saving in {svd_path}")
    pylogger.info(f"Final SVD path: {svd_path}")
    pylogger.info(f"{Path(svd_path)}")

    if Path(svd_path).exists():
        pylogger.info(f"Loading precomputed SVD dictionary from: {svd_path}")
        svd_dict = torch.load(svd_path, map_location="cuda")

        if set(svd_dict.keys()) == set(datasets):
            return svd_dict

        pylogger.warning("Mismatch in datasets. Recomputing SVD dictionary...")

    else:
        pylogger.info("No precomputed SVD dictionary found. Computing from scratch...")

    svd_dict = compress_tv(task_dicts, compression_ratio, compress_ratio_per_task)
    torch.save(svd_dict, svd_path)
    pylogger.info(f"SVD dictionary saved at: {svd_path}")

    return svd_dict


def measure_cosine_similarity(delta1: torch.Tensor, delta2: torch.Tensor) -> float:
    """
    Compute cosine similarity between two flattened matrices delta1, delta2.
    Both delta1, delta2 should be 1D, or we flatten them inside this function.
    Returns a float in [-1, 1].
    """
    # Flatten if not already
    d1 = delta1.view(-1)
    d2 = delta2.view(-1)

    dot = torch.dot(d1, d2).item()
    norm1 = torch.norm(d1).item()
    norm2 = torch.norm(d2).item()

    if norm1 < 1e-9 or norm2 < 1e-9:
        # Avoid division by zero; treat this as zero similarity if either is near zero
        return 0.0

    return dot / (norm1 * norm2)


@torch.no_grad()
def sum_svd_no_redundant_tasks_simple(
    ref_state_dict: dict,
    svd_dict: dict,
    device: str = "cuda",
    similarity_threshold: float = 0.2,
):
    """
    Takes the SVD for each vector in the task_vectors, concatenates the low-rank matrices,
    and merges them. If two tasks are more similar than `similarity_threshold`,
    we skip the second one.

    Args:
        ref_state_dict (dict): The reference pretrained model state dict.
        svd_dict (dict): {dataset_name -> {layer_name -> {"u","s","v"}}}.
        device (str): e.g. "cuda" or "cpu".
        similarity_threshold (float): If the cosine similarity between the new task
                                      delta and any accepted delta is above this,
                                      we skip merging it.

    Returns:
        dict: A dictionary containing the new merged weights.
    """

    aggregated_model_dict = ref_state_dict
    layer_names = list(aggregated_model_dict.keys())
    datasets = list(svd_dict.keys())

    for layer_name in tqdm(layer_names, desc="Summing SVD"):
        # check if this layer is 2D (weight matrix) or not
        new_key = layer_name
        is_layer_matrix = aggregated_model_dict[layer_name].dim() == 2
        offset = 0

        # We'll hold tasks that we "accept" (not skip) for merging
        accepted_tasks = []
        # Keep a flattened version of each accepted delta for similarity checks
        accepted_deltas = []

        for i, dataset in enumerate(datasets):
            if "text_projection" in layer_name:
                continue

            if is_layer_matrix:
                # Retrieve the SVD factors
                delta_layer_svd = svd_dict[dataset][new_key]
                u, s, v = (
                    delta_layer_svd["u"].to(device),
                    delta_layer_svd["s"].to(device),
                    delta_layer_svd["v"].to(device),
                )
                # Reconstruct the matrix delta_i
                # shape: [m, rank] * [rank, rank] * [rank, n] => [m, n]
                delta = u @ torch.diag_embed(s) @ v

                # Flatten for similarity check
                delta_flat = delta.view(-1)

                # Compare with each accepted delta
                skip_this = False
                for accepted_flat in accepted_deltas:
                    sim = measure_cosine_similarity(delta_flat, accepted_flat)
                    if sim > similarity_threshold:
                        # This new task is too similar, skip it
                        pylogger.info(
                            f"Skipping task {dataset} for layer {layer_name} due to similarity {sim}"
                        )
                        skip_this = True
                        break

                if not skip_this:
                    # If no overlap > threshold, accept it
                    accepted_tasks.append((u, s, v))
                    accepted_deltas.append(delta_flat)

            else:
                # For 1D layers, we do the usual average
                delta_layer = svd_dict[dataset][new_key]["dim1"].to(device)
                if i == 0:
                    aggregated_model_dict[layer_name] = delta_layer
                else:
                    aggregated_model_dict[layer_name] += (
                        delta_layer - aggregated_model_dict[layer_name]
                    ) / (i + 1)

        # Now that we've decided which tasks are accepted for this layer,
        # we proceed with the same logic as before to build sum_u, sum_s, sum_v
        # from the accepted tasks only
        if "text_projection" in layer_name or not is_layer_matrix:
            continue

        if len(accepted_tasks) == 0:
            # If we ended up skipping all tasks for this layer, just keep it as ref
            # or set it to zero, up to you. We'll just keep pretrained weights.
            continue

        # Build the big (sum_u, sum_s, sum_v) from accepted tasks
        # We do the same "concatenate columns" approach
        # first, figure out total rank
        total_rank = sum(task_s.shape[0] for (_, task_s, _) in accepted_tasks)

        # Prepare placeholders
        sum_u = torch.zeros(
            accepted_tasks[0][0].shape[0], total_rank, device=device
        )  # [m, total_rank]
        sum_s = torch.zeros(total_rank, device=device)
        sum_v = torch.zeros(total_rank, accepted_tasks[0][2].shape[1], device=device)

        offset = 0
        for u_i, s_i, v_i in accepted_tasks:
            rank_i = s_i.shape[0]
            sum_u[:, offset : offset + rank_i] = u_i
            sum_s[offset : offset + rank_i] = s_i
            sum_v[offset : offset + rank_i, :] = v_i
            offset += rank_i

        # Now do your multi-step SVD approach
        u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
        u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)

        # Reconstruct the final merged matrix
        # aggregated_model_dict[layer_name] = ...
        merged = torch.linalg.multi_dot((u_u, v_u, torch.diag(sum_s), u_v, v_v))
        aggregated_model_dict[layer_name] = merged.to(device)

    return aggregated_model_dict
