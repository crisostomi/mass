import copy
import os
from typing import Tuple

import torch
from tqdm import tqdm
import numpy as np
from pathlib import Path
import logging

from mass.utils.utils import is_matrix

pylogger = logging.getLogger(__name__)


@torch.no_grad()
def isotropic_sum(ref_state_dict, svd_dict, device="cuda"):
    aggregated_model_dict = ref_state_dict
    layer_names = list(aggregated_model_dict.keys())

    datasets = list(svd_dict.keys())

    for layer_name in tqdm(layer_names, desc="Summing SVD"):
        is_matrix = aggregated_model_dict[layer_name].dim() == 2

        for i, dataset in enumerate(datasets):

            if "text_projection" in layer_name:
                continue

            if is_matrix:

                delta_layer_svd = svd_dict[dataset][layer_name]

                u, s, v = (
                    delta_layer_svd["u"],
                    delta_layer_svd["s"],
                    delta_layer_svd["v"],
                )
                u, s, v = u.to(device), s.to(device), v.to(device)
                delta = u @ torch.diag_embed(s) @ v

                if i == 0:
                    sum = torch.zeros_like(delta)

                sum += delta

            else:
                delta_layer = svd_dict[datasets[i]][layer_name]["dim1"].to(device)

                if i == 0:
                    aggregated_model_dict[layer_name] = delta_layer
                else:
                    aggregated_model_dict[layer_name] += (
                        delta_layer - aggregated_model_dict[layer_name]
                    ) / (i + 1)

        if "text_projection" in layer_name or not is_matrix:
            continue

        u, s, v = torch.linalg.svd(sum, full_matrices=False)

        iso_factor = torch.mean(s)

        aggregated_model_dict[layer_name] = iso_factor * u @ v

    return aggregated_model_dict


@torch.no_grad()
def sum_svd(ref_state_dict, svd_dicts, device="cuda"):
    """
    Takes the (SVD) for each vector in the task_vectors, and concatenate the low-rank matrices.
    If the vector is not a 2D tensor or is "text_projection", it computes the mean of the vectors.
    Computation of the SVD is performed also for the second operation.

    Args:
        task_vectors (list): A list of task vector objects, where each object contains a
                             dictionary of vectors.
        config (object): Configuration object containing the following attributes:
                         - DATASETS (list): List of datasets.
                         - device (torch.device): The device to perform computations on.

    Returns:
        dict: A dictionary containing the new vectors after SVD computation and merging.
    """

    aggregated_model_dict = ref_state_dict
    layer_names = list(aggregated_model_dict.keys())

    datasets = list(svd_dicts.keys())

    for layer_name in tqdm(layer_names, desc="Summing SVD"):
        is_matrix = aggregated_model_dict[layer_name].dim() == 2
        new_key = layer_name.replace(".transformer", "")
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

                if i == 0:
                    aggregated_model_dict[layer_name] = delta_layer
                else:
                    aggregated_model_dict[layer_name] += (
                        delta_layer - aggregated_model_dict[layer_name]
                    ) / (i + 1)

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


def get_uncompressed_weights(task_dicts, compress_rate: float, layer: str):
    layer = layer.replace("encoder.", "")
    routing_weights = []
    routing_sigmas = []
    with torch.no_grad():
        for dataset, task_dict in tqdm(
            task_dicts.items(), desc="Computing and compressing SVD"
        ):
            u, s, v = compute_svd_and_compress(task_dict[layer], compress_rate)

            reduced_index_s = int(s.shape[0] * compress_rate)

            routing_weights.append(v[:reduced_index_s, :])
            routing_sigmas.append(s[:reduced_index_s])

        return torch.stack(routing_weights), torch.stack(routing_sigmas), None


def compress_svd_dict(svd_dict, compress_rate):
    with torch.no_grad():
        for dataset, svd in tqdm(svd_dict.items(), desc="Compressing SVD"):
            for key, layer in svd.items():
                if "text_projection" in key:
                    continue
                if "dim1" in layer.keys() or "model.logit_scale" == key:
                    continue

                s = layer["s"]
                reduced_index_s = int(s.shape[0] * compress_rate)

                layer["u"] = layer["u"][:, :reduced_index_s]
                layer["s"] = s[:reduced_index_s]
                layer["v"] = layer["v"][:reduced_index_s, :]

        return svd_dict


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
                new_key = key.replace(".transformer", "")

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

    compression_factor = compression_factor or len(datasets)
    compression_ratio = 1 / compression_factor
    pylogger.info(f"Using compression ratio: {compression_ratio:.4f}")

    svd_path = (
        Path(svd_path).with_suffix("").as_posix() + f"_compress_{compression_factor}.pt"
    )

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


def whiten(x):
    """
    Compute the whitened transformation of the input matrix x.

    Parameters:
    x : np.ndarray
        Input data matrix of shape (n_samples, n_features)

    Returns:
    np.ndarray
        Whitened data matrix
    """
    cov = x.T @ x  # Compute the covariance-like matrix
    eigvals, eigvecs = np.linalg.eigh(cov)  # Eigen decomposition
    eigvals = np.where(
        eigvals > 1e-10, eigvals, 1e-10
    )  # Avoid sqrt of non-positive values
    whitening_matrix = (
        eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    )  # Whitening matrix
    whitened_x = x @ whitening_matrix  # Apply whitening transformation

    return whitened_x


import copy
import os
import torch
import logging
from tqdm import tqdm
from typing import Tuple

pylogger = logging.getLogger(__name__)


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
def sum_svd_no_redundant_tasks(
    ref_state_dict: dict,
    svd_dict: dict,
    similarity_threshold,
    device: str = "cuda",
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
                delta_layer_svd = svd_dict[dataset][layer_name]
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
                match = False
                for j, accepted in enumerate(accepted_deltas):
                    for accepted_flat in accepted:
                        sim = measure_cosine_similarity(delta_flat, accepted_flat)
                        if sim > similarity_threshold:
                            accepted_tasks[j].append((u, s, v))
                            accepted_deltas[j].append(delta_flat)
                            match = True
                            pylogger.info(
                                f"Merging {datasets[i]} and {datasets[j]} at layer {layer_name} due to similarity {sim}"
                            )
                            break

                if not match:
                    # If no overlap > threshold, accept it
                    accepted_tasks.append([(u, s, v)])
                    accepted_deltas.append([delta_flat])

            else:
                # For 1D layers, we do the usual average
                delta_layer = svd_dict[dataset][layer_name]["dim1"].to(device)
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
            continue

        averaged_tasks = []

        for tasks in accepted_tasks:
            if len(tasks) == 1:
                averaged_tasks.append(tasks[0])
            else:
                deltas = torch.stack(
                    [u @ torch.diag_embed(s) @ v for (u, s, v) in tasks]
                )
                deltas = deltas.mean(dim=0)
                u, s, v = compute_svd_and_compress(deltas, compress_ratio=1.0)
                averaged_tasks.append((u, s, v))

        # Build the big (sum_u, sum_s, sum_v) from accepted tasks
        # We do the same "concatenate columns" approach
        # first, figure out total rank
        total_rank = sum(task_s.shape[0] for (_, task_s, _) in averaged_tasks)

        # Prepare placeholders
        sum_u = torch.zeros(
            averaged_tasks[0][0].shape[0], total_rank, device=device
        )  # [m, total_rank]
        sum_s = torch.zeros(total_rank, device=device)
        sum_v = torch.zeros(total_rank, averaged_tasks[0][2].shape[1], device=device)

        offset = 0
        for u_i, s_i, v_i in averaged_tasks:
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


@torch.no_grad()
def sum_svd_no_redundant_space(
    ref_state_dict: dict,
    svd_dict: dict,
    similarity_threshold,
    device: str = "cuda",
):

    aggregated_model_dict = ref_state_dict
    layer_names = list(aggregated_model_dict.keys())
    datasets = list(svd_dict.keys())

    for layer_name in tqdm(layer_names, desc="Summing SVD"):
        # Determine if the layer is a 2D weight matrix or not.
        is_layer_matrix = aggregated_model_dict[layer_name].dim() == 2

        # For non-matrix layers (or special layers like text_projection), do the usual averaging.
        if not is_layer_matrix or "text_projection" in layer_name:
            for i, dataset in enumerate(datasets):
                if "text_projection" in layer_name:
                    continue
                delta_layer = svd_dict[dataset][layer_name].get("dim1", None)
                if delta_layer is not None:
                    delta_layer = delta_layer.to(device)
                    if i == 0:
                        aggregated_model_dict[layer_name] = delta_layer
                    else:
                        aggregated_model_dict[layer_name] += (
                            delta_layer - aggregated_model_dict[layer_name]
                        ) / (i + 1)
            continue

        accepted_components = []
        accepted_vs = []  # stores accepted singular vectors (as 1D tensors)

        for i, dataset in enumerate(datasets):
            if "text_projection" in layer_name:
                continue

            # Retrieve the SVD factors for this dataset and layer.
            svd_factors = svd_dict[dataset][layer_name]
            u = svd_factors["u"].to(device)  # shape: [m, r]
            s = svd_factors["s"].to(device)  # shape: [r]
            v = svd_factors["v"].to(device)  # shape: [r, n]
            rank_i = s.shape[0]

            # For the first dataset, accept all singular vectors.
            if i == 0 and len(accepted_components) == 0:
                for j in range(rank_i):
                    accepted_components.append((u[:, j], s[j], v[j, :].view(-1)))
                    accepted_vs.append(v[j, :].view(-1))
            else:
                # For each singular vector in this dataset, compare with already accepted ones.
                for j in range(rank_i):
                    new_v = v[j, :].view(-1)
                    skip = False
                    for stored_v in accepted_vs:
                        sim = measure_cosine_similarity(new_v, stored_v)
                        if sim > similarity_threshold:
                            pylogger.info(f"skipping vector {j} for task {dataset}")
                            # Too similar: discard this singular vector.
                            skip = True
                            break
                    if not skip:
                        accepted_components.append((u[:, j], s[j], new_v))
                        accepted_vs.append(new_v)

        # If no singular vectors were accepted, keep the original pretrained weights.
        if len(accepted_components) == 0:
            continue

        # Build aggregated SVD factors from the accepted components.
        total_rank = len(accepted_components)
        m = accepted_components[0][0].shape[0]
        n = accepted_components[0][2].shape[0]
        sum_u = torch.zeros(m, total_rank, device=device)
        sum_s = torch.zeros(total_rank, device=device)
        sum_v = torch.zeros(total_rank, n, device=device)

        for idx, (u_comp, s_comp, v_comp) in enumerate(accepted_components):
            sum_u[:, idx] = u_comp
            sum_s[idx] = s_comp
            sum_v[idx, :] = v_comp

        # Use a multi-step SVD approach on the aggregated factors.
        u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
        u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)

        merged = torch.linalg.multi_dot((u_u, torch.diag(sum_s), v_u, u_v, v_v))
        aggregated_model_dict[layer_name] = merged.to(device)

    return aggregated_model_dict


def svd_tuple(svd_dict, dataset, layer_name, device="cuda"):
    delta_layer_svd = svd_dict[dataset][layer_name]
    return (
        delta_layer_svd["u"].to(device),
        delta_layer_svd["s"].to(device),
        delta_layer_svd["v"].to(device),
    )


def unpack_svd_dict(svd_dict, dataset, layer_name, mode="full", device="cuda"):
    """
    Extracts and returns U, V, or the fully reconstructed delta matrix from the SVD dictionary.

    Args:
        svd_dict (dict): Dictionary containing SVD decompositions.
        dataset (str): Dataset name.
        layer_name (str): Layer name.
        mode (str): "u" for U matrix, "v" for V matrix, "full" for U @ S @ V.
        device (str): Device to move tensors to.

    Returns:
        torch.Tensor: The requested matrix based on the mode.
    """
    delta_layer_svd = svd_dict[dataset][layer_name]
    u, s, v = (
        delta_layer_svd["u"].to(device),
        delta_layer_svd["s"].to(device),
        delta_layer_svd["v"].to(device),
    )

    if mode == "u":
        return u
    elif mode == "v":
        return v
    elif mode == "full":
        return u @ torch.diag_embed(s) @ v
    else:
        raise ValueError("Invalid mode. Choose from 'u', 'v', or 'full'.")


def cosine_sim(delta1, delta2):
    return (
        delta1.flatten() @ delta2.flatten() / (torch.norm(delta1) * torch.norm(delta2))
    )


def right_sar(delta, v):
    proj = delta @ v.T @ v
    return torch.norm(proj, p="fro") / (torch.norm(delta, p="fro") + 1e-3)


def left_sar(delta, u):
    proj = u @ u.T @ delta
    return torch.norm(proj, p="fro") / torch.norm(delta, p="fro")


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
        new_key = layer_name.replace(".transformer", "")
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
