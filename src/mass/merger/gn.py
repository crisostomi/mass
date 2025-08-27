import copy
import os
from typing import Dict, List, Tuple

import torch

try:
    import matplotlib.pyplot as plt  # optional, only used if plots_dir is provided
except Exception:  # pragma: no cover - plotting is optional
    plt = None

from mass.merger.merger import TaskVectorBasedMerger
from mass.modules.encoder import ImageEncoder
from mass.utils.utils import (
    apply_dict_to_model,
    compute_task_dict,
    is_matrix,
    print_memory,
)


# Lightweight per-weight SVD cache keyed by tensor data_ptr()
_SVD_CACHE: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = {}


def _get_cached_svd(
    weight: torch.Tensor, *, svd_every: int = 1, full_matrices: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute or retrieve a cached SVD(U, S, Vt) of `weight`.

    The cache is refreshed every `svd_every` calls per unique tensor identity.
    """
    assert weight.dim() >= 2, "SVD requires a matrix-like tensor"

    key = int(weight.data_ptr())
    calls = 0
    if key in _SVD_CACHE:
        U, S, Vt, calls = _SVD_CACHE[key]
        calls += 1
        if calls % max(int(svd_every), 1) != 0:
            return U, S, Vt

    W = weight
    device = W.device
    try:
        U, S, Vt = torch.linalg.svd(W, full_matrices=full_matrices)
    except RuntimeError:
        Uc, Sc, Vtc = torch.linalg.svd(W.cpu(), full_matrices=full_matrices)
        U, S, Vt = Uc.to(device), Sc.to(device), Vtc.to(device)

    _SVD_CACHE[key] = (U, S, Vt, 0)
    return U, S, Vt


def soft_update_weights(
    current_weight,
    update_weight,
    plots_dir=None,
    beta_scale: float = 0.1,
    return_stats: bool = False,
    s_quantile: float = 0.1e-5,
    u_quantile: float = 0.9999,
    svd_every: int = 1,
):
    """Soft spectral masking of the update tensor, used for model merging.

    This function projects the update into the SVD basis of the current weight,
    applies a smooth, two-quadrant mask controlled by singular value and update
    magnitudes, and then maps the masked update back to the original basis.
    """
    if update_weight.dim() < 2:
        return update_weight

    # SVD of current weight (cached)
    U_current, S_current, Vt_current = _get_cached_svd(
        current_weight, svd_every=svd_every, full_matrices=False
    )

    if plots_dir is not None and plt is not None:
        os.makedirs(plots_dir, exist_ok=True)
        plt.figure(figsize=(10, 4))
        plt.scatter(x=range(len(S_current)), y=S_current.detach().cpu(), label="S_current")
        plt.legend()
        plt.title("Singular Value Distributions")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "svd_distributions.png"))
        plt.close()

    # Project update onto current weight basis
    C = U_current.transpose(-2, -1) @ update_weight @ Vt_current.transpose(-2, -1)
    u = torch.abs(C)

    # Energy thresholds
    zeta_s = torch.quantile(S_current, s_quantile)
    zeta_u = torch.quantile(u, u_quantile)

    # Broadcast singular values to match C matrix shape (r x r)
    r = S_current.shape[0]
    s_matrix = S_current.unsqueeze(1).expand(r, r)

    # Soft masks via sigmoid blending
    beta_s = beta_scale * zeta_s + 1e-8
    beta_u = beta_scale * zeta_u + 1e-8

    soft_cond1 = torch.sigmoid((u - zeta_u) / beta_u) * torch.sigmoid((zeta_s - s_matrix) / beta_s)
    soft_cond2 = torch.sigmoid((s_matrix - zeta_s) / beta_s) * torch.sigmoid((zeta_u - u) / beta_u)
    M = torch.clamp(soft_cond1 + soft_cond2, 0.0, 1.0)

    C_prime = M * C
    new_update = U_current @ C_prime @ Vt_current

    if return_stats:
        stats = {
            "s_current_mean": S_current.detach().cpu().mean().item(),
            "zeta_s": zeta_s.detach().cpu().item(),
            "zeta_u": zeta_u.detach().cpu().item(),
            "masked_elements_ratio": M.detach().cpu().mean().item(),
            "beta_scale": float(beta_scale),
        }
        return new_update, stats
    else:
        return new_update


class GNMerger(TaskVectorBasedMerger):
    """
    Merger that aggregates updates across multiple finetuned models using
    soft spectral masking on each matrix weight, following the same structure
    as `CircuitsMerger`.
    """

    def __init__(
        self,
        svd_path,
        svd_compress_factor,
        *,
        s_quantile: float = 0.1e-10,
        u_quantile: float = 0.999999999,
        beta_scale: float = 1e-10,
        svd_every: int = 1,
    ):
        super().__init__()
        self.s_quantile = float(s_quantile)
        self.u_quantile = float(u_quantile)
        self.beta_scale = float(beta_scale)
        self.svd_every = int(svd_every)

    def merge(self, base_model, finetuned_models):
        # 1) Compute per-dataset task deltas (ΔW for each parameter)
        task_dicts: Dict[str, Dict[str, torch.Tensor]] = {}
        datasets: List[str] = list(finetuned_models.keys())

        for dataset in datasets:
            task_dicts[dataset] = compute_task_dict(
                base_model.state_dict(), finetuned_models[dataset]
            )
            del finetuned_models[dataset]
            torch.cuda.empty_cache()

        print_memory("after computing task dicts")

        # 2) Aggregate with soft spectral masking per layer
        aggregated: Dict[str, torch.Tensor] = {}
        base_state = base_model.state_dict()

        for key, base_weight in base_state.items():
            if base_weight.dtype in [torch.int64, torch.uint8]:
                continue

            per_task_deltas: List[torch.Tensor] = [
                task_dicts[ds][key] for ds in datasets if key in task_dicts[ds]
            ]
            if len(per_task_deltas) == 0:
                continue

            if base_weight.dim() >= 2 and is_matrix(base_weight):
                masked_deltas: List[torch.Tensor] = []
                current_W = base_weight.cuda()
                for delta in per_task_deltas:
                    masked_delta = soft_update_weights(
                        current_W,
                        delta.cuda(),
                        plots_dir=None,
                        beta_scale=self.beta_scale,
                        return_stats=False,
                        s_quantile=self.s_quantile,
                        u_quantile=self.u_quantile,
                        svd_every=self.svd_every,
                    )
                    masked_deltas.append(masked_delta)

                aggregated[key] = torch.stack(masked_deltas, dim=0).mean(dim=0)
            else:
                aggregated[key] = torch.stack(per_task_deltas, dim=0).mean(dim=0)

        # 3) Apply aggregated task vector to a fresh copy of the base model
        merged_encoder: ImageEncoder = copy.deepcopy(base_model)
        merged_encoder = apply_dict_to_model(
            aggregated,
            merged_encoder,
        )

        return merged_encoder


