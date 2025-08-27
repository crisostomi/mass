import copy
from typing import Dict, List

import torch

from mass.merger.merger import TaskVectorBasedMerger
from mass.modules.encoder import ImageEncoder
from mass.utils.utils import (
    apply_dict_to_model,
    compute_task_dict,
    print_memory,
    is_matrix,
)


def _compose(A: torch.Tensor, B: torch.Tensor, how: str = "mean") -> torch.Tensor:
    """
    Combine two non-negative importance matrices A and B into an effective one.

    Supported modes: 'geom', 'min', 'max', 'harm', 'mean', 'sum'.
    Defaults to 'mean'.
    """
    mode = (how or "mean").lower()
    if mode in ("geom", "geometric"):
        return torch.sqrt((A.clamp(min=0.0) + 1e-12) * (B.clamp(min=0.0) + 1e-12))
    if mode in ("min",):
        return torch.minimum(A, B)
    if mode in ("max", "maximum"):
        return torch.maximum(A, B)
    if mode in ("harm", "harmonic"):
        eps = 1e-12
        return (2.0 * (A * B)) / (A + B + eps)
    if mode in ("sum",):
        return A + B
    # default: mean
    return 0.5 * (A + B)


class CircuitsMerger(TaskVectorBasedMerger):

    def __init__(
        self,
        svd_path,
        svd_compress_factor,
        *,
        # soft-mask hyperparameters (single-layer variant)
        s_quantile: float = 1e-5,
        u_quantile: float = 0.9999,
        beta_scale: float = 1e-5,
        # composition of row/col importances
        compose: str = "mean",
        # optionally normalise singular values per layer
        normalise_s: bool = False, #usually false
    ):
        super().__init__()

        # kept for API parity with other mergers (not used directly here)
        self.svd_path = svd_path
        self.svd_compress_factor = svd_compress_factor

        # compsoft-inspired controls
        self.s_quantile = float(s_quantile)
        self.u_quantile = float(u_quantile)
        self.beta_scale = float(beta_scale)
        self.compose = compose
        self.normalise_s = bool(normalise_s)

    def merge(self, base_model, finetuned_models):
        # 1) Compute per-dataset task deltas (ΔW for each parameter)
        task_dicts: Dict[str, Dict[str, torch.Tensor]] = {}
        datasets: List[str] = list(finetuned_models.keys())

        for dataset in datasets:
            task_dicts[dataset] = compute_task_dict(
                base_model.state_dict(), finetuned_models[dataset]
            )
            # free memory as in other mergers
            del finetuned_models[dataset]
            torch.cuda.empty_cache()

        print_memory("after computing task dicts")

        # 2) Precompute SVDs and neighbor bases for compositional alignment
        aggregated: Dict[str, torch.Tensor] = {}
        base_state = base_model.state_dict()
        # collect ordered matrix layer keys for simple prev/next mapping
        matrix_keys: List[str] = [
            k for k, w in base_state.items()
            if (w.dtype not in [torch.int64, torch.uint8]) and (w.dim() >= 2) and is_matrix(w)
        ]
        # basic stable order: by key name
        matrix_keys.sort()

        # precompute SVDs for all matrix layers
        svd_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        for k in matrix_keys:
            Wk = base_state[k].cuda()
            try:
                Uk, Sk, Vhk = torch.linalg.svd(Wk, full_matrices=False)
            except RuntimeError:
                Uk, Sk, Vhk = torch.linalg.svd(Wk.cpu(), full_matrices=False)
                Uk = Uk.cuda(); Sk = Sk.cuda(); Vhk = Vhk.cuda()
            if self.normalise_s:
                s_sum = Sk.sum().clamp(min=1e-12)
                Sk = Sk / s_sum
            svd_cache[k] = {"U": Uk, "S": Sk, "Vt": Vhk}

        # neighbor helpers
        def _prev_key(idx: int) -> str | None:
            return matrix_keys[idx - 1] if idx - 1 >= 0 else None
        def _next_key(idx: int) -> str | None:
            return matrix_keys[idx + 1] if idx + 1 < len(matrix_keys) else None

        # 3) CompSoft-inspired masked combination with compositional alignment
        for key, base_weight in base_state.items():
            # skip non-floating tensors
            if base_weight.dtype in [torch.int64, torch.uint8]:
                continue

            # collect available deltas for this key
            per_task_deltas: List[torch.Tensor] = [
                task_dicts[ds][key]
                for ds in datasets
                if key in task_dicts[ds]
            ]
            if len(per_task_deltas) == 0:
                continue

            # matrix weights: apply compsoft with 1-hop alignment in SVD basis
            if base_weight.dim() >= 2 and is_matrix(base_weight):
                # current layer svd
                Uc = svd_cache[key]["U"]
                Sc = svd_cache[key]["S"]
                Vtc = svd_cache[key]["Vt"]
                r = Sc.shape[0]

                # neighbor bases (1-hop): previous.U and next.V
                try:
                    idx = matrix_keys.index(key)
                except ValueError:
                    idx = -1
                U_prev = None
                V_next = None
                pk = _prev_key(idx)
                nk = _next_key(idx)
                if pk is not None:
                    U_prev = svd_cache[pk]["U"]  # d_{l-1} x r_{l-1}
                if nk is not None:
                    V_next = svd_cache[nk]["Vt"].transpose(-2, -1)  # d_l x r_{l+1}

                # α_in(j)
                if U_prev is not None:
                    # align dims best-effort
                    if U_prev.shape[0] != Vtc.shape[1]:
                        d_shared = min(U_prev.shape[0], Vtc.shape[1])
                        U_prev_eff = U_prev[:d_shared, :]
                        V_in_eff = Vtc.transpose(-2, -1)[:d_shared, :]
                    else:
                        U_prev_eff = U_prev
                        V_in_eff = Vtc.transpose(-2, -1)
                    Tin = (U_prev_eff.transpose(-2, -1) @ V_in_eff).abs()  # r_{l-1} x r
                    alpha_in = Tin.mean(dim=0)
                else:
                    alpha_in = torch.ones(r, device=Sc.device, dtype=Sc.dtype)

                # α_out(i)
                if V_next is not None:
                    if Uc.shape[0] != V_next.shape[0]:
                        d_shared = min(Uc.shape[0], V_next.shape[0])
                        U_out_eff = Uc[:d_shared, :]
                        V_next_eff = V_next[:d_shared, :]
                    else:
                        U_out_eff = Uc
                        V_next_eff = V_next
                    Tout = (U_out_eff.transpose(-2, -1) @ V_next_eff).abs()  # r x r_{l+1}
                    alpha_out = Tout.mean(dim=1)
                else:
                    alpha_out = torch.ones(r, device=Sc.device, dtype=Sc.dtype)

                # compositional importances
                row_imp = (Sc * alpha_out).clamp(min=0.0)
                col_imp = (Sc * alpha_in).clamp(min=0.0)
                S_row = row_imp.unsqueeze(1).expand(r, r)
                S_col = col_imp.unsqueeze(0).expand(r, r)
                S_eff = _compose(S_row, S_col, how=self.compose)

                # smooth thresholds for S and |C|
                zeta_s = torch.quantile(S_eff.reshape(-1), self.s_quantile)
                beta_s = self.beta_scale * zeta_s + 1e-8

                masked_deltas: List[torch.Tensor] = []
                for delta in per_task_deltas:
                    # project update into current basis
                    C = Uc.transpose(-2, -1) @ delta @ Vtc.transpose(-2, -1)
                    U_abs = C.abs()

                    # update-dependent magnitude threshold
                    zeta_u = torch.quantile(U_abs.reshape(-1), self.u_quantile)
                    beta_u = self.beta_scale * zeta_u + 1e-8

                    # two-quadrant smooth gate
                    cond1 = torch.sigmoid((U_abs - zeta_u) / beta_u) * torch.sigmoid(
                        (zeta_s - S_eff) / beta_s
                    )
                    cond2 = torch.sigmoid((S_eff - zeta_s) / beta_s) * torch.sigmoid(
                        (zeta_u - U_abs) / beta_u
                    )
                    M = torch.clamp(cond1 + cond2, 0.0, 1.0)

                    C_prime = M * C
                    delta_prime = Uc @ C_prime @ Vtc
                    masked_deltas.append(delta_prime)

                # aggregate masked updates across tasks (mean)
                aggregated[key] = torch.stack(masked_deltas, dim=0).mean(dim=0)

            else:
                # biases, layernorm params, etc.: simple mean
                aggregated[key] = torch.stack(per_task_deltas, dim=0).mean(dim=0)

        # 3) Apply aggregated task vector to a fresh copy of the base model
        merged_encoder: ImageEncoder = copy.deepcopy(base_model)
        merged_encoder = apply_dict_to_model(
            aggregated,
            merged_encoder,
        )

        return merged_encoder



