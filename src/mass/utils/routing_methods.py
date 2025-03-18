import torch


def get_mahalonobis_cov(v, eps=1e-3):
    n_tasks, r, d = v.shape
    cov_matrices = []
    task_means = []

    for t in range(n_tasks):
        vt = v[t]  # shape [r, d]
        mean_t = vt.mean(dim=0)  # shape [d]
        task_means.append(mean_t)
        cov_t = torch.cov(vt.T) + torch.eye(d, device=v.device) * eps
        cov_matrices.append(cov_t)

    cov_matrices = torch.stack(cov_matrices, dim=0)
    task_means = torch.stack(task_means, dim=0)

    L = torch.linalg.cholesky(cov_matrices)

    return torch.cholesky_inverse(L)


def mahalanobis(x, y, inv_cov):
    dist = torch.einsum(
        "Btd, tde, Bte -> Bt", (x.unsqueeze(1) - y), inv_cov, (x.unsqueeze(1) - y)
    )
    return torch.sqrt(torch.clamp(dist, min=0))


def get_distance(norm, v):
    if norm == "mahalanobis":
        inv_cov = get_mahalonobis_cov(v)
        return lambda x, y: mahalanobis(x, y, inv_cov)
    elif norm == "l2":
        return lambda x, y: torch.norm((x.unsqueeze(1) - y), dim=2)
    elif norm == "l1":
        return lambda x, y: torch.norm((x.unsqueeze(1) - y), dim=2, p=1)


def get_projector(norm, v, s):
    if norm == "mahalanobis":
        return lambda x: torch.einsum(
            "Btr, trd -> Btd", torch.einsum("Bd, tdr -> Btr", x, v.transpose(1, 2)), v
        )
    elif norm == "l1":
        return lambda x: torch.einsum(
            "Btr, trd -> Btd", torch.einsum("Bd, tdr -> Btr", x, v.transpose(1, 2)), v
        )
    elif norm == "l2":
        return lambda x: torch.einsum(
            "Btr, trd -> Btd", torch.einsum("Bd, tdr -> Btr", x, v.transpose(1, 2)), v
        )


def compute_residual_norm(x, v, s=None, norm="l2"):

    valid_norms = ["l2", "l1", "mahalanobis"]

    if norm not in valid_norms:
        raise NotImplementedError(f"Valid norms are {valid_norms}")

    p = get_projector(norm, v, s)

    x_v = p(x)

    d = get_distance(norm, v)

    return d(x, x_v)
