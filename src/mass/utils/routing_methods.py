import torch


def get_distance(norm, cov):
    assert norm in ["l2", "mahalanobis"], "Only l2 norm is supported for distance"
    if norm == "mahalanobis":
        return lambda x, y: torch.sqrt(torch.einsum('btd,dr,btr->bt', (x.unsqueeze(1)-y), cov, (x.unsqueeze(1)-y))) # 128 x 8 x 768 @ 128 x 8 x 768 = 128 x 8 
    else: 
        return lambda x, y: torch.norm((x.unsqueeze(1) - y), dim=2)


def get_projector(norm, v, s):
    return lambda x: torch.einsum(
        "Btr, trd -> Btd", torch.einsum("Bd, tdr -> Btr", x, v.transpose(1, 2)), v
    )


def compute_residual_norm(x, v, s=None, cov=None, norm="l2"):
    assert (norm == "mahalanobis" and cov is not None) or (norm == "l2"), "Covariance matrix must be provided for mahalanobis norm"

    p = get_projector(norm, v, s)

    x_v = p(x)

    d = get_distance(norm, cov if norm == "mahalanobis" else None)
    print(f"x shape: {x.shape}, x_v shape: {x_v.shape}, cov shape: {cov.shape if cov is not None else 'N/A'}")
    return d(x, x_v)


    
    
    
    
