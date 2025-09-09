import torch



def get_distance(norm, v):
    assert norm == "l2", "Only l2 norm is supported for distance"
    return lambda x, y: torch.norm((x.unsqueeze(1) - y), dim=2)


def get_projector(norm, v, s):
    assert norm == "l2", "Only l2 norm is supported for projector"
    return lambda x: torch.einsum(
        "Btr, trd -> Btd", torch.einsum("Bd, tdr -> Btr", x, v.transpose(1, 2)), v
    )


def compute_residual_norm(x, v, s=None, norm="l2"):

    assert norm == "l2", "Only l2 norm is supported for residual norm"

    p = get_projector(norm, v, s)

    x_v = p(x)

    d = get_distance(norm, v)

    return d(x, x_v)
