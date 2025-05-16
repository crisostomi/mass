import copy
from tqdm import tqdm
import torch
from typing import Union
import numpy as np


def inner_product(point, tangent_vector_a, tangent_vector_b):
    return torch.tensordot(tangent_vector_a, tangent_vector_b, dims=tangent_vector_a.ndim)


def dist(point_a, point_b):
    inner = max(min(inner_product(point_a, point_a, point_b), 1), -1)
    return torch.arccos(inner)


def norm(point, tangent_vector):
    return torch.norm(tangent_vector)


def projection(point, vector):
    return vector - inner_product(point, point, vector) * point


def ell_q(q, p):
    vector = projection(q, p - q)
    distance = dist(q, p)
    epsilon = np.finfo(np.float64).eps
    factor = (distance + epsilon) / (norm(q, vector) + epsilon)
    return factor * vector


def exp_q(point, tangent_vector):
    norm_value = norm(point, tangent_vector)
    return point * torch.cos(norm_value) + tangent_vector * torch.sinc(norm_value / np.pi)


def spherical_weighted_average(points, weights, tol=1e-8, max_iter=1000, dim=2, verbose=False):
    """
    Compute the spherical weighted average of points on a sphere with given weights using PyTorch.

    Args:
    - points (torch.Tensor): A tensor of shape (n, d+1) representing n points on the d-dimensional sphere S^d.
    - weights (torch.Tensor): A tensor of shape (n,) representing the non-negative weights with sum 1.
    - tol (float): Tolerance for the stopping criterion based on the norm of u.
    - max_iter (int): Maximum number of iterations for the main loop.

    Returns:
    - q (torch.Tensor): The spherical weighted average of the input points.
    """

    print(f"Points shape: {points.shape}")
    print(f"Weights shape: {weights.shape}")

    points = points.cuda()
    weights = weights.cuda()

    points_copy = copy.deepcopy(points.clone())

    points = torch.nn.functional.normalize(points, p=2, dim=1)

    # (num_points, d+1)
    assert points.shape[-1] == dim + 1, f"points.shape = {points.shape}, dim = {dim}"

    # points have shape (num_tasks, num_params)
    # weights have shape (num_tasks,)

    with torch.no_grad():
        # Ensure weights sum to 1
        weights = weights / weights.sum()

        # Initialization, q has shape (d+1,)
        q = (weights[:, None] * points).sum(dim=0)
        q = q / (torch.norm(q))

        assert q.shape[0] == dim + 1, f"q.shape = {q.shape}, dim = {dim}"

        for _ in tqdm(range(max_iter)):
            # Compute p_i^* for each point

            # (num_points, d+1)
            p_i_stars = torch.stack([ell_q(q, p) for p in points])
            u = (weights[:, None] * (p_i_stars)).sum(dim=0)
            q = exp_q(q, u)

            q = q / (torch.norm(q))

            # Check if u is sufficiently small
            if torch.norm(u) < tol:
                break

            if verbose:
                print(f"Norm: {torch.norm(u)}")

            # solve for alphas such that alphas * points = q and such that sum(alphas) = 1

        # (num_points, d+1) --> (2, 32762332 + 1)
        constraint_weight = 100
        weights_sum_to_one = torch.full(
            (points.shape[0],), fill_value=constraint_weight, device=points.device
        ).unsqueeze(1)
        points_with_constraint = torch.cat([points, weights_sum_to_one], dim=1)
        q_with_constraint = torch.cat([q, torch.tensor([constraint_weight], device=points.device)])

        # solve the system of linear equations Ax = B, where A ~ (num_eqs, num_variables), x ~ (num_variables), B ~ (num_eqs)
        # for us, each row of A is a point and the last row is the constraint
        # this means that for each param in the models, we have an equation that
        alphas = torch.linalg.lstsq(points_with_constraint.T, q_with_constraint).solution

        interpolated_vector = (alphas[:, None] * points_copy).sum(dim=0)

        print(f"Found spherical interpolation coefficients: {alphas}, summing to {alphas.sum()}")
        print(f"Reconstruction error: {(interpolated_vector - q).sum()}")

        return interpolated_vector


def slerp(
    t: Union[float, torch.Tensor],
    v0: torch.Tensor,
    v1: torch.Tensor,
    DOT_THRESHOLD: float = 0.9995,
    eps: float = 1e-8,
):
    """
    Spherical linear interpolation using PyTorch

    Args:
        t (float/torch.Tensor): Float value between 0.0 and 1.0
        v0 (torch.Tensor): Starting vector
        v1 (torch.Tensor): Final vector
        DOT_THRESHOLD (float): Threshold for considering the two vectors as
                               colinear. Not recommended to alter this.
    Returns:
        v2 (torch.Tensor): Interpolation vector between v0 and v1
    """
    # Ensure inputs are tensors
    v0 = torch.tensor(v0, dtype=torch.float32).cuda()
    v1 = torch.tensor(v1, dtype=torch.float32).cuda()

    v0_copy = v0.clone()
    v1_copy = v1.clone()

    # Normalize the vectors to get the directions and angles
    v0 = v0 / (torch.norm(v0) + eps)
    v1 = v1 / (torch.norm(v1) + eps)

    # Dot product with the normalized vectors
    dot = torch.sum(v0 * v1)

    # If absolute value of dot product is almost 1, vectors are ~colinear, so use lerp
    if torch.abs(dot) > DOT_THRESHOLD:
        print("colinear vectors")
        return None

    # Calculate initial angle between v0 and v1
    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)

    # Angle at timestep t
    theta_t = theta_0 * t
    sin_theta_t = torch.sin(theta_t)

    # Finish the slerp algorithm
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0

    res = s0 * v0_copy + s1 * v1_copy

    print(f"Interpolation coefficients: {s0, s1}")
