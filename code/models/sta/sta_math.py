"""Symmetry math utilities.

These helpers implement plane normalization, point reflection and pairwise
mirror distances used by STA modules. Functions are intentionally kept
stateless for ease of testing.
"""

from typing import Tuple

import torch
from torch import Tensor


def normalize_plane(n: Tensor, d: Tensor) -> Tuple[Tensor, Tensor]:
    """Normalize plane parameters.

    Parameters
    ----------
    n : Tensor
        Normal vectors of shape ``(B, 3)``.
    d : Tensor
        Offsets of shape ``(B, 1)``.

    Returns
    -------
    Tuple[Tensor, Tensor]
        Normalized ``(n, d)`` so that ``||n||_2 == 1``.
    """
    # (B,1) avoid divide-by-zero
    norm = torch.linalg.norm(n, dim=-1, keepdims=True).clamp_min(1e-6)
    n_norm = n / norm
    d_norm = d / norm
    return n_norm, d_norm


def reflect_points(x: Tensor, n: Tensor, d: Tensor) -> Tensor:
    """Reflect points about a plane.

    Parameters
    ----------
    x : Tensor
        Input points of shape ``(B, N, 3)``.
    n : Tensor
        Unit normals ``(B, 3)``.
    d : Tensor
        Offsets ``(B, 1)``.

    Returns
    -------
    Tensor
        Reflected points with shape ``(B, N, 3)``.
    """
    n = n.reshape(n.shape[0], 1, 3)
    d = d.reshape(d.shape[0], 1, 1)
    t = (x * n).sum(-1, keepdim=True) + d
    return x - 2.0 * t * n


def pairwise_mirror_dist(x_tokens: Tensor, n: Tensor, d: Tensor) -> Tensor:
    """Distances between tokens and their mirrored counterparts.

    Parameters
    ----------
    x_tokens : Tensor
        Token coordinates ``(B, M, 3)``.
    n : Tensor
        Plane normals ``(B, 3)``.
    d : Tensor
        Plane offsets ``(B, 1)``.

    Returns
    -------
    Tensor
        Pairwise Euclidean distances ``(B, M, M)``.
    """
    mirrored = reflect_points(x_tokens, n, d)
    return torch.cdist(x_tokens, mirrored, p=2)