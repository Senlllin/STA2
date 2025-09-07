"""Symmetry-aware transformer components."""

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from code.models.sta.sta_math import pairwise_mirror_dist


class STAAttentionBias(nn.Module):
    """Compute an additive attention bias based on mirror proximity."""

    def __init__(self, beta: float = 1.0, k_mirror: int = 8, temperature: float = 0.07) -> None:
        super().__init__()
        self.beta = beta
        self.k = k_mirror
        self.temperature = temperature

    def forward(self, anchors: Tensor, n: Tensor, d: Tensor) -> Tensor:
        """Build the bias matrix.

        Parameters
        ----------
        anchors : Tensor
            Token coordinates ``(B, M, 3)``.
        n : Tensor
            Plane normals ``(B, 3)``.
        d : Tensor
            Plane offsets ``(B, 1)``.

        Returns
        -------
        Tensor
            Additive bias of shape ``(B, M, M)``.
        """
        dist = pairwise_mirror_dist(anchors, n, d)
        # negative for smallest; (B,M,k)
        _, idx = torch.topk(-dist, self.k, dim=-1)
        B, M = dist.shape[:2]
        bias = torch.zeros(B, M, M, device=anchors.device)
        expand_idx = idx.unsqueeze(2)
        bias.scatter_(2, expand_idx, self.beta)
        return bias / self.temperature


class SymAwareDecoder(nn.Module):
    """Wrapper decoder that injects symmetry-aware bias."""

    def __init__(self, dim: int, num_heads: int, use_sta_bias: bool = True, **bias_kwargs) -> None:
        super().__init__()
        self.use_sta_bias = use_sta_bias
        self.mha = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.bias = STAAttentionBias(**bias_kwargs)

    def forward(
        self,
        x: Tensor,
        anchors: Tensor,
        n: Tensor,
        d: Tensor,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Run the decoder with optional symmetry bias."""

        if self.use_sta_bias:
            b_sym = self.bias(anchors, n, d)
            attn_mask = b_sym if attn_mask is None else attn_mask + b_sym
        out, _ = self.mha(x, x, x, attn_mask=attn_mask)
        return out