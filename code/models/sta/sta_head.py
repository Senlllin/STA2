"""Symmetry parameter prediction head."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class STAParamHead(nn.Module):
    """Predict symmetry plane parameters and confidence.

    Parameters
    ----------
    in_dim : int
        Dimension of the incoming global feature.
    hidden_dim : int, optional
        Width of the MLP, by default ``256``.
    dropout : float, optional
        Dropout probability, by default ``0.0``.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 256, dropout: float = 0.0) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 5),
        )
        self.norm = nn.LayerNorm(5)

    def forward(self, feat: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass.

        Parameters
        ----------
        feat : Tensor
            Global feature ``(B, C)``.

        Returns
        -------
        Tuple[Tensor, Tensor, Tensor]
            Normal ``(B,3)``, offset ``(B,1)`` and confidence ``alpha`` ``(B,1)``.
        """
        x = self.norm(self.mlp(feat))
        n = F.normalize(x[:, :3], dim=-1)
        d = x[:, 3:4]
        alpha = torch.sigmoid(x[:, 4:5])
        return n, d, alpha