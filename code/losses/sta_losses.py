"""Symmetry-related loss functions for STA."""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from code.models.sta.sta_math import reflect_points


@dataclass(frozen=True)
class STALosses:
    """Collection of symmetry-aware loss terms."""

    w_cd: float = 1.0
    w_emd: float = 0.0
    w_symp: float = 0.1
    w_symg: float = 0.1

    def chamfer_l1(self, x: Tensor, y: Tensor) -> Tensor:
        """L1 Chamfer distance between point sets ``x`` and ``y``."""

        dist = torch.cdist(x, y, p=1)
        return dist.min(dim=-1).values.mean(-1) + dist.min(dim=-2).values.mean(-1)

    def sym_plane_loss(self, n: Tensor) -> Tensor:
        """Regularize normals to unit length."""

        return (torch.linalg.norm(n, dim=-1) - 1.0).abs().mean()

    def sym_geo_loss(
        self,
        y: Tensor,
        n: Tensor,
        d: Tensor,
        alpha: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Encourage geometric symmetry of predictions."""

        y_ref = reflect_points(y, n, d)
        if mask is not None:
            y = y[mask]
            y_ref = y_ref[mask]
        cd = self.chamfer_l1(y, y_ref)
        return (alpha.view(-1) * cd).mean()

    def __call__(
        self,
        pred: Tensor,
        target: Tensor,
        n: Tensor,
        d: Tensor,
        alpha: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute the weighted STA loss."""

        l_cd = self.chamfer_l1(pred, target)
        l_sp = self.sym_plane_loss(n)
        l_sg = self.sym_geo_loss(pred, n, d, alpha, mask)
        return self.w_cd * l_cd + self.w_symp * l_sp + self.w_symg * l_sg