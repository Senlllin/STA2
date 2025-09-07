"""Utility to evaluate symmetry score of a point cloud."""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from code.losses.sta_losses import STALosses

logger = logging.getLogger(__name__)


def load_pcd(path: Path) -> torch.Tensor:
    """Load a point cloud from ``path``."""

    arr = np.loadtxt(path).astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description="Evaluate symmetry of a point cloud")
    parser.add_argument("--input", type=Path, required=True, help="Path to point cloud (txt with Nx3)")
    parser.add_argument("--n", nargs=3, type=float, required=True, help="Plane normal")
    parser.add_argument("--d", type=float, default=0.0, help="Plane offset")
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    logger.info("Loading point cloud from %s", args.input)

    pts = load_pcd(args.input)
    n = torch.tensor([args.n], dtype=torch.float32)
    d = torch.tensor([[args.d]], dtype=torch.float32)
    losses = STALosses()
    with torch.inference_mode():
        score = losses.sym_geo_loss(pts, n, d, torch.tensor([[1.0]]))
    logger.info("Symmetry score: %.6f", score.item())


if __name__ == "__main__":
    main()