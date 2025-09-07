import torch

from code.models.sta.sta_math import normalize_plane, reflect_points


def test_reflect_points() -> None:
    """Reflection should flip coordinates across the plane."""

    n = torch.tensor([[0.0, 1.0, 0.0]])
    d = torch.tensor([[0.0]])
    pts = torch.tensor([[[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]])
    refl = reflect_points(pts, n, d)
    expected = torch.tensor([[[1.0, -2.0, 3.0], [-1.0, 2.0, -3.0]]])
    assert torch.allclose(refl, expected, atol=1e-5)


def test_normalize_plane() -> None:
    """Plane normals should be unit length after normalization."""

    n = torch.tensor([[0.0, 2.0, 0.0]])
    d = torch.tensor([[2.0]])
    n_norm, d_norm = normalize_plane(n, d)
    assert torch.allclose(torch.linalg.norm(n_norm, dim=-1), torch.ones(1), atol=1e-6)
    assert torch.allclose(d_norm, torch.tensor([[1.0]]))