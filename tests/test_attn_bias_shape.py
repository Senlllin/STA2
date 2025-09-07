import torch

from code.models.sta.sta_transformer import STAAttentionBias, SymAwareDecoder


def test_bias_shape_and_sparsity() -> None:
    """Bias should be (B,M,M) with exactly ``k`` positives per row."""

    batch, tokens = 2, 5
    anchors = torch.randn(batch, tokens, 3)
    n = torch.tensor([[0.0, 1.0, 0.0]] * batch)
    d = torch.zeros(batch, 1)
    bias_module = STAAttentionBias(beta=1.5, k_mirror=2)
    bias = bias_module(anchors, n, d)
    assert bias.shape == (batch, tokens, tokens)
    assert torch.all((bias > 0).sum(-1) == 2)


def test_decoder_forward() -> None:
    """Decoder should preserve input shape."""

    batch, tokens, dim = 2, 5, 16
    x = torch.randn(batch, tokens, dim)
    anchors = torch.randn(batch, tokens, 3)
    n = torch.tensor([[0.0, 1.0, 0.0]] * batch)
    d = torch.zeros(batch, 1)
    decoder = SymAwareDecoder(dim, num_heads=4, use_sta_bias=True)
    out = decoder(x, anchors, n, d)
    assert out.shape == x.shape