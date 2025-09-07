import torch

from code.models.sta.sta_transformer import SymAwareDecoder


def test_decoder_parity_without_bias() -> None:
    """When bias is disabled, decoder matches bare MultiheadAttention."""
    torch.manual_seed(0)
    batch, tokens, dim = 1, 4, 8
    x = torch.randn(batch, tokens, dim)
    anchors = torch.randn(batch, tokens, 3)
    n = torch.zeros(batch, 3)
    d = torch.zeros(batch, 1)

    decoder = SymAwareDecoder(dim, num_heads=2, use_sta_bias=False)
    baseline, _ = decoder.mha(x, x, x)
    out = decoder(x, anchors, n, d)
    assert torch.allclose(out, baseline)