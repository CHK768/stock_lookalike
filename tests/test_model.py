"""模型前向通过 + L2 归一化测试"""
import torch
import pytest
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from stock_lookalike.model import StockTransformerEncoder


@pytest.fixture
def model():
    return StockTransformerEncoder(
        input_dim=5, d_model=64, nhead=4, num_layers=3,
        ffn_dim=256, dropout=0.0, embed_dim=128, window_size=30,
    )


def test_forward_shape(model):
    x = torch.randn(4, 30, 5)
    z = model(x)
    assert z.shape == (4, 128)


def test_l2_normalized(model):
    x = torch.randn(4, 30, 5)
    z = model(x)
    norms = z.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5), f"norms={norms}"


def test_batch_size_1(model):
    x = torch.randn(1, 30, 5)
    z = model(x)
    assert z.shape == (1, 128)


def test_no_nan(model):
    x = torch.randn(8, 30, 5)
    z = model(x)
    assert not torch.isnan(z).any()
