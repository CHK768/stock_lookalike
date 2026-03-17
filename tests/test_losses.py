"""NT-Xent Loss 数值测试"""
import torch
import pytest
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from stock_lookalike.losses import NTXentLoss


def test_loss_positive():
    """loss 应为正数"""
    loss_fn = NTXentLoss(temperature=0.07)
    z1 = torch.randn(8, 128)
    z1 = z1 / z1.norm(dim=-1, keepdim=True)
    z2 = torch.randn(8, 128)
    z2 = z2 / z2.norm(dim=-1, keepdim=True)
    loss = loss_fn(z1, z2)
    assert loss.item() > 0


def test_loss_decreases_for_similar_pairs():
    """完全相同的正样本对应有较低的 loss"""
    loss_fn = NTXentLoss(temperature=0.07)
    z = torch.randn(16, 128)
    z = z / z.norm(dim=-1, keepdim=True)

    # 完全相同的正样本对（最优情况）
    loss_perfect = loss_fn(z, z.clone())

    # 随机负样本（随机）
    z2_rand = torch.randn(16, 128)
    z2_rand = z2_rand / z2_rand.norm(dim=-1, keepdim=True)
    loss_rand = loss_fn(z, z2_rand)

    assert loss_perfect < loss_rand


def test_loss_shape():
    """loss 应为标量"""
    loss_fn = NTXentLoss()
    z1 = torch.randn(4, 64)
    z1 = z1 / z1.norm(dim=-1, keepdim=True)
    z2 = torch.randn(4, 64)
    z2 = z2 / z2.norm(dim=-1, keepdim=True)
    loss = loss_fn(z1, z2)
    assert loss.shape == torch.Size([])
