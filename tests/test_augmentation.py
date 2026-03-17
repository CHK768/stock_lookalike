"""增强策略：形状不变 + 不改变幅度量级"""
import numpy as np
import pytest
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from stock_lookalike.augmentation import StockAugmentor


def _make_window(T: int = 30) -> np.ndarray:
    np.random.seed(0)
    return np.random.randn(T, 5).astype(np.float32)


def test_output_shape():
    aug = StockAugmentor()
    w = _make_window()
    v1, v2 = aug(w)
    assert v1.shape == (30, 5)
    assert v2.shape == (30, 5)


def test_two_views_differ():
    """两个视图应不完全相同（概率极小才会相同）"""
    aug = StockAugmentor()
    w = _make_window()
    v1, v2 = aug(w)
    assert not np.allclose(v1, v2)


def test_amplitude_not_scaled():
    """增强后均值量级应与原始接近（无幅度缩放）"""
    aug = StockAugmentor(noise_std=0.005, max_shift=0, mask_prob=0.0)
    w = _make_window()
    v1, _ = aug(w)
    # OHLC 均值差异应 < 0.05（仅高斯噪声 std=0.005）
    diff = abs(v1[:, :4].mean() - w[:, :4].mean())
    assert diff < 0.05, f"幅度差异过大: {diff}"


def test_dtype_preserved():
    aug = StockAugmentor()
    w = _make_window()
    v1, v2 = aug(w)
    assert v1.dtype == np.float32
    assert v2.dtype == np.float32
