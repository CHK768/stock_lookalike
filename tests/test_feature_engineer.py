"""特征归一化正确性测试"""
import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from stock_lookalike.feature_engineer import normalize_window, sliding_windows, get_latest_window


def _make_df(n: int = 30) -> pd.DataFrame:
    np.random.seed(42)
    close = 10.0 * np.cumprod(1 + np.random.randn(n) * 0.01)
    data = {
        "date": pd.date_range("2024-01-01", periods=n),
        "open": close * (1 + np.random.randn(n) * 0.005),
        "high": close * (1 + abs(np.random.randn(n)) * 0.01),
        "low": close * (1 - abs(np.random.randn(n)) * 0.01),
        "close": close,
        "volume": np.random.randint(1_000_000, 10_000_000, n).astype(float),
    }
    return pd.DataFrame(data)


def test_close0_is_zero():
    """第0天收盘价归一化后应为 0"""
    df = _make_df(30)
    w = normalize_window(df)
    assert w is not None
    # close 是第3列（ohlc 顺序：open=0,high=1,low=2,close=3）
    assert abs(w[0, 3]) < 1e-6, f"close[0]={w[0, 3]} 不为0"


def test_output_shape():
    df = _make_df(30)
    w = normalize_window(df)
    assert w is not None
    assert w.shape == (30, 5)
    assert w.dtype == np.float32


def test_nan_returns_none():
    df = _make_df(30)
    df.loc[5, "close"] = np.nan
    w = normalize_window(df)
    assert w is None


def test_too_many_suspend_returns_none():
    df = _make_df(30)
    df.loc[0:5, "volume"] = 0.0  # 6天停牌 > 5
    w = normalize_window(df)
    assert w is None


def test_sliding_windows_count():
    df = _make_df(50)
    windows = sliding_windows(df, window_size=30, step=1)
    # 50-30+1 = 21 个窗口
    assert len(windows) == 21


def test_get_latest_window():
    df = _make_df(60)
    w = get_latest_window(df, window_size=30)
    assert w is not None
    assert w.shape == (30, 5)


def test_get_latest_window_insufficient_data():
    df = _make_df(20)
    w = get_latest_window(df, window_size=30)
    assert w is None
