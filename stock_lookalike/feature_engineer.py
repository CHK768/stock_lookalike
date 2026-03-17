"""OHLCV 特征归一化 + 滑动窗口提取"""
import numpy as np
import pandas as pd


def normalize_window(df_window: pd.DataFrame) -> np.ndarray | None:
    """
    对单个窗口进行特征归一化。

    OHLC: price[t] / close[0] - 1  （保留幅度信息，消除绝对价格）
    Volume: log1p(vol[t] / mean(vol_window))

    Args:
        df_window: DataFrame with columns [date, open, high, low, close, volume]
                   长度必须等于 window_size

    Returns:
        np.ndarray shape [window_size, 5] float32，失败返回 None
    """
    arr = df_window[["open", "high", "low", "close", "volume"]].values.astype(np.float64)

    # 过滤含 NaN
    if np.isnan(arr).any():
        return None

    # 过滤停牌（volume=0）超过 5 天
    suspend_days = (arr[:, 4] == 0).sum()
    if suspend_days > 5:
        return None

    close0 = arr[0, 3]
    if close0 <= 0:
        return None

    # OHLC 归一化
    ohlc = arr[:, :4] / close0 - 1.0

    # Volume 归一化
    mean_vol = arr[:, 4].mean()
    if mean_vol <= 0:
        mean_vol = 1.0
    vol = np.log1p(arr[:, 4] / mean_vol)

    result = np.concatenate([ohlc, vol[:, None]], axis=1).astype(np.float32)
    return result


def sliding_windows(
    df: pd.DataFrame,
    window_size: int = 30,
    step: int = 1,
) -> list[np.ndarray]:
    """
    对单只股票的完整历史数据提取所有滑动窗口。

    Returns:
        list of np.ndarray, each shape [window_size, 5]
    """
    windows = []
    n = len(df)
    for start in range(0, n - window_size + 1, step):
        end = start + window_size
        w = normalize_window(df.iloc[start:end])
        if w is not None:
            windows.append(w)
    return windows


def get_latest_window(df: pd.DataFrame, window_size: int = 30) -> np.ndarray | None:
    """
    取最近 window_size 天的窗口。

    Returns:
        np.ndarray shape [window_size, 5] 或 None
    """
    if len(df) < window_size:
        return None
    return normalize_window(df.iloc[-window_size:])


def get_window_at_date(
    df: pd.DataFrame,
    end_date: str,
    window_size: int = 30,
) -> np.ndarray | None:
    """
    取截止到 end_date（含）的最近 window_size 个交易日窗口。

    Args:
        df: 包含 date 列的 DataFrame
        end_date: 截止日期，格式 yyyy-mm-dd 或 yyyymmdd
        window_size: 窗口大小

    Returns:
        np.ndarray shape [window_size, 5] 或 None
    """
    end_ts = pd.Timestamp(end_date)
    sub = df[df["date"] <= end_ts]
    if len(sub) < window_size:
        return None
    return normalize_window(sub.iloc[-window_size:])
