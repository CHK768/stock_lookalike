"""
OHLCV 特征工程 + 滑动窗口提取

特征（共 5 维，窗口内 Z-Score 归一化）：
  [0] close_ret_z    日收益率 Z-Score           主价格方向信号
  [1] body_z         K线实体 Z-Score            (close-open)/prev_close，实体大小和方向
  [2] upper_shadow_z 上影线 Z-Score             (high-close)/prev_close，上方抛压
  [3] lower_shadow_z 下影线 Z-Score             (open-low)/prev_close，下方承接
  [4] vol_x_ret_z    量价协同 Z-Score           vol_z × sign(raw_close_ret)
                       正值=放量上涨/缩量下跌（同向），负值=放量下跌/缩量上涨（背离）

归一化策略：
  - 各维度独立在窗口内做 Z-Score，保留跨维度幅度差异
  - vol_x_ret_z 使用 Z-Score 前的原始涨跌符号，准确反映绝对量价方向
  - 去除跨周期的绝对价位和波动率差异，支持形态跨期对比
"""
import numpy as np
import pandas as pd


def _z(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """对数组沿指定轴做 Z-Score，σ 过小时设为 1.0。"""
    mu = x.mean(axis=axis, keepdims=True)
    sigma = x.std(axis=axis, keepdims=True)
    sigma = np.where(sigma > 1e-8, sigma, 1.0)
    return (x - mu) / sigma


def _make_features(
    open_: np.ndarray,   # [T] or [N, T]
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    axis: int = -1,
) -> tuple[np.ndarray, bool]:
    """
    从 OHLCV 数组计算 5 维归一化特征。

    Args:
        axis: 时间维度（单窗口 axis=-1，向量化批量 axis=1）

    Returns:
        (features, valid)
          features: 与输入同时间维度的 5 列输出
          valid: False 表示窗口无效（无波动或存在 NaN）
    """
    # prev_close：t=0 以自身为基准，t>0 以前一天 close 为基准
    if axis == -1:
        # 单窗口：[T]
        prev_close = np.empty_like(close)
        prev_close[0] = close[0]
        prev_close[1:] = close[:-1]
    else:
        # 批量：[N, T]
        prev_close = np.empty_like(close)
        prev_close[:, 0] = close[:, 0]
        prev_close[:, 1:] = close[:, :-1]

    # ── 原始特征（未归一化） ─────────────────────────────────────────────────
    close_ret    = close / prev_close - 1.0                  # 日收益率
    body         = (close - open_) / prev_close              # K线实体
    upper_shadow = (high  - close) / prev_close              # 上影线（≥0）
    lower_shadow = (open_ - low)   / prev_close              # 下影线（≥0）
    log_vol      = np.log1p(volume)

    # ── 有效性检验 ──────────────────────────────────────────────────────────
    # 收益率标准差过小：无价格波动，窗口无意义
    sigma_c = close_ret.std(axis=axis)
    if np.any(sigma_c < 1e-8):
        return None, False

    # ── 量价协同特征 ────────────────────────────────────────────────────────
    # sign 基于原始收益率（Z-Score 前），准确反映绝对涨跌方向
    raw_sign = np.sign(close_ret)           # +1 上涨日，-1 下跌日，0 平盘

    vol_z_raw = _z(log_vol, axis=axis)
    vol_x_ret = vol_z_raw * raw_sign        # 量价协同方向

    # ── 各维度独立 Z-Score ──────────────────────────────────────────────────
    close_ret_z    = _z(close_ret,    axis=axis)
    body_z         = _z(body,         axis=axis)
    upper_shadow_z = _z(upper_shadow, axis=axis)
    lower_shadow_z = _z(lower_shadow, axis=axis)
    vol_x_ret_z    = _z(vol_x_ret,   axis=axis)

    return (close_ret_z, body_z, upper_shadow_z, lower_shadow_z, vol_x_ret_z), True


def _normalize_arr(arr: np.ndarray) -> np.ndarray | None:
    """
    对单个窗口 numpy 数组进行特征提取和归一化。

    Args:
        arr: float64 array shape [T, 5]，列顺序 open/high/low/close/volume

    Returns:
        float32 array shape [T, 5]（新特征顺序）或 None
    """
    if np.isnan(arr).any():
        return None
    if (arr[:, 4] == 0).sum() > 5:
        return None
    if arr[0, 3] <= 0:
        return None

    feats, valid = _make_features(
        arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4], axis=-1
    )
    if not valid:
        return None

    close_ret_z, body_z, upper_shadow_z, lower_shadow_z, vol_x_ret_z = feats
    out = np.stack(
        [close_ret_z, body_z, upper_shadow_z, lower_shadow_z, vol_x_ret_z], axis=-1
    ).astype(np.float32)
    return out


def _sliding_windows_vectorized(
    mat: np.ndarray,
    window_size: int,
) -> list[np.ndarray]:
    """
    全量向量化滑动窗口特征提取。

    Args:
        mat: float64 array [T, 5]，列顺序 open/high/low/close/volume

    Returns:
        list of float32 [window_size, 5]（新特征），已过滤无效窗口
    """
    from numpy.lib.stride_tricks import sliding_window_view

    T = len(mat)
    if T < window_size:
        return []

    # w3d: [N, window_size, 5]，零拷贝 view
    w3d = sliding_window_view(mat, (window_size, 5))[:, 0, :, :]
    N = w3d.shape[0]

    # ── 基础过滤 ────────────────────────────────────────────────────────────
    has_nan      = np.isnan(w3d).any(axis=(1, 2))
    too_suspended = (w3d[:, :, 4] == 0).sum(axis=1) > 5
    bad_close0   = w3d[:, 0, 3] <= 0
    invalid_mask = has_nan | too_suspended | bad_close0

    # ── prev_close：[N, T] ──────────────────────────────────────────────────
    prev_close = np.empty((N, window_size))
    prev_close[:, 0] = w3d[:, 0, 3]
    prev_close[:, 1:] = w3d[:, :-1, 3]

    # ── 原始特征：[N, T] ────────────────────────────────────────────────────
    open_  = w3d[:, :, 0]
    high   = w3d[:, :, 1]
    low    = w3d[:, :, 2]
    close  = w3d[:, :, 3]
    volume = w3d[:, :, 4]

    close_ret    = close / prev_close - 1.0
    body         = (close - open_) / prev_close
    upper_shadow = (high  - close) / prev_close
    lower_shadow = (open_ - low)   / prev_close
    log_vol      = np.log1p(volume)

    # ── 过滤：价格无波动 ────────────────────────────────────────────────────
    sigma_c = close_ret.std(axis=1)
    invalid_mask |= sigma_c < 1e-8

    # ── 量价协同：sign 基于原始涨跌 ─────────────────────────────────────────
    raw_sign   = np.sign(close_ret)             # [N, T]，+1/-1/0

    # ── 各维度独立 Z-Score：[N, T] ──────────────────────────────────────────
    def z_batch(x):
        mu = x.mean(axis=1, keepdims=True)
        sigma = x.std(axis=1, keepdims=True)
        sigma = np.where(sigma > 1e-8, sigma, 1.0)
        return (x - mu) / sigma

    close_ret_z    = z_batch(close_ret)
    body_z         = z_batch(body)
    upper_shadow_z = z_batch(upper_shadow)
    lower_shadow_z = z_batch(lower_shadow)
    vol_z_raw      = z_batch(log_vol)
    vol_x_ret_z    = z_batch(vol_z_raw * raw_sign)

    # ── 拼合输出：[N, T, 5] ─────────────────────────────────────────────────
    out = np.stack(
        [close_ret_z, body_z, upper_shadow_z, lower_shadow_z, vol_x_ret_z], axis=2
    ).astype(np.float32)

    valid_idx = np.where(~invalid_mask)[0]
    return [out[i] for i in valid_idx]


def normalize_window(df_window: pd.DataFrame) -> np.ndarray | None:
    """
    对单个窗口进行特征提取和归一化。

    Args:
        df_window: DataFrame，含 open/high/low/close/volume 列

    Returns:
        np.ndarray shape [T, 5] float32，失败返回 None
    """
    arr = df_window[["open", "high", "low", "close", "volume"]].values.astype(np.float64)
    return _normalize_arr(arr)


def sliding_windows(
    df: pd.DataFrame,
    window_size: int = 30,
    step: int = 1,
) -> list[np.ndarray]:
    """
    对单只股票完整历史提取所有滑动窗口特征。

    Returns:
        list of np.ndarray, each shape [window_size, 5]
    """
    mat = df[["open", "high", "low", "close", "volume"]].values.astype(np.float64)
    if step == 1:
        return _sliding_windows_vectorized(mat, window_size)
    windows = []
    for start in range(0, len(mat) - window_size + 1, step):
        w = _normalize_arr(mat[start: start + window_size])
        if w is not None:
            windows.append(w)
    return windows


def get_latest_window(df: pd.DataFrame, window_size: int = 30) -> np.ndarray | None:
    """取最近 window_size 天的窗口特征。"""
    if len(df) < window_size:
        return None
    return normalize_window(df.iloc[-window_size:])


def get_window_at_date(
    df: pd.DataFrame,
    end_date: str,
    window_size: int = 30,
) -> np.ndarray | None:
    """
    取截止到 end_date（含）的最近 window_size 个交易日窗口特征。

    Args:
        df: 含 date 列的 DataFrame
        end_date: 截止日期，格式 yyyy-mm-dd 或 yyyymmdd
        window_size: 窗口大小
    """
    end_ts = pd.Timestamp(end_date)
    sub = df[df["date"] <= end_ts]
    if len(sub) < window_size:
        return None
    return normalize_window(sub.iloc[-window_size:])
