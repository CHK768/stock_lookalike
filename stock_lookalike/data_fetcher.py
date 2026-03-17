"""AKShare 数据获取封装 + Parquet 缓存"""
import time
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import akshare as ak
import pandas as pd

logger = logging.getLogger(__name__)


def get_stock_list(save_path: str = "data/stock_list.csv") -> pd.DataFrame:
    """获取A股全量股票列表并缓存到CSV"""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = ak.stock_info_a_code_name()
    df.columns = ["code", "name"]
    df.to_csv(path, index=False)
    logger.info(f"股票列表已保存: {len(df)} 只，路径: {path}")
    return df


def _market_prefix(code: str) -> str:
    """根据股票代码返回新浪接口所需的市场前缀（sz/sh/bj）"""
    if code.startswith("6") or code.startswith("9"):
        return "sh"
    if code.startswith("4") or code.startswith("8"):
        return "bj"
    return "sz"


def fetch_single_stock(
    code: str,
    start_date: str,
    end_date: str,
    raw_dir: str = "data/raw",
    force_refresh: bool = False,
) -> pd.DataFrame | None:
    """
    获取单只股票前复权日线数据，支持 Parquet 缓存增量更新。
    使用新浪财经接口（stock_zh_a_daily），避免东方财富接口访问限制。

    Returns:
        DataFrame with columns: date, open, high, low, close, volume
        失败返回 None
    """
    path = Path(raw_dir) / f"{code}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)

    # 读取缓存
    cached = None
    fetch_start = start_date
    fetch_end = end_date

    if path.exists() and not force_refresh:
        try:
            cached = pd.read_parquet(path)
            cached["date"] = pd.to_datetime(cached["date"])
            cached_start = cached["date"].min()
            cached_end = cached["date"].max()

            need_forward = pd.Timestamp(cached_end) < pd.Timestamp(end_date)
            need_backward = pd.Timestamp(cached_start) > pd.Timestamp(start_date)

            # 缓存已完整覆盖所需范围，直接返回
            if not need_forward and not need_backward:
                mask = (cached["date"] >= start_date) & (cached["date"] <= end_date)
                return cached[mask].reset_index(drop=True)

            # 只需向前补充（更新近期数据）
            if need_forward and not need_backward:
                fetch_start = (cached_end + timedelta(days=1)).strftime("%Y%m%d")
                fetch_end = end_date
            # 只需向后补充（回溯历史数据）
            elif need_backward and not need_forward:
                fetch_start = start_date
                fetch_end = (cached_start - timedelta(days=1)).strftime("%Y%m%d")
            # 两端都需要补充（缓存中间有，但两头都缺）
            else:
                fetch_start = start_date
                fetch_end = end_date
        except Exception:
            cached = None

    # 拉取新数据（使用新浪财经接口，需要加市场前缀）
    symbol = _market_prefix(code) + code
    try:
        df_new = ak.stock_zh_a_daily(
            symbol=symbol,
            start_date=fetch_start.replace("-", ""),
            end_date=fetch_end.replace("-", ""),
            adjust="qfq",
        )
    except Exception as e:
        if cached is not None:
            logger.debug(f"拉取 {code} 失败（使用缓存）: {e}")
            mask = (cached["date"] >= start_date) & (cached["date"] <= end_date)
            return cached[mask].reset_index(drop=True)
        logger.warning(f"拉取 {code} 失败（无缓存）: {e}")
        return None

    if df_new is None or df_new.empty:
        if cached is not None:
            mask = (cached["date"] >= start_date) & (cached["date"] <= end_date)
            return cached[mask].reset_index(drop=True)
        return None

    # stock_zh_a_daily 已返回英文列名，只保留所需列
    df_new = df_new[["date", "open", "high", "low", "close", "volume"]].copy()
    df_new["date"] = pd.to_datetime(df_new["date"])

    # 合并缓存与新数据
    if cached is not None:
        df_all = pd.concat([cached, df_new], ignore_index=True)
        df_all = df_all.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    else:
        df_all = df_new.sort_values("date").reset_index(drop=True)

    # 写回缓存
    df_all.to_parquet(path, index=False)

    mask = (df_all["date"] >= pd.Timestamp(start_date)) & (df_all["date"] <= pd.Timestamp(end_date))
    return df_all[mask].reset_index(drop=True)


def fetch_all_stocks(
    codes: list[str],
    start_date: str,
    end_date: str,
    raw_dir: str = "data/raw",
    max_workers: int = 2,
    rate_limit_per_sec: float = 2.0,
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    并发拉取多只股票数据，全局令牌桶限速（防止触发数据源限流）。

    Returns:
        {code: DataFrame} 字典，失败的股票不包含在结果中
    """
    import threading
    results = {}
    interval = 1.0 / rate_limit_per_sec
    lock = threading.Lock()
    last_call = [0.0]  # 用列表使闭包可修改

    def _is_cached(code: str) -> bool:
        """快速判断缓存是否完整覆盖所需时间范围（两端都检查）"""
        if force_refresh:
            return False
        path = Path(raw_dir) / f"{code}.parquet"
        if not path.exists():
            return False
        try:
            cached = pd.read_parquet(path, columns=["date"])
            dates = pd.to_datetime(cached["date"])
            return (dates.max() >= pd.Timestamp(end_date) and
                    dates.min() <= pd.Timestamp(start_date))
        except Exception:
            return False

    def _fetch(code: str):
        # 仅对实际需要网络请求的股票限速
        if not _is_cached(code):
            with lock:
                now = time.time()
                wait = interval - (now - last_call[0])
                if wait > 0:
                    time.sleep(wait)
                last_call[0] = time.time()
        return code, fetch_single_stock(code, start_date, end_date, raw_dir, force_refresh)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch, code): code for code in codes}
        for future in as_completed(futures):
            code, df = future.result()
            if df is not None and not df.empty:
                results[code] = df
            else:
                logger.debug(f"跳过 {code}（数据为空）")

    logger.info(f"成功获取 {len(results)}/{len(codes)} 只股票数据")
    return results
