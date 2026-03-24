"""相似度检索接口"""
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .feature_engineer import get_latest_window, get_window_at_date
from .train import load_model

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    rank: int
    code: str
    name: str
    similarity: float
    window_end: str


class StockSearcher:
    """
    加载预构建索引，对给定目标股票的当前窗口进行相似度检索。

    流程：
        1. 加载 embeddings.npy 和 metadata.json
        2. 对目标窗口推理得到 query embedding
        3. 暴力余弦相似度：similarities = embeddings @ query（已 L2 归一化）
        4. top-k 排序返回
    """

    def __init__(self, config: dict, device: str = "cpu", index_dir_override: str | None = None):
        self.config = config
        self.device = device
        self.index_dir_override = index_dir_override
        self.model = None
        self.embeddings = None   # [N, 128]
        self.metadata = None

    def load(self) -> None:
        """加载模型和索引"""
        if self.index_dir_override:
            index_dir = Path(self.index_dir_override)
        else:
            index_dir = Path(self.config.get("data", {}).get("index_dir", "data/index"))
        emb_path = index_dir / "embeddings.npy"
        meta_path = index_dir / "metadata.json"

        if not emb_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"索引文件不存在，请先运行 build-index。目录: {index_dir}")

        self.embeddings = np.load(emb_path)          # [N, 128]
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.model = load_model(self.config, device=self.device)
        if self.metadata.get("type") == "full_history":
            logger.info(f"已加载全历史索引: {self.metadata['count']} 个窗口")
        else:
            logger.info(f"已加载索引: {self.metadata['count']} 只股票")

    def search_by_window(
        self,
        window: np.ndarray,
        top_k: int = 10,
        exclude_codes: list[str] | None = None,
        query_end_date: str | None = None,
    ) -> list[SearchResult]:
        """
        Args:
            window: np.ndarray [T, 5]，已归一化
            top_k: 返回最相似数量
            exclude_codes: 排除的股票代码列表（如查询自身）
            query_end_date: 查询窗口截止日期（全历史模式下用于过滤同股票近期窗口）

        Returns:
            list[SearchResult]
        """
        if self.embeddings is None:
            self.load()

        x = torch.from_numpy(window).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            query_emb = self.model(x).squeeze(0).cpu().numpy()  # [128]

        # 余弦相似度（已 L2 归一化，直接点积）
        similarities = self.embeddings @ query_emb  # [N]

        codes = self.metadata["codes"]
        names = self.metadata["names"]
        # 全历史索引：每条记录有独立 window_end；快照索引：共用一个 end_date
        per_window_ends = self.metadata.get("window_ends")
        global_end_date = self.metadata.get("end_date", "")

        exclude_set = set(exclude_codes or [])
        window_size = self.config.get("data", {}).get("window_size", 30)

        # 全历史模式：解析查询截止日期，用于排除同股票近期窗口
        query_ts = None
        if per_window_ends and query_end_date:
            try:
                query_ts = pd.Timestamp(query_end_date)
            except Exception:
                pass

        # 按相似度降序排列
        sorted_idx = np.argsort(-similarities)

        results = []
        rank = 1
        for idx in sorted_idx:
            code = codes[idx]
            window_end = per_window_ends[idx] if per_window_ends else global_end_date

            # 排除指定代码（如查询自身）
            if code in exclude_set:
                # 全历史模式：同一股票只排除时间上相近的窗口（< window_size 天），保留远期历史
                if per_window_ends and query_ts is not None:
                    try:
                        we_ts = pd.Timestamp(window_end)
                        if abs((we_ts - query_ts).days) < window_size:
                            continue
                        # 时间足够远的同股票历史窗口保留
                    except Exception:
                        continue
                else:
                    continue

            results.append(SearchResult(
                rank=rank,
                code=code,
                name=names[idx],
                similarity=float(similarities[idx]),
                window_end=window_end,
            ))
            rank += 1
            if rank > top_k:
                break

        return results

    def search_by_code(
        self,
        code: str,
        df,
        top_k: int = 10,
        end_date: str | None = None,
    ) -> list[SearchResult]:
        """
        通过股票代码和 DataFrame 查询最相似股票。

        Args:
            code: 查询股票代码（用于自排除）
            df: 该股票的 DataFrame（data_fetcher 返回的格式）
            top_k: 返回数量
            end_date: 窗口截止日期（yyyy-mm-dd），None 则取最新

        Returns:
            list[SearchResult]
        """
        window_size = self.config.get("data", {}).get("window_size", 30)
        if end_date:
            window = get_window_at_date(df, end_date=end_date, window_size=window_size)
            # 推算实际窗口截止日（可能比 end_date 早）
            sub = df[df["date"] <= pd.Timestamp(end_date)]
            actual_end = sub["date"].iloc[-1].strftime("%Y-%m-%d") if len(sub) >= window_size else end_date
        else:
            window = get_latest_window(df, window_size=window_size)
            actual_end = df["date"].iloc[-1].strftime("%Y-%m-%d") if len(df) > 0 else None
        if window is None:
            raise ValueError(f"股票 {code} 在 {end_date or '最新'} 前数据不足以提取 {window_size} 天窗口")
        return self.search_by_window(
            window, top_k=top_k, exclude_codes=[code], query_end_date=actual_end
        )
