"""批量推理建索引：为所有股票当前窗口生成 embedding"""
import json
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .feature_engineer import get_window_at_date, normalize_window
from .train import load_model

logger = logging.getLogger(__name__)


def build_index(
    stock_data: dict,
    stock_names: dict,
    config: dict,
    end_date: str,
    device: str = "cpu",
    index_dir_override: str | None = None,
) -> None:
    """
    为所有股票生成指定日期窗口的 embedding，写入 index 目录。

    Args:
        stock_data: {code: DataFrame}
        stock_names: {code: name}
        config: 完整 config
        end_date: 窗口截止日期（yyyy-mm-dd 或 yyyymmdd）
        device: 推理设备
        index_dir_override: 覆盖 config 中的索引目录（历史索引场景）
    """
    window_size = config.get("data", {}).get("window_size", 30)
    if index_dir_override:
        index_dir = Path(index_dir_override)
    else:
        index_dir = Path(config.get("data", {}).get("index_dir", "data/index"))
    index_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(config, device=device)
    model.eval()

    codes = []
    embeddings = []

    with torch.no_grad():
        for code, df in tqdm(stock_data.items(), desc="构建索引"):
            w = get_window_at_date(df, end_date=end_date, window_size=window_size)
            if w is None:
                logger.debug(f"跳过 {code}（窗口提取失败）")
                continue
            x = torch.from_numpy(w).float().unsqueeze(0).to(device)  # [1, T, 5]
            emb = model(x).squeeze(0).cpu().numpy()                   # [128]
            codes.append(code)
            embeddings.append(emb)

    if not embeddings:
        raise RuntimeError("所有股票均未能提取到有效窗口，无法建索引")

    emb_array = np.stack(embeddings, axis=0).astype(np.float32)  # [N, 128]
    np.save(index_dir / "embeddings.npy", emb_array)

    metadata = {
        "codes": codes,
        "names": [stock_names.get(c, c) for c in codes],
        "end_date": end_date,
        "count": len(codes),
    }
    with open(index_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logger.info(f"索引已建立: {len(codes)} 只股票，embeddings shape={emb_array.shape}")


def build_full_history_index(
    stock_data: dict,
    stock_names: dict,
    config: dict,
    device: str = "cpu",
    step: int = 1,
    index_dir_override: str | None = None,
) -> None:
    """
    构建全历史索引：每只股票的所有滑动窗口都生成 embedding。
    搜索结果会返回（股票代码 + 具体时间窗口），用于"历史上谁和目标最像"。

    Args:
        stock_data: {code: DataFrame}
        stock_names: {code: name}
        config: 完整 config
        device: 推理设备
        step: 滑动窗口步长（1=最精细，5=快5倍但略粗）
        index_dir_override: 覆盖索引目录
    """
    window_size = config.get("data", {}).get("window_size", 30)
    if index_dir_override:
        index_dir = Path(index_dir_override)
    else:
        index_dir = Path(config.get("data", {}).get("index_dir", "data/index")) / "full_history"
    index_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(config, device=device)
    model.eval()

    BATCH = 512
    batch_windows: list = []
    batch_meta: list = []   # (code, name, window_end_str)
    all_codes: list = []
    all_names: list = []
    all_window_ends: list = []
    all_embeddings: list = []

    def flush():
        if not batch_windows:
            return
        x = torch.from_numpy(np.stack(batch_windows)).float().to(device)
        with torch.no_grad():
            embs = model(x).cpu().numpy()
        for i, (c, nm, we) in enumerate(batch_meta):
            all_codes.append(c)
            all_names.append(nm)
            all_window_ends.append(we)
            all_embeddings.append(embs[i])
        batch_windows.clear()
        batch_meta.clear()

    for code, df in tqdm(stock_data.items(), desc="构建全历史索引"):
        n = len(df)
        for start in range(0, n - window_size + 1, step):
            end_idx = start + window_size
            w = normalize_window(df.iloc[start:end_idx])
            if w is None:
                continue
            ts = df.iloc[end_idx - 1]["date"]
            we = ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)[:10]
            batch_windows.append(w)
            batch_meta.append((code, stock_names.get(code, code), we))
            if len(batch_windows) >= BATCH:
                flush()

    flush()

    if not all_embeddings:
        raise RuntimeError("所有股票均未能提取到有效窗口")

    emb_array = np.stack(all_embeddings, axis=0).astype(np.float32)
    np.save(index_dir / "embeddings.npy", emb_array)

    metadata = {
        "codes": all_codes,
        "names": all_names,
        "window_ends": all_window_ends,
        "count": len(all_codes),
        "type": "full_history",
    }
    with open(index_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logger.info(f"全历史索引已建立: {len(all_codes)} 个窗口，来自 {len(stock_data)} 只股票")
