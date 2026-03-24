"""FastAPI 服务：w20/w30 模型相似股票检索（局域网访问）"""
import copy
import logging
from enum import IntEnum
from pathlib import Path
from contextlib import asynccontextmanager

import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse


class WindowSize(IntEnum):
    w20 = 20
    w30 = 30

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("server")

# ─── 全局状态 ──────────────────────────────────────────────────────────────────

BASE_CONFIG: dict = {}
stock_names: dict = {}
raw_dir: Path = Path("data/raw")

# 按窗口大小维护独立的 current 搜索器
_current_searchers: dict[int, object] = {}

SUPPORTED_WINDOWS = [20, 30]

# 各窗口模型的展示名称和架构说明
MODEL_INFO = {
    20: {"name": "w20-zscore", "arch": "dual_channel", "desc": "双通道 Transformer + Z-Score 跨周期特征（市场中性）"},
    30: {"name": "w30",        "arch": "single",       "desc": "单通道 Transformer（OHLCV 统一输入）"},
}


def load_config(path="config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_config(base: dict, window_size: int) -> dict:
    """基于 base config 生成指定窗口大小的配置副本。"""
    cfg = copy.deepcopy(base)
    cfg.setdefault("data", {})["window_size"] = window_size
    if window_size == 20:
        cfg["data"]["index_dir"] = "data/index/w20-zscore"
        cfg.setdefault("train", {})["checkpoint_dir"] = "checkpoints/w20-zscore"
    elif window_size != 30:
        cfg["data"]["index_dir"] = f"data/index/w{window_size}"
        cfg.setdefault("train", {})["checkpoint_dir"] = f"checkpoints/w{window_size}"
    return cfg


def get_searcher(window_size: int = 30, index_dir_override=None):
    """获取搜索器：current 模式复用单例，其余每次新建。"""
    from stock_lookalike.searcher import StockSearcher

    cfg = make_config(BASE_CONFIG, window_size)

    if index_dir_override is None:
        if window_size not in _current_searchers:
            s = StockSearcher(cfg, device="cpu")
            s.load()
            _current_searchers[window_size] = s
        return _current_searchers[window_size]

    # sync / history：临时搜索器，加载指定索引
    s = StockSearcher(cfg, device="cpu", index_dir_override=index_dir_override)
    s.load()
    return s


@asynccontextmanager
async def lifespan(app: FastAPI):
    global BASE_CONFIG, stock_names, raw_dir

    BASE_CONFIG = load_config()
    dc = BASE_CONFIG.get("data", {})
    raw_dir = Path(dc.get("raw_dir", "data/raw"))

    stock_list_file = dc.get("stock_list_file", "data/stock_list.csv")
    if Path(stock_list_file).exists():
        df_list = pd.read_csv(stock_list_file, dtype=str)
        stock_names = dict(zip(df_list["code"], df_list["name"]))
        logger.info(f"加载股票列表: {len(stock_names)} 只")

    # 预加载所有支持窗口的 current 搜索器
    for w in SUPPORTED_WINDOWS:
        index_dir = Path(make_config(BASE_CONFIG, w)["data"]["index_dir"])
        if (index_dir / "embeddings.npy").exists():
            try:
                get_searcher(w)
                logger.info(f"w{w} 模型和索引加载完成")
            except Exception as e:
                logger.warning(f"w{w} 索引加载失败（跳过）: {e}")
        else:
            logger.warning(f"w{w} 索引不存在，跳过预加载")

    logger.info("服务就绪")

    yield


# ─── 应用 ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Stock Lookalike API",
    description=(
        "A股相似走势检索服务\n\n"
        "## 可用模型\n"
        "| 参数值 | 名称 | 架构 | 说明 |\n"
        "|--------|------|------|------|\n"
        "| `window=20` | **w20-zscore** | dual_channel | 双通道 Transformer + Z-Score 市场中性特征，支持跨周期形态对比 |\n"
        "| `window=30` | **w30** | single | 单通道 Transformer，OHLCV 统一输入 |\n"
    ),
    version="2.0.0",
)


@app.get("/health")
def health():
    loaded = [
        {"window": w, "name": MODEL_INFO[w]["name"], "arch": MODEL_INFO[w]["arch"], "desc": MODEL_INFO[w]["desc"]}
        for w in SUPPORTED_WINDOWS if w in _current_searchers
    ]
    return {"status": "ok", "models_loaded": loaded}


@app.get("/search")
def search(
    code: str = Query(..., description="股票代码，如 000001"),
    top_k: int = Query(10, ge=1, le=50),
    mode: str = Query("current", description="current | sync | history"),
    end_date: str = Query(None, description="窗口截止日期 yyyy-mm-dd"),
    window: WindowSize = Query(WindowSize.w30, description="模型窗口：20=w20-dct（双通道Transformer） / 30=w30（单通道Transformer，默认）"),
):
    """
    查询与目标股票最相似的股票。

    - **code**: 股票代码，如 000001
    - **top_k**: 返回数量（1~50）
    - **mode**: `current`=当前最新索引 | `sync`=同期历史索引 | `history`=全历史索引
    - **end_date**: 窗口截止日期（sync/history 模式下有效）
    - **window**: 模型选择
      - `20` → **w20-zscore**：双通道 Transformer + Z-Score 市场中性特征，支持跨周期形态对比
      - `30` → **w30**：单通道 Transformer（OHLCV 统一输入）
    """

    p = raw_dir / f"{code}.parquet"
    if not p.exists():
        raise HTTPException(404, detail=f"未找到 {code} 的数据，请先运行 download")

    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])

    cfg = make_config(BASE_CONFIG, window)
    base_index_dir = Path(cfg["data"]["index_dir"])

    # 确定索引目录
    index_dir_override = None
    if mode == "history":
        full_history_dir = base_index_dir / "full_history"
        if not full_history_dir.exists():
            raise HTTPException(404, detail="全历史索引不存在，请先运行 build-full-index")
        index_dir_override = str(full_history_dir)
    elif mode == "sync" and end_date:
        historical_dir = base_index_dir / end_date
        if not historical_dir.exists():
            raise HTTPException(404, detail=f"未找到 {end_date} 的历史索引，请先运行 build-index --end-date {end_date}")
        index_dir_override = str(historical_dir)

    searcher = get_searcher(window, index_dir_override)
    results = searcher.search_by_code(code, df, top_k=top_k, end_date=end_date)

    # 计算查询窗口日期
    sub = df[df["date"] <= pd.Timestamp(end_date)] if end_date else df
    if len(sub) >= window:
        window_start = sub["date"].iloc[-window].strftime("%Y-%m-%d")
        window_end_str = sub["date"].iloc[-1].strftime("%Y-%m-%d")
    else:
        window_start = window_end_str = None

    info = MODEL_INFO.get(int(window), {})
    return JSONResponse({
        "query": {
            "code": code,
            "name": stock_names.get(code, code),
            "window_start": window_start,
            "window_end": window_end_str,
            "mode": mode,
            "window_size": window,
            "model_name": info.get("name", f"w{window}"),
            "model_arch": info.get("arch", "unknown"),
        },
        "results": [
            {
                "rank": r.rank,
                "code": r.code,
                "name": r.name,
                "similarity": round(r.similarity, 4),
                "window_end": r.window_end,
            }
            for r in results
        ],
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
