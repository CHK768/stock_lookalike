"""CLI 入口：download / train / build-index / search / pipeline"""
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import click
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cli")


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@click.group()
@click.option("--config", default="config.yaml", show_default=True, help="配置文件路径")
@click.option("--window-size", default=None, type=int,
              help="覆盖 window_size，自动隔离 checkpoint 和索引目录（如 --window-size 20）")
@click.pass_context
def cli(ctx, config, window_size):
    ctx.ensure_object(dict)
    cfg = load_config(config)
    if window_size:
        cfg["data"]["window_size"] = window_size
        cfg["train"]["checkpoint_dir"] = f"checkpoints/w{window_size}"
        cfg["data"]["index_dir"] = f"data/index/w{window_size}"
    ctx.obj["config"] = cfg


# ─── download ────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--lookback-days", default=None, type=int, help="回看天数（覆盖 config）")
@click.option("--codes", default=None, help="逗号分隔的股票代码，默认全量")
@click.option("--force-refresh", is_flag=True, help="强制重新下载（忽略缓存）")
@click.option("--max-workers", default=8, show_default=True)
@click.option("--rate-limit", default=3.0, show_default=True, type=float, help="每秒最大请求数")
@click.pass_context
def download(ctx, lookback_days, codes, force_refresh, max_workers, rate_limit):
    """下载 A 股 OHLCV 数据到本地 Parquet 缓存"""
    config = ctx.obj["config"]
    dc = config.get("data", {})

    lb = lookback_days or dc.get("lookback_days", 120)
    end_date = datetime.today().strftime("%Y%m%d")
    start_date = (datetime.today() - timedelta(days=lb)).strftime("%Y%m%d")

    from stock_lookalike.data_fetcher import get_stock_list, fetch_all_stocks
    import pandas as pd

    stock_list_file = dc.get("stock_list_file", "data/stock_list.csv")

    if codes:
        # 直接使用指定代码，跳过股票列表获取
        code_list = [c.strip() for c in codes.split(",")]
    else:
        # 先尝试获取最新列表，失败则使用本地缓存
        click.echo("获取股票列表...")
        try:
            df_list = get_stock_list(stock_list_file)
        except Exception as e:
            logger.warning(f"在线获取股票列表失败: {e}")
            if Path(stock_list_file).exists():
                df_list = pd.read_csv(stock_list_file, dtype=str)
                click.echo(f"使用缓存列表（{len(df_list)} 只）")
            else:
                click.echo("无股票列表缓存，请先指定 --codes 或联网获取", err=True)
                sys.exit(1)
        code_list = df_list["code"].tolist()

    click.echo(f"下载 {len(code_list)} 只股票，{start_date} ~ {end_date}，lookback={lb}天")

    fetch_all_stocks(
        codes=code_list,
        start_date=start_date,
        end_date=end_date,
        raw_dir=dc.get("raw_dir", "data/raw"),
        max_workers=max_workers,
        rate_limit_per_sec=rate_limit,
        force_refresh=force_refresh,
    )
    click.echo("下载完成")


# ─── train ────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--epochs", default=None, type=int, help="训练轮数（覆盖 config）")
@click.option("--max-stocks", default=None, type=int, help="最多使用几只股票（调试用）")
@click.option("--device", default=None, help="cuda / mps / cpu")
@click.pass_context
def train(ctx, epochs, max_stocks, device):
    """用对比学习训练 Transformer 编码器"""
    config = ctx.obj["config"]
    if epochs:
        config.setdefault("train", {})["epochs"] = epochs

    dc = config.get("data", {})
    raw_dir = dc.get("raw_dir", "data/raw")
    window_size = dc.get("window_size", 30)

    # 收集所有已缓存数据
    import pandas as pd
    raw_path = Path(raw_dir)
    parquet_files = list(raw_path.glob("*.parquet"))
    if max_stocks:
        parquet_files = parquet_files[:max_stocks]

    click.echo(f"加载 {len(parquet_files)} 只股票数据...")
    stock_data = {}
    for p in parquet_files:
        try:
            df = pd.read_parquet(p)
            df["date"] = pd.to_datetime(df["date"])
            stock_data[p.stem] = df
        except Exception as e:
            logger.warning(f"读取 {p} 失败: {e}")

    if not stock_data:
        click.echo("无可用数据，请先运行 download", err=True)
        sys.exit(1)

    from stock_lookalike.dataset import build_dataset_from_stocks
    from stock_lookalike.train import train as do_train

    click.echo("构建训练集...")
    dataset = build_dataset_from_stocks(stock_data, window_size=window_size)
    click.echo(f"训练样本数: {len(dataset)}")

    model, history = do_train(dataset, config, device=device)
    click.echo(f"训练完成，最终 loss={history[-1]['loss']:.4f}")


# ─── build-index ──────────────────────────────────────────────────────────────

@cli.command("build-index")
@click.option("--device", default="cpu", show_default=True)
@click.option("--end-date", default=None, help="历史截止日期 yyyy-mm-dd，默认今天")
@click.pass_context
def build_index(ctx, device, end_date):
    """为所有股票生成 embedding 索引（支持历史截止日期）"""
    config = ctx.obj["config"]
    dc = config.get("data", {})
    raw_dir = dc.get("raw_dir", "data/raw")
    stock_list_file = dc.get("stock_list_file", "data/stock_list.csv")
    base_index_dir = Path(dc.get("index_dir", "data/index"))

    import pandas as pd

    # 加载股票名称映射
    stock_names = {}
    if Path(stock_list_file).exists():
        df_list = pd.read_csv(stock_list_file, dtype=str)
        stock_names = dict(zip(df_list["code"], df_list["name"]))

    # 加载所有缓存数据
    raw_path = Path(raw_dir)
    stock_data = {}
    for p in raw_path.glob("*.parquet"):
        try:
            df = pd.read_parquet(p)
            df["date"] = pd.to_datetime(df["date"])
            stock_data[p.stem] = df
        except Exception as e:
            logger.warning(f"读取 {p} 失败: {e}")

    if not stock_data:
        click.echo("无可用数据，请先运行 download", err=True)
        sys.exit(1)

    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")
        index_dir_override = None
    else:
        index_dir_override = str(base_index_dir / end_date)

    from stock_lookalike.build_index import build_index as do_build
    do_build(stock_data, stock_names, config, end_date=end_date, device=device,
             index_dir_override=index_dir_override)
    click.echo("索引构建完成")


# ─── build-full-index ─────────────────────────────────────────────────────────

@cli.command("build-full-index")
@click.option("--device", default="cpu", show_default=True)
@click.option("--step", default=1, show_default=True, type=int, help="滑动窗口步长（1=最精细，5=快5倍）")
@click.pass_context
def build_full_index_cmd(ctx, device, step):
    """构建全历史索引（所有股票×所有窗口），用于 search --mode history"""
    config = ctx.obj["config"]
    dc = config.get("data", {})
    raw_dir = dc.get("raw_dir", "data/raw")
    stock_list_file = dc.get("stock_list_file", "data/stock_list.csv")

    import pandas as pd

    stock_names = {}
    if Path(stock_list_file).exists():
        df_list = pd.read_csv(stock_list_file, dtype=str)
        stock_names = dict(zip(df_list["code"], df_list["name"]))

    raw_path = Path(raw_dir)
    stock_data = {}
    for p in raw_path.glob("*.parquet"):
        try:
            df = pd.read_parquet(p)
            df["date"] = pd.to_datetime(df["date"])
            stock_data[p.stem] = df
        except Exception as e:
            logger.warning(f"读取 {p} 失败: {e}")

    if not stock_data:
        click.echo("无可用数据，请先运行 download", err=True)
        sys.exit(1)

    from stock_lookalike.build_index import build_full_history_index
    build_full_history_index(stock_data, stock_names, config, device=device, step=step)
    click.echo("全历史索引构建完成")


# ─── search ───────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("code")
@click.option("--top-k", default=None, type=int, help="返回数量（覆盖 config）")
@click.option("--end-date", default=None, help="窗口截止日期 yyyy-mm-dd，默认取最新")
@click.option("--mode", default="sync",
              type=click.Choice(["sync", "current", "history"]),
              help="sync=同期历史索引 | current=当前最新索引 | history=全历史所有窗口")
@click.option("--format", "fmt", default="table", type=click.Choice(["table", "json"]))
@click.option("--device", default="cpu", show_default=True)
@click.pass_context
def search(ctx, code, top_k, end_date, fmt, device, mode):
    """查询相似股票。--mode: sync=同期对比 | current=现在谁最像 | history=历史上谁最像"""
    config = ctx.obj["config"]
    k = top_k or config.get("search", {}).get("top_k", 10)
    dc = config.get("data", {})
    raw_dir = dc.get("raw_dir", "data/raw")
    stock_list_file = dc.get("stock_list_file", "data/stock_list.csv")
    base_index_dir = Path(dc.get("index_dir", "data/index"))

    import pandas as pd

    # 加载股票名称映射
    stock_names = {}
    if Path(stock_list_file).exists():
        df_list = pd.read_csv(stock_list_file, dtype=str)
        stock_names = dict(zip(df_list["code"], df_list["name"]))

    # 加载查询股票数据
    p = Path(raw_dir) / f"{code}.parquet"
    if not p.exists():
        click.echo(f"未找到 {code} 的缓存数据，请先运行 download --codes {code}", err=True)
        sys.exit(1)

    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])

    # 确定索引目录
    index_dir_override = None
    if mode == "current":
        # 始终用当前最新索引，不管 end_date
        index_dir_override = None
    elif mode == "history":
        full_history_dir = base_index_dir / "full_history"
        if not full_history_dir.exists():
            click.echo("全历史索引不存在，请先运行: python cli.py build-full-index", err=True)
            sys.exit(1)
        index_dir_override = str(full_history_dir)
    elif end_date:  # mode == "sync"
        historical_dir = base_index_dir / end_date
        if not historical_dir.exists():
            click.echo(f"未找到 {end_date} 的历史索引，正在即时构建（约需 1~2 分钟）...")
            raw_path = Path(raw_dir)
            all_stock_data = {}
            for fp in raw_path.glob("*.parquet"):
                try:
                    d = pd.read_parquet(fp)
                    d["date"] = pd.to_datetime(d["date"])
                    all_stock_data[fp.stem] = d
                except Exception:
                    pass
            from stock_lookalike.build_index import build_index as do_build
            do_build(all_stock_data, stock_names, config, end_date=end_date, device=device,
                     index_dir_override=str(historical_dir))
            click.echo(f"历史索引已构建并缓存至 {historical_dir}")
        index_dir_override = str(historical_dir)

    from stock_lookalike.searcher import StockSearcher
    searcher = StockSearcher(config, device=device, index_dir_override=index_dir_override)
    searcher.load()
    results = searcher.search_by_code(code, df, top_k=k, end_date=end_date)

    # 计算实际窗口日期用于展示
    window_size = dc.get("window_size", 30)
    if end_date:
        sub = df[df["date"] <= pd.Timestamp(end_date)]
    else:
        sub = df
    if len(sub) >= window_size:
        start_d = sub["date"].iloc[-window_size].strftime("%Y-%m-%d")
        end_d = sub["date"].iloc[-1].strftime("%Y-%m-%d")
    else:
        start_d, end_d = "N/A", "N/A"

    name = stock_names.get(code, code)
    click.echo(f"\n查询: {name}({code}) | 窗口: {start_d} ~ {end_d}\n")

    if fmt == "json":
        out = [
            {"rank": r.rank, "code": r.code, "name": r.name, "similarity": round(r.similarity, 4)}
            for r in results
        ]
        click.echo(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        click.echo(f"{'排名':<4} {'代码':<8} {'名称':<12} {'相似度':<8} {'窗口'}")
        click.echo("-" * 55)
        for r in results:
            click.echo(f"{r.rank:<4} {r.code:<8} {r.name:<12} {r.similarity:<8.4f} {r.window_end}")


# ─── pipeline ─────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--lookback-days", default=120, show_default=True)
@click.option("--epochs", default=None, type=int)
@click.option("--max-stocks", default=None, type=int, help="调试用：限制股票数量")
@click.option("--device", default=None)
@click.option("--skip-download", is_flag=True, help="跳过下载步骤，直接用本地缓存（内网/离线场景）")
@click.pass_context
def pipeline(ctx, lookback_days, epochs, max_stocks, device, skip_download):
    """一键运行完整流程：download → train → build-index"""
    if skip_download:
        click.echo("跳过下载步骤（使用本地缓存）")
    else:
        click.echo("=" * 50)
        click.echo("步骤 1/3: 下载数据")
        click.echo("=" * 50)
        ctx.invoke(download, lookback_days=lookback_days, codes=None, force_refresh=False, max_workers=8)

    click.echo("=" * 50)
    click.echo("步骤 2/3: 训练模型")
    click.echo("=" * 50)
    ctx.invoke(train, epochs=epochs, max_stocks=max_stocks, device=device)

    click.echo("=" * 50)
    click.echo("步骤 3/3: 建立索引")
    click.echo("=" * 50)
    ctx.invoke(build_index, device=device or "cpu")

    click.echo("\n全流程完成！可使用 `python cli.py search <code>` 查询")


if __name__ == "__main__":
    cli()
