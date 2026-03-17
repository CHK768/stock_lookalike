# A股股票走势相似度检索系统

基于 Transformer 对比学习，找出与目标股票近30日走势最相似的当前个股。

## 核心特点

- **保留幅度信息**：OHLC 归一化为 `price[t]/close[0] - 1`，+3% 和 +6% 涨幅不算相似
- **无监督训练**：SimCLR 风格 NT-Xent Loss，无需人工标注
- **轻量模型**：约 75K 参数，CPU 推理毫秒级
- **增量更新**：每日重建索引无需重新训练

## 快速开始

```bash
pip install -r requirements.txt

# 首次运行（下载数据→训练→建索引）
python cli.py pipeline

# 查询相似股票
python cli.py search 000001
python cli.py search 000001 --top-k 20 --format json
```

## 分步运行

```bash
# 1. 下载数据（默认120天历史）
python cli.py download

# 2. 训练模型（50 epochs，early stopping）
python cli.py train

# 3. 建立索引
python cli.py build-index

# 4. 查询
python cli.py search 000001
```

## 每日增量更新

```bash
python cli.py download --lookback-days 5   # 只补充最近5天
python cli.py build-index                   # 重建索引（无需重训练）
```

## 项目结构

```
stock_lookalike/
├── cli.py                     # Click CLI 入口
├── config.yaml                # 超参数配置
├── requirements.txt
├── stock_lookalike/
│   ├── data_fetcher.py        # AKShare + Parquet 缓存
│   ├── feature_engineer.py    # OHLCV 归一化 + 滑动窗口
│   ├── augmentation.py        # 对比学习数据增强
│   ├── dataset.py             # ContrastiveStockDataset
│   ├── model.py               # Transformer 编码器
│   ├── losses.py              # NT-Xent Loss
│   ├── train.py               # 训练主循环
│   ├── build_index.py         # 批量推理建索引
│   └── searcher.py            # 查询接口
└── tests/                     # 单元测试
```

## 运行测试

```bash
pytest tests/ -v
```

## 模型架构

```
[batch, 30, 5]
  ↓ Linear(5→64) + LayerNorm
  ↓ LearnablePositionalEncoding
  ↓ TransformerEncoderLayer × 3 (Pre-LN, nhead=4)
  ↓ AttentionPooling → [batch, 64]
  ↓ ProjectionHead → [batch, 128]
  ↓ L2 归一化
[batch, 128]
```
