"""对比学习 Dataset：支持相邻窗口正样本对 + 验证集分割"""
import numpy as np
import torch
from torch.utils.data import Dataset

from .augmentation import StockAugmentor


class ContrastiveStockDataset(Dataset):
    """
    输入: list of np.ndarray, each shape [T, 5]

    正样本对策略（混合）：
    - neighbor_prob 概率：从 neighbor_pairs 中随机取一对相邻窗口（同股票软正样本）
    - 1-neighbor_prob 概率：对同一窗口生成两个增强视图（传统 SimCLR 正样本）
    """

    def __init__(
        self,
        windows: list,
        neighbor_pairs: list | None = None,
        augmentor: StockAugmentor | None = None,
        neighbor_prob: float = 0.7,
    ):
        self.windows = windows
        self.neighbor_pairs = neighbor_pairs or []
        self.augmentor = augmentor or StockAugmentor()
        self.neighbor_prob = neighbor_prob if self.neighbor_pairs else 0.0

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.neighbor_pairs and np.random.random() < self.neighbor_prob:
            pair_idx = np.random.randint(len(self.neighbor_pairs))
            ia, ib = self.neighbor_pairs[pair_idx]
            view1 = self.augmentor._augment(self.windows[ia])
            view2 = self.augmentor._augment(self.windows[ib])
        else:
            view1, view2 = self.augmentor(self.windows[idx])
        return (
            torch.from_numpy(view1).float(),
            torch.from_numpy(view2).float(),
        )


def build_dataset_from_stocks(
    stock_data: dict,
    window_size: int = 30,
    val_ratio: float = 0.1,
    neighbor_step: int = 5,
) -> tuple["ContrastiveStockDataset", "ContrastiveStockDataset"]:
    """
    从多只股票数据构建训练集和验证集。

    每只股票按时序划分：前 (1-val_ratio) 为训练集，后 val_ratio 为验证集。
    相邻窗口正样本对仅在训练集内部构建（步长 neighbor_step）。

    Args:
        stock_data: {code: DataFrame}
        window_size: 滑动窗口大小
        val_ratio: 验证集比例（按时序末尾截取）
        neighbor_step: 相邻窗口步长

    Returns:
        (train_dataset, val_dataset)
    """
    from .feature_engineer import sliding_windows

    train_windows, val_windows, neighbor_pairs = [], [], []

    for code, df in stock_data.items():
        ws = sliding_windows(df, window_size=window_size)
        if not ws:
            continue
        n = len(ws)
        split = max(2, int(n * (1 - val_ratio)))
        train_ws = ws[:split]
        val_ws = ws[split:]

        start_idx = len(train_windows)
        train_windows.extend(train_ws)
        val_windows.extend(val_ws)

        # 只在训练集内部构建相邻正样本对
        for i in range(len(train_ws) - neighbor_step):
            neighbor_pairs.append((start_idx + i, start_idx + i + neighbor_step))

    train_dataset = ContrastiveStockDataset(
        train_windows, neighbor_pairs=neighbor_pairs, neighbor_prob=0.7
    )
    val_dataset = ContrastiveStockDataset(val_windows, neighbor_pairs=None)
    return train_dataset, val_dataset
