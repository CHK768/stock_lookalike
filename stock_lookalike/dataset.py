"""对比学习 Dataset：每次返回同一窗口的两个增强视图"""
import numpy as np
import torch
from torch.utils.data import Dataset

from .augmentation import StockAugmentor


class ContrastiveStockDataset(Dataset):
    """
    输入: list of np.ndarray, each shape [T, 5]
    每次 __getitem__ 对同一窗口生成两个增强视图作为正样本对。
    """

    def __init__(self, windows: list[np.ndarray], augmentor: StockAugmentor | None = None):
        self.windows = windows
        self.augmentor = augmentor or StockAugmentor()

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        window = self.windows[idx]
        view1, view2 = self.augmentor(window)
        return (
            torch.from_numpy(view1).float(),
            torch.from_numpy(view2).float(),
        )


def build_dataset_from_stocks(
    stock_data: dict,
    window_size: int = 30,
) -> "ContrastiveStockDataset":
    """
    从多只股票数据构建训练集。

    Args:
        stock_data: {code: DataFrame} from data_fetcher
        window_size: 滑动窗口大小

    Returns:
        ContrastiveStockDataset
    """
    from .feature_engineer import sliding_windows

    all_windows = []
    for code, df in stock_data.items():
        ws = sliding_windows(df, window_size=window_size)
        all_windows.extend(ws)

    return ContrastiveStockDataset(all_windows)
