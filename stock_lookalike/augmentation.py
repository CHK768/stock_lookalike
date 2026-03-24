"""对比学习时序数据增强（不改变幅度信息）"""
import numpy as np


class StockAugmentor:
    """
    对比学习增强策略：
    - 高斯噪声 (std=0.015)，不改变幅度
    - 时序平移 ±3天，边界填0
    - 随机遮蔽 p=0.12/天

    注意：不使用幅度缩放，以保留绝对涨跌幅信息。
    """

    def __init__(
        self,
        noise_std: float = 0.015,
        max_shift: int = 3,
        mask_prob: float = 0.12,
    ):
        self.noise_std = noise_std
        self.max_shift = max_shift
        self.mask_prob = mask_prob

    def __call__(self, window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        对同一窗口生成两个增强视图。

        Args:
            window: np.ndarray [T, 5]

        Returns:
            (view1, view2): 各 shape [T, 5]
        """
        return self._augment(window), self._augment(window)

    def _augment(self, window: np.ndarray) -> np.ndarray:
        x = window.copy()
        x = self._add_gaussian_noise(x)
        x = self._temporal_shift(x)
        x = self._random_mask(x)
        return x

    def _add_gaussian_noise(self, x: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0, self.noise_std, size=x.shape).astype(np.float32)
        return x + noise

    def _temporal_shift(self, x: np.ndarray) -> np.ndarray:
        shift = np.random.randint(-self.max_shift, self.max_shift + 1)
        if shift == 0:
            return x
        T = x.shape[0]
        result = np.zeros_like(x)
        if shift > 0:
            result[shift:] = x[:T - shift]
        else:
            result[:T + shift] = x[-shift:]
        return result

    def _random_mask(self, x: np.ndarray) -> np.ndarray:
        T = x.shape[0]
        mask = np.random.random(T) < self.mask_prob
        x = x.copy()
        x[mask] = 0.0
        return x
