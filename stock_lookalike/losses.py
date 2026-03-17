"""NT-Xent Loss（SimCLR 风格对比损失）"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    对 batch 中的 2N 个样本（N 对正样本）计算 NT-Xent Loss。

    正样本对约定：(i, i+N) 互为正样本，其余 2N-2 个为负样本。

    Args:
        temperature: softmax 温度，越小越难训练但效果更好，通常 0.05~0.1
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1, z2: L2 归一化后的 embedding，各 shape [N, D]

        Returns:
            scalar loss
        """
        N = z1.size(0)
        # [2N, D]
        z = torch.cat([z1, z2], dim=0)

        # [2N, 2N] 余弦相似度矩阵（已 L2 归一化，直接点积）
        sim = torch.mm(z, z.t()) / self.temperature

        # 排除自身
        mask_self = torch.eye(2 * N, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(mask_self, float("-inf"))

        # 正样本位置：i 的正样本是 i+N（对 i<N），i-N（对 i>=N）
        labels = torch.cat([
            torch.arange(N, 2 * N, device=z.device),
            torch.arange(0, N, device=z.device),
        ])  # [2N]

        loss = F.cross_entropy(sim, labels)
        return loss
