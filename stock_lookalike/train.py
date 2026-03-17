"""训练主循环：对比学习训练 Transformer 编码器"""
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from .model import build_model
from .losses import NTXentLoss
from .dataset import ContrastiveStockDataset

logger = logging.getLogger(__name__)


def train(
    dataset: ContrastiveStockDataset,
    config: dict,
    device: str | None = None,
) -> tuple:
    """
    训练入口。

    Args:
        dataset: ContrastiveStockDataset
        config: 完整 config dict（来自 config.yaml）
        device: 'cuda' / 'mps' / 'cpu'，None 则自动检测

    Returns:
        (model, history) where history = list of {'epoch', 'loss'}
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    logger.info(f"使用设备: {device}")

    tc = config.get("train", {})
    batch_size = tc.get("batch_size", 256)
    lr = tc.get("lr", 3e-4)
    weight_decay = tc.get("weight_decay", 1e-4)
    epochs = tc.get("epochs", 50)
    patience = tc.get("early_stopping_patience", 10)
    temperature = tc.get("temperature", 0.07)
    ckpt_dir = Path(tc.get("checkpoint_dir", "checkpoints"))
    mixed_precision = tc.get("mixed_precision", True) and device == "cuda"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )

    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = NTXentLoss(temperature=temperature)
    scaler = GradScaler("cuda", enabled=mixed_precision)

    best_loss = float("inf")
    no_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for view1, view2 in loader:
            view1 = view1.to(device)
            view2 = view2.to(device)

            optimizer.zero_grad()
            with autocast("cuda", enabled=mixed_precision):
                z1 = model(view1)
                z2 = model(view2)
                loss = criterion(z1, z2)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        history.append({"epoch": epoch, "loss": avg_loss})
        logger.info(f"Epoch {epoch}/{epochs}  loss={avg_loss:.4f}")

        # 保存最优
        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve = 0
            ckpt_path = ckpt_dir / "best.pt"
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "loss": best_loss}, ckpt_path)
            logger.info(f"  保存最优模型 loss={best_loss:.4f} → {ckpt_path}")
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch}（连续 {patience} epoch 无改善）")
                break

    # 加载最优权重
    ckpt_path = ckpt_dir / "best.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        logger.info(f"已加载最优权重（epoch={ckpt['epoch']}, loss={ckpt['loss']:.4f}）")

    return model, history


def load_model(config: dict, device: str = "cpu") -> torch.nn.Module:
    """从 checkpoint 加载模型"""
    ckpt_dir = Path(config.get("train", {}).get("checkpoint_dir", "checkpoints"))
    ckpt_path = ckpt_dir / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"未找到 checkpoint: {ckpt_path}")
    model = build_model(config).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model
