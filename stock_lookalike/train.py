"""训练主循环：对比学习训练 Transformer 编码器"""
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from .model import build_model, DualChannelStockEncoder
from .losses import NTXentLoss
from .dataset import ContrastiveStockDataset

logger = logging.getLogger(__name__)


def train(
    dataset: ContrastiveStockDataset,
    config: dict,
    device: str | None = None,
    resume: bool = False,
    val_dataset: ContrastiveStockDataset | None = None,
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
        num_workers=0,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )
    val_loader = None
    if val_dataset and len(val_dataset) >= batch_size:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device == "cuda"),
            drop_last=False,
        )

    model = build_model(config).to(device)

    # 双通道模型：volume 分支使用 2x 学习率
    if isinstance(model, DualChannelStockEncoder):
        vol_param_ids = {
            id(p) for name, p in model.named_parameters()
            if name.startswith("vol_")
        }
        optimizer = torch.optim.AdamW([
            {"params": [p for p in model.parameters() if id(p) not in vol_param_ids], "lr": lr},
            {"params": [p for p in model.parameters() if id(p) in vol_param_ids],     "lr": lr * 2},
        ], weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    warmup_epochs = tc.get("warmup_epochs", 5)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs - warmup_epochs, 1)
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )
    criterion = NTXentLoss(temperature=temperature)
    scaler = GradScaler("cuda", enabled=mixed_precision)

    best_loss = float("inf")
    no_improve = 0
    history = []
    start_epoch = 1

    last_ckpt_path = ckpt_dir / "last.pt"
    if resume and last_ckpt_path.exists():
        last_ckpt = torch.load(last_ckpt_path, map_location=device)
        model.load_state_dict(last_ckpt["model_state"])
        optimizer.load_state_dict(last_ckpt["optimizer_state"])
        scheduler.load_state_dict(last_ckpt["scheduler_state"])
        start_epoch = last_ckpt["epoch"] + 1
        best_loss = last_ckpt.get("best_loss", float("inf"))
        no_improve = last_ckpt.get("no_improve", 0)
        history = last_ckpt.get("history", [])
        logger.info(f"从 epoch {start_epoch} 恢复训练（best_loss={best_loss:.4f}，no_improve={no_improve}）")
    elif resume:
        logger.warning(f"未找到 {last_ckpt_path}，从头开始训练")

    for epoch in range(start_epoch, epochs + 1):
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

        # 验证集 loss
        val_loss = None
        if val_loader is not None:
            model.eval()
            val_total, val_n = 0.0, 0
            with torch.no_grad():
                for v1, v2 in val_loader:
                    v1, v2 = v1.to(device), v2.to(device)
                    z1, z2 = model(v1), model(v2)
                    val_total += criterion(z1, z2).item()
                    val_n += 1
            val_loss = val_total / max(val_n, 1)
            model.train()

        monitor_loss = val_loss if val_loss is not None else avg_loss
        record = {"epoch": epoch, "loss": avg_loss}
        if val_loss is not None:
            record["val_loss"] = val_loss
        history.append(record)
        if val_loss is not None:
            logger.info(f"Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  val_loss={val_loss:.4f}")
        else:
            logger.info(f"Epoch {epoch}/{epochs}  loss={avg_loss:.4f}")

        model_type = config.get("model", {}).get("model_type", "single")

        # 保存最优（以 val_loss 为准，无验证集则用 train loss）
        if monitor_loss < best_loss:
            best_loss = monitor_loss
            no_improve = 0
            ckpt_path = ckpt_dir / "best.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "loss": best_loss,
                "model_type": model_type,
            }, ckpt_path)
            logger.info(f"  保存最优模型 loss={best_loss:.4f} → {ckpt_path}")
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch}（连续 {patience} epoch 无改善）")
                # 保存最终 last.pt 供下次续训参考
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "loss": avg_loss,
                    "best_loss": best_loss,
                    "no_improve": no_improve,
                    "history": history,
                    "model_type": model_type,
                }, last_ckpt_path)
                break

        # 每 epoch 保存 last.pt（断点续训用）
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "loss": avg_loss,
            "best_loss": best_loss,
            "no_improve": no_improve,
            "history": history,
            "model_type": model_type,
        }, last_ckpt_path)

    # 加载最优权重
    ckpt_path = ckpt_dir / "best.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        logger.info(f"已加载最优权重（epoch={ckpt['epoch']}, loss={ckpt['loss']:.4f}）")

    return model, history


def load_model(config: dict, device: str = "cpu") -> torch.nn.Module:
    """从 checkpoint 加载模型（自动检测 model_type）"""
    ckpt_dir = Path(config.get("train", {}).get("checkpoint_dir", "checkpoints"))
    ckpt_path = ckpt_dir / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"未找到 checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    # checkpoint 中的 model_type 优先；老 checkpoint 无此字段则回退 single
    if "model_type" in ckpt:
        config.setdefault("model", {})["model_type"] = ckpt["model_type"]
    else:
        config.setdefault("model", {})["model_type"] = "single"
    model = build_model(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model
