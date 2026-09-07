"""
src/training/train.py
═══════════════════════════════════════════════════════════════════════════════
Training loop completo — MBH-Seg25 SwinUNETR

Características:
  - AMP float16 (RTX 5060 Ti Blackwell cu128)
  - Sliding window inference en validación
  - Dice por clase en validación (para monitorear Epidural específicamente)
  - Checkpoint del mejor modelo por val_mean_dice
  - TensorBoard logging
  - Reanudación de entrenamiento desde checkpoint

Uso:
    python -m src.training.train --config config.yaml
    python -m src.training.train --config config.yaml --resume experiments/checkpoints/best.pth
"""

import argparse
import time
from pathlib import Path

import torch
from torch.amp import GradScaler
from torch import autocast
from torch.utils.tensorboard import SummaryWriter
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.utils import set_determinism
from monai.data import decollate_batch
from monai.transforms import AsDiscrete, Compose

from src.models.swin_unetr import build_model, load_checkpoint
from src.training.losses import build_loss, CLASS_WEIGHTS
from src.training.data import build_dataloaders
from src.utils.common import setup_logger, load_config, set_seed

logger = setup_logger("Train", log_file="experiments/logs/train.log")

CLASS_NAMES = {
    0: "Background",
    1: "Epidural",
    2: "Intraparenchymal",
    3: "Intraventricular",
    4: "Subarachnoid",
    5: "Subdural",
}


# ─── Post-processing para métricas ────────────────────────────────────────────

def build_post_transforms(num_classes: int):
    """Convierte logits → one-hot para DiceMetric de MONAI."""
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=num_classes)])
    post_label = Compose([AsDiscrete(to_onehot=num_classes)])
    return post_pred, post_label


# ─── Validación ───────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, val_loader, dice_metric, post_pred, post_label, cfg, device):
    """
    Validación con sliding window inference.
    Retorna (mean_dice, dice_por_clase_dict).
    """
    model.eval()
    dice_metric.reset()

    roi_size   = tuple(cfg["training"]["sw_roi_size"])
    sw_batch   = cfg["training"]["sw_batch_size"]
    sw_overlap = cfg["training"]["sw_overlap"]
    num_cls    = cfg["training"]["model"]["out_channels"]

    for batch in val_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        with autocast("cuda", dtype=torch.float16):
            outputs = sliding_window_inference(
                inputs      = images,
                roi_size    = roi_size,
                sw_batch_size = sw_batch,
                predictor   = model,
                overlap     = sw_overlap,
                mode        = "gaussian",    # ponderación gaussiana en bordes de patches
            )

        # Post-processing
        outputs_list = decollate_batch(outputs)
        labels_list  = decollate_batch(labels)

        outputs_onehot = [post_pred(o)  for o in outputs_list]
        labels_onehot  = [post_label(l) for l in labels_list]

        dice_metric(y_pred=outputs_onehot, y=labels_onehot)

    # Dice por clase: tensor (num_classes,)
    dice_per_class = dice_metric.aggregate()         # (C,) incluyendo background
    mean_dice_no_bg = dice_per_class[1:].mean().item()  # sin background

    dice_dict = {
        CLASS_NAMES[i]: float(dice_per_class[i].item())
        for i in range(num_cls)
    }

    return mean_dice_no_bg, dice_dict


# ─── Training loop ────────────────────────────────────────────────────────────

def train(cfg: dict, resume_path: str | None = None):
    set_seed(cfg["project"]["seed"])
    set_determinism(seed=cfg["project"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

    # ── Directorios ──
    ckpt_dir = Path(cfg["experiments"]["ckpt_dir"])
    tb_dir   = Path(cfg["experiments"]["tb_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    # ── Modelo ──
    model = build_model(cfg).to(device)

    # ── Loss ──
    loss_fn = build_loss(cfg).to(device)
    logger.info(f"Loss: {cfg['training']['loss']['name']}")

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg["training"]["lr"],
        weight_decay = cfg["training"]["weight_decay"],
    )

    # ── Scheduler: cosine annealing con warm restarts ──
    epochs = cfg["training"]["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0     = 50,     # reinicia cada 50 épocas
        T_mult  = 2,      # duplica el periodo en cada reinicio
        eta_min = 1e-6,
    )

    # ── AMP ──
    use_amp = cfg["training"]["amp"]
    scaler  = GradScaler("cuda", enabled=use_amp)
    amp_dtype = torch.float16  # float16 estable en nightly cu128

    # ── Data ──
    logger.info("Construyendo DataLoaders (cacheando dataset en RAM)...")
    train_loader, val_loader = build_dataloaders(cfg)

    # ── Métricas ──
    num_cls    = cfg["training"]["model"]["out_channels"]
    dice_metric = DiceMetric(
        include_background = True,
        reduction          = "mean_batch",
        get_not_nans       = False,
    )
    post_pred, post_label = build_post_transforms(num_cls)

    # ── TensorBoard ──
    writer = SummaryWriter(log_dir=str(tb_dir))

    # ── Estado inicial ──
    start_epoch = 0
    best_dice   = 0.0
    history     = {"train_loss": [], "val_dice": [], "val_dice_per_class": []}

    # ── Reanudar desde checkpoint ──
    if resume_path and Path(resume_path).exists():
        ckpt = load_checkpoint(model, resume_path, device=str(device))
        optimizer.load_state_dict(ckpt.get("optimizer_state_dict", {}))
        scheduler.load_state_dict(ckpt.get("scheduler_state_dict", {}))
        start_epoch = ckpt.get("epoch", 0) + 1
        best_dice   = ckpt.get("best_dice", 0.0)
        logger.info(f"Reanudando desde época {start_epoch}, best_dice={best_dice:.4f}")

    val_every = cfg["training"]["val_every"]
    logger.info(f"\nIniciando entrenamiento — {epochs} épocas")
    logger.info(f"Validación cada {val_every} épocas")
    logger.info("─" * 62)

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                logits = model(images)
                loss   = loss_fn(logits, labels)

            scaler.scale(loss).backward()

            # Gradient clipping (estabilidad con AMP)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()

        mean_loss = epoch_loss / max(n_batches, 1)
        elapsed   = time.time() - epoch_start
        lr_now    = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(mean_loss)
        writer.add_scalar("Loss/train", mean_loss, epoch)
        writer.add_scalar("LR",         lr_now,    epoch)

        # ── Logging por época ──
        logger.info(
            f"Epoch {epoch+1:03d}/{epochs} | "
            f"loss={mean_loss:.4f} | "
            f"lr={lr_now:.2e} | "
            f"time={elapsed:.1f}s"
        )

        # ── Validación ──
        if (epoch + 1) % val_every == 0 or epoch == epochs - 1:
            logger.info("  → Validando...")
            val_dice, dice_dict = validate(
                model, val_loader, dice_metric, post_pred, post_label, cfg, device
            )

            history["val_dice"].append(val_dice)
            history["val_dice_per_class"].append(dice_dict)

            writer.add_scalar("Dice/val_mean_no_bg", val_dice, epoch)
            for cls_name, d in dice_dict.items():
                writer.add_scalar(f"Dice/val_{cls_name}", d, epoch)

            # Log por clase (especialmente útil para Epidural)
            cls_log = "  ".join(
                f"{name[:5]}={d:.3f}"
                for name, d in dice_dict.items()
                if name != "Background"
            )
            logger.info(f"  Val Dice (no_bg): {val_dice:.4f}  [{cls_log}]")

            # ── Guardar mejor checkpoint ──
            if val_dice > best_dice:
                best_dice = val_dice
                ckpt_path = ckpt_dir / "best.pth"
                torch.save({
                    "epoch":                   epoch,
                    "model_state_dict":        model.state_dict(),
                    "optimizer_state_dict":    optimizer.state_dict(),
                    "scheduler_state_dict":    scheduler.state_dict(),
                    "best_dice":               best_dice,
                    "val_dice_per_class":      dice_dict,
                    "config":                  cfg,
                }, str(ckpt_path))
                logger.info(f"  ★ Nuevo mejor modelo: dice={best_dice:.4f} → {ckpt_path}")

        # ── Checkpoint periódico (cada 50 épocas) ──
        if (epoch + 1) % 50 == 0:
            periodic_path = ckpt_dir / f"epoch_{epoch+1:03d}.pth"
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_dice":            best_dice,
            }, str(periodic_path))
            logger.info(f"  Checkpoint periódico guardado: {periodic_path}")

    writer.close()
    logger.info("=" * 62)
    logger.info(f"  Entrenamiento completado ✓")
    logger.info(f"  Mejor Val Dice (sin BG): {best_dice:.4f}")
    logger.info(f"  Checkpoint: {ckpt_dir}/best.pth")
    logger.info("=" * 62)

    return best_dice, history


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MBH-Seg25 Training")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--resume", type=str, default=None,
                        help="Ruta al checkpoint para reanudar entrenamiento")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, resume_path=args.resume)
