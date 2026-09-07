"""
src/models/swin_unetr.py
Construcción del modelo SwinUNETR con configuración desde config.yaml.
"""

import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

from src.utils.common import setup_logger

logger = setup_logger("Model")


def build_model(cfg: dict) -> nn.Module:
    """
    Construye SwinUNETR según la configuración.
    Con feature_size=48 e img_size=128³ usa ~8-9 GB de VRAM con AMP float16.
    """
    mcfg = cfg["training"]["model"]

    model = SwinUNETR(
        in_channels     = mcfg["in_channels"],
        out_channels    = mcfg["out_channels"],      # 6 clases
        feature_size    = mcfg["feature_size"],      # 48
        use_checkpoint  = mcfg["use_checkpoint"],    # gradient checkpointing
        use_v2          = True,                      # SwinUNETR v2
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"SwinUNETR v2 — parámetros: {n_params/1e6:.1f}M")
    logger.info(f"  img_size={mcfg['img_size']}  "
                f"feature_size={mcfg['feature_size']}  "
                f"out_channels={mcfg['out_channels']}")

    return model


def load_checkpoint(model: nn.Module, ckpt_path: str, device: str = "cuda") -> dict:
    """Carga un checkpoint guardado por el training loop."""
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info(f"Checkpoint cargado: {ckpt_path}  (epoch {ckpt.get('epoch', '?')})")
    return ckpt
