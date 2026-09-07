"""
src/training/data.py
═══════════════════════════════════════════════════════════════════════════════
DataLoaders y transforms MONAI para MBH-Seg25.

Los volúmenes procesados son ~248×248×162 voxels a 1×1×1 mm.
Entrenamos con patches aleatorios de 128×128×128 (RandCropByPosNegLabeld).
"""

import json
from pathlib import Path

from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ScaleIntensityRangePercentilesd,
    SpatialPadd,
    ToTensord,
)

from src.utils.common import setup_logger

logger = setup_logger("Data")


def get_transforms(mode: str, cfg: dict) -> Compose:
    """
    Construye el pipeline de transforms MONAI.

    mode: "train" | "val"

    Los datos ya están normalizados (Z-score en preprocesamiento),
    pero aplicamos augmentaciones de intensidad adicionales en train.
    """
    patch_size = tuple(cfg["training"]["model"]["img_size"])   # (128, 128, 128)

    base = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        # Orientación canónica RAS (no debería cambiar nada si el resample fue correcto,
        # pero es buena práctica garantizarlo)
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        # Asegurar float32 en imagen y uint8 en label
        EnsureTyped(keys=["image"], dtype="float32"),
        EnsureTyped(keys=["label"], dtype="long"),
        # Pad mínimo para garantizar que el patch cabe en cualquier volumen
        # (algunos casos tienen shape_z=132 que es > 128, OK)
        SpatialPadd(keys=["image", "label"], spatial_size=patch_size, mode="constant"),
    ]

    if mode == "train":
        augmentations = [
            # Crop inteligente: 2 patches positivos por cada 1 negativo
            # (garantiza que el modelo vea hemorragia en la mayoría de los patches)
            RandCropByPosNegLabeld(
                keys         = ["image", "label"],
                label_key    = "label",
                spatial_size = patch_size,
                pos          = 3,     # 3 crops con al menos 1 voxel positivo
                neg          = 1,     # 1 crop aleatorio (puede ser solo background)
                num_samples  = 2,     # 4 patches por volumen por época
                image_key    = "image",
                image_threshold = 0,  # umbral para considerar voxel "cerebro"
            ),
            # Flips geométricos (el cerebro tiene simetría bilateral)
            RandFlipd(keys=["image","label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image","label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image","label"], prob=0.5, spatial_axis=2),
            # Rotación 90° (invariante para CT axial)
            RandRotate90d(keys=["image","label"], prob=0.5, max_k=3),
            # Augmentaciones de intensidad (solo imagen)
            RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.1),
            RandGaussianSmoothd(
                keys=["image"], prob=0.2,
                sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0),
            ),
            RandScaleIntensityd(keys=["image"], factors=0.15, prob=0.5),
            RandShiftIntensityd(keys=["image"], offsets=0.1,  prob=0.5),
            ToTensord(keys=["image", "label"]),
        ]
        return Compose(base + augmentations)

    else:  # val
        val_only = [
            # En validación usamos el volumen completo (sliding window en inferencia)
            # Solo necesitamos el tensor listo para el modelo
            ToTensord(keys=["image", "label"]),
        ]
        return Compose(base + val_only)


def build_dataloaders(cfg: dict) -> tuple:
    """
    Construye train y val DataLoaders.

    Con 164 casos de train, CacheDataset con cache_rate=1.0 y 48GB RAM
    carga todo en memoria → iteraciones muy rápidas.

    Returns: (train_loader, val_loader)
    """
    splits_dir = Path(cfg["data"]["splits_dir"])

    with open(splits_dir / "train.json") as f:
        train_files = json.load(f)
    with open(splits_dir / "val.json") as f:
        val_files = json.load(f)

    logger.info(f"Train cases: {len(train_files)}  |  Val cases: {len(val_files)}")

    train_transforms = get_transforms("train", cfg)
    val_transforms   = get_transforms("val",   cfg)

    # CacheDataset carga y aplica transforms deterministas en RAM
    # Con 48GB RAM y ~164 casos de ~50MB c/u → ~8GB total, entra holgado
    train_ds = CacheDataset(
        data       = train_files,
        transform  = train_transforms,
        cache_rate = 1.0,        # cachear todo en RAM
        num_workers = 0,
    )

    val_ds = CacheDataset(
        data       = val_files,
        transform  = val_transforms,
        cache_rate = 1.0,
        num_workers = 0,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size  = cfg["training"]["batch_size"],
        shuffle     = True,
        num_workers = 0,
        pin_memory  = True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size  = 1,          # validación siempre batch=1 (sliding window)
        shuffle     = False,
        num_workers = 0,
        pin_memory  = True,
    )

    return train_loader, val_loader
