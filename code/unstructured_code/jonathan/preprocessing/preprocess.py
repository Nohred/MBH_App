"""
src/preprocessing/preprocess.py
═══════════════════════════════════════════════════════════════════════════════
Pipeline de preprocesamiento — MBH-Seg25
Basado en los hallazgos del EDA:

  - Shape original:   512 × 512 × 32  voxels
  - Spacing original: 0.486 × 0.486 × 5.09  mm  (anisotropía ~10.5×)
  - Acción:           Resample a 1×1×1 mm → ~250 × 250 × 162 voxels
  - HU clip:          [-30, 160] (ventana cerebral estándar)
  - Label:            Se usa label_consensus_staple.nii.gz directamente

Salida por caso (en data/processed/):
  images/  <case_id>.nii.gz   → imagen CT resampleada y normalizada
  labels/  <case_id>.nii.gz   → máscara STAPLE resampleada (nearest-neighbor)

También genera:
  data/splits/train.json  y  val.json  con los manifest de los casos

Uso:
    python -m src.preprocessing.preprocess --config config.yaml
    python -m src.preprocessing.preprocess --config config.yaml --max_cases 10
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from src.utils.common import setup_logger, load_config, set_seed

logger = setup_logger("Preprocess")


# ─── Resample ─────────────────────────────────────────────────────────────────

def resample_volume(
    sitk_img: sitk.Image,
    target_spacing: tuple = (1.0, 1.0, 1.0),
    interpolator = sitk.sitkLinear,
) -> sitk.Image:
    """
    Resamplea un volumen SimpleITK al spacing objetivo.
    Preserva la orientación y origen originales.
    """
    original_spacing = sitk_img.GetSpacing()          # (x, y, z)
    original_size    = sitk_img.GetSize()             # (x, y, z)

    # Calcular el nuevo tamaño para mantener el FOV físico
    new_size = [
        int(round(orig_sz * orig_sp / tgt_sp))
        for orig_sz, orig_sp, tgt_sp
        in zip(original_size, original_spacing, target_spacing)
    ]

    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetOutputSpacing(target_spacing)
    resample_filter.SetSize(new_size)
    resample_filter.SetOutputDirection(sitk_img.GetDirection())
    resample_filter.SetOutputOrigin(sitk_img.GetOrigin())
    resample_filter.SetTransform(sitk.Transform())
    resample_filter.SetDefaultPixelValue(
        sitk_img.GetPixelIDValue() == sitk.sitkFloat32 and -1000.0 or 0
    )
    resample_filter.SetInterpolator(interpolator)

    return resample_filter.Execute(sitk_img)


# ─── Clip y normalización HU ──────────────────────────────────────────────────

def clip_and_normalize(
    arr: np.ndarray,
    clip_min: float = -30.0,
    clip_max: float = 160.0,
) -> np.ndarray:
    """
    1. Clip a la ventana HU cerebral [-30, 160]
    2. Normalización Z-score sobre los voxels dentro de la ventana
       (excluye el fondo/aire de la normalización)

    El resultado tiene media≈0 y std≈1 sobre el tejido cerebral.
    """
    arr = arr.astype(np.float32)
    arr = np.clip(arr, clip_min, clip_max)

    # Máscara de "tejido" (excluye el fondo totalmente negro)
    brain_mask = arr > clip_min

    if brain_mask.sum() > 100:
        mean = arr[brain_mask].mean()
        std  = arr[brain_mask].std()
        arr  = (arr - mean) / (std + 1e-8)
    else:
        # Caso degenrado — normalizar sobre todo el volumen
        arr = (arr - arr.mean()) / (arr.std() + 1e-8)

    return arr


# ─── Pipeline por caso ────────────────────────────────────────────────────────

def preprocess_case(
    case: dict,
    out_image_dir: Path,
    out_label_dir: Path,
    target_spacing: tuple = (1.0, 1.0, 1.0),
    clip_hu: tuple = (-30.0, 160.0),
    overwrite: bool = False,
) -> dict | None:
    """
    Procesa un solo caso: imagen + STAPLE.

    Returns:
        dict con metadata del caso procesado, o None si hubo error.
    """
    case_id = case["case_id"]
    out_img  = out_image_dir / f"{case_id}.nii.gz"
    out_lbl  = out_label_dir / f"{case_id}.nii.gz"

    if not overwrite and out_img.exists() and out_lbl.exists():
        return {
            "case_id": case_id,
            "image":   str(out_img),
            "label":   str(out_lbl),
            "status":  "skipped",
        }

    try:
        # ── 1. Cargar imagen CT ──
        img_sitk = sitk.ReadImage(str(case["image"]), sitk.sitkFloat32)

        # ── 2. Resample imagen (linear) ──
        img_resampled = resample_volume(img_sitk, target_spacing, sitk.sitkLinear)

        # ── 3. Clip + normalizar HU ──
        arr = sitk.GetArrayFromImage(img_resampled)   # (z, y, x) en SimpleITK
        arr = clip_and_normalize(arr, clip_hu[0], clip_hu[1])

        img_out = sitk.GetImageFromArray(arr)
        img_out.CopyInformation(img_resampled)
        sitk.WriteImage(img_out, str(out_img))

        # ── 4. Cargar y resamplear STAPLE (nearest-neighbor para labels) ──
        if case["staple"]:
            lbl_sitk = sitk.ReadImage(str(case["staple"]), sitk.sitkUInt8)
            lbl_resampled = resample_volume(lbl_sitk, target_spacing,
                                             sitk.sitkNearestNeighbor)
            sitk.WriteImage(lbl_resampled, str(out_lbl))
        else:
            # Fallback: usar primer rater disponible
            if not case["raters"]:
                logger.warning(f"{case_id}: sin STAPLE ni raters — saltando label.")
                return None
            rp = next(iter(case["raters"].values()))
            lbl_sitk = sitk.ReadImage(str(rp), sitk.sitkUInt8)
            lbl_resampled = resample_volume(lbl_sitk, target_spacing,
                                             sitk.sitkNearestNeighbor)
            sitk.WriteImage(lbl_resampled, str(out_lbl))
            logger.warning(f"{case_id}: sin STAPLE — se usó primer rater.")

        # ── 5. Registrar metadata ──
        final_size    = img_resampled.GetSize()   # (x, y, z)
        final_spacing = img_resampled.GetSpacing()

        return {
            "case_id":       case_id,
            "image":         str(out_img),
            "label":         str(out_lbl),
            "status":        "processed",
            "original_shape": list(sitk.ReadImage(str(case["image"])).GetSize()),
            "processed_shape": list(final_size),
            "spacing":        list(final_spacing),
        }

    except Exception as e:
        logger.error(f"Error procesando {case_id}: {e}")
        return None


# ─── Escaneo (igual que en EDA) ───────────────────────────────────────────────

def scan_dataset(voxel_dir: Path, rater_prefix: str, rater_ids: list) -> list:
    """Reutiliza la misma lógica de escaneo del EDA."""
    cases = []
    subdirs = sorted([d for d in voxel_dir.iterdir() if d.is_dir()])

    for case_dir in subdirs:
        image_path = case_dir / "image.nii.gz"
        if not image_path.exists():
            continue

        rater_paths = {}
        for rid in rater_ids:
            p = case_dir / f"{rater_prefix}{rid}.nii.gz"
            if p.exists():
                rater_paths[rid] = p

        staple_path = case_dir / "label_consensus_staple.nii.gz"
        unc_path    = case_dir / "label_uncertainty.nii.gz"

        cases.append({
            "case_id":     case_dir.name,
            "image":       image_path,
            "raters":      rater_paths,
            "num_raters":  len(rater_paths),
            "has_staple":  staple_path.exists(),
            "staple":      staple_path if staple_path.exists() else None,
            "uncertainty": unc_path if unc_path.exists() else None,
        })

    return cases


# ─── Train / Val split ────────────────────────────────────────────────────────

def make_splits(
    processed_records: list[dict],
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[list, list]:
    """
    Divide en train/val de forma reproducible.
    Estratificamos por: casos con ≥3 raters (más confiables) → preferimos
    que haya representación de ambos grupos en train y val.
    """
    random.seed(seed)

    # Separar por número de raters (proxy de calidad de anotación)
    high_quality = [r for r in processed_records if r.get("num_raters", 0) >= 3]
    low_quality  = [r for r in processed_records if r.get("num_raters", 0) < 3]

    random.shuffle(high_quality)
    random.shuffle(low_quality)

    # Val proporcional de cada grupo
    n_val_hq = max(1, int(len(high_quality) * val_fraction))
    n_val_lq = max(0, int(len(low_quality)  * val_fraction))

    val   = high_quality[:n_val_hq] + low_quality[:n_val_lq]
    train = high_quality[n_val_hq:] + low_quality[n_val_lq:]

    random.shuffle(train)
    random.shuffle(val)

    return train, val


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_preprocessing(
    config_path: str = "config.yaml",
    max_cases: int | None = None,
    overwrite: bool = False,
):
    cfg = load_config(config_path)
    set_seed(cfg["project"]["seed"])

    voxel_dir    = Path(cfg["data"]["voxel_label_dir"])
    rater_prefix = cfg["data"]["rater_prefix"]
    rater_ids    = cfg["data"]["raters_train"]

    target_spacing = tuple(cfg["preprocessing"]["target_spacing"])
    clip_hu        = tuple(cfg["preprocessing"]["clip_hu"])
    val_fraction   = cfg["data"]["val_split"]

    out_img_dir = Path(cfg["data"]["processed_dir"]) / "images"
    out_lbl_dir = Path(cfg["data"]["processed_dir"]) / "labels"
    splits_dir  = Path(cfg["data"]["splits_dir"])

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 62)
    logger.info("  MBH-Seg25 — Preprocessing")
    logger.info("=" * 62)
    logger.info(f"Target spacing: {target_spacing} mm")
    logger.info(f"HU window:      [{clip_hu[0]}, {clip_hu[1]}]")
    logger.info(f"Val fraction:   {val_fraction}")
    if max_cases:
        logger.info(f"Modo rápido:    {max_cases} casos")

    # Escaneo
    cases = scan_dataset(voxel_dir, rater_prefix, rater_ids)
    if max_cases:
        cases = cases[:max_cases]
    logger.info(f"Casos a procesar: {len(cases)}")

    # Preprocesamiento
    processed = []
    failed    = []

    for case in tqdm(cases, desc="Preprocessing"):
        result = preprocess_case(
            case, out_img_dir, out_lbl_dir,
            target_spacing=target_spacing,
            clip_hu=clip_hu,
            overwrite=overwrite,
        )
        if result is not None:
            result["num_raters"] = case["num_raters"]
            processed.append(result)
        else:
            failed.append(case["case_id"])

    logger.info(f"\nProcesados: {len(processed)}  |  Fallidos: {len(failed)}")
    if failed:
        logger.warning(f"Casos fallidos: {failed}")

    # Splits
    ok = [r for r in processed if r["status"] != "error"]
    train_records, val_records = make_splits(ok, val_fraction, cfg["project"]["seed"])

    # Formato de manifest para MONAI DataLoader
    def to_manifest(records):
        return [{"image": r["image"], "label": r["label"]} for r in records]

    train_manifest = to_manifest(train_records)
    val_manifest   = to_manifest(val_records)

    with open(splits_dir / "train.json", "w") as f:
        json.dump(train_manifest, f, indent=2)
    with open(splits_dir / "val.json", "w") as f:
        json.dump(val_manifest, f, indent=2)

    logger.info(f"\nSplits guardados en {splits_dir}")
    logger.info(f"  Train: {len(train_manifest)} casos")
    logger.info(f"  Val:   {len(val_manifest)} casos")

    # Resumen de shapes procesados
    shapes = [r["processed_shape"] for r in processed if "processed_shape" in r]
    if shapes:
        shapes = np.array(shapes)
        logger.info(f"\nShape procesado (x, y, z):")
        logger.info(f"  mean: {shapes.mean(axis=0).astype(int).tolist()}")
        logger.info(f"  min:  {shapes.min(axis=0).tolist()}")
        logger.info(f"  max:  {shapes.max(axis=0).tolist()}")

    # Guardar metadata completa
    meta = {
        "total_processed": len(processed),
        "total_failed":    len(failed),
        "failed_cases":    failed,
        "train_cases":     len(train_manifest),
        "val_cases":       len(val_manifest),
        "config": {
            "target_spacing": list(target_spacing),
            "clip_hu":        list(clip_hu),
        },
    }
    with open(splits_dir / "preprocessing_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("\n" + "=" * 62)
    logger.info("  Preprocesamiento completado ✓")
    logger.info(f"  Imágenes: {out_img_dir}")
    logger.info(f"  Labels:   {out_lbl_dir}")
    logger.info(f"  Splits:   {splits_dir}")
    logger.info("=" * 62)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    type=str, default="config.yaml")
    parser.add_argument("--max_cases", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true",
                        help="Reprocesar casos aunque ya existan en disco")
    args = parser.parse_args()
    run_preprocessing(args.config, args.max_cases, args.overwrite)
