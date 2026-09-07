"""
src/preprocessing/eda.py
═══════════════════════════════════════════════════════════════════════════════
Análisis Exploratorio de Datos — MBH-Seg25
Adaptado a la estructura real del dataset:

  ID_xxxxxxxx_ID_yyyyyy/
    image.nii.gz
    label_annot_1.nii.gz   (no siempre presente)
    label_annot_2.nii.gz
    label_annot_3.nii.gz
    label_annot_4.nii.gz   (no siempre presente)
    label_consensus_staple.nii.gz   ← ya calculado por los autores
    label_uncertainty.nii.gz        ← mapa de incertidumbre inter-rater

Hallazgos del inspector:
  - Spacing: ~0.488 x 0.488 x 5.3 mm (muy anisotrópico)
  - Shape:   512 x 512 x 27 aprox
  - Labels:  valores no necesariamente consecutivos → se detectan automáticamente
  - 6 clases en CSV: epidural, intraparenchymal, intraventricular,
                     subarachnoid, subdural (+background)

Uso:
    python -m src.preprocessing.eda --config config.yaml
    python -m src.preprocessing.eda --config config.yaml --max_cases 15
"""

import argparse
import json
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from tqdm import tqdm

from src.utils.common import setup_logger, load_config, set_seed
from src.utils.nifti_io import load_nifti_nib, get_volume_info

warnings.filterwarnings("ignore")
logger = setup_logger("EDA")

# ─── Constantes ───────────────────────────────────────────────────────────────

CLASS_NAMES = {
    0: "Background",
    1: "Epidural",
    2: "Intraparenchymal",
    3: "Intraventricular",
    4: "Subarachnoid",
    5: "Subdural",
}

CLASS_COLORS = {
    0: "#333333",
    1: "#E63946",
    2: "#F4A261",
    3: "#A8DADC",
    4: "#457B9D",
    5: "#2A9D8F",
}

plt.rcParams.update({
    "figure.dpi":      150,
    "font.family":     "DejaVu Sans",
    "axes.titlesize":  11,
    "axes.labelsize":  10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})


# ─── 1. Escaneo ───────────────────────────────────────────────────────────────

def scan_dataset(voxel_dir: Path, rater_prefix: str, rater_ids: list) -> list:
    cases = []
    subdirs = sorted([d for d in voxel_dir.iterdir() if d.is_dir()])
    logger.info(f"Carpetas encontradas: {len(subdirs)}")

    for case_dir in subdirs:
        image_path = case_dir / "image.nii.gz"
        if not image_path.exists():
            logger.warning(f"Sin imagen: {case_dir.name}")
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

    logger.info(f"Casos válidos: {len(cases)}")
    logger.info(f"Con STAPLE:    {sum(1 for c in cases if c['has_staple'])}")
    for n in [4, 3, 2, 1]:
        logger.info(f"Con {n} rater(s): {sum(1 for c in cases if c['num_raters'] == n)}")
    return cases


# ─── 2. Descubrir valores de labels ──────────────────────────────────────────

def discover_label_values(cases: list, max_cases: int = 20) -> dict:
    """Detecta qué valores enteros aparecen en las máscaras."""
    logger.info("Descubriendo valores únicos en labels...")
    rater_vals_set  = set()
    staple_vals_set = set()

    for case in tqdm(cases[:max_cases], desc="Label values"):
        for rp in case["raters"].values():
            try:
                arr, _ = load_nifti_nib(rp)
                rater_vals_set.update(np.unique(arr).tolist())
            except Exception:
                pass
        if case["staple"]:
            try:
                arr, _ = load_nifti_nib(case["staple"])
                staple_vals_set.update(np.unique(arr).tolist())
            except Exception:
                pass

    rater_int  = sorted([int(v) for v in rater_vals_set  if v == int(v) and v >= 0])
    staple_all = sorted(staple_vals_set)
    is_prob    = len(staple_all) > 10

    logger.info(f"Valores en rater labels: {rater_int}")
    logger.info(f"Valores en STAPLE (muestra): {staple_all[:10]}"
                f"{'...' if len(staple_all) > 10 else ''}")
    if is_prob:
        logger.info("STAPLE es un mapa de probabilidad (soft label)")
    else:
        logger.info("STAPLE es una máscara dura (hard label)")

    return {
        "rater_label_values": rater_int,
        "staple_values":      staple_all[:20],
        "staple_is_prob_map": is_prob,
        "n_unique_staple":    len(staple_all),
    }


# ─── 3. Metadata de volúmenes ─────────────────────────────────────────────────

def analyze_volumes(cases: list, max_cases=None) -> pd.DataFrame:
    logger.info("Analizando metadata de volúmenes...")
    records = []
    subset = cases[:max_cases] if max_cases else cases

    for case in tqdm(subset, desc="Volume metadata"):
        try:
            info = get_volume_info(case["image"])
            records.append({
                "case_id":    case["case_id"],
                "shape_z":    info.shape[0],
                "shape_y":    info.shape[1],
                "shape_x":    info.shape[2],
                "spacing_x":  info.spacing[0],
                "spacing_y":  info.spacing[1],
                "spacing_z":  info.spacing[2],
                "hu_min":     info.hu_min,
                "hu_max":     info.hu_max,
                "hu_mean":    info.hu_mean,
                "hu_std":     info.hu_std,
                "num_raters": case["num_raters"],
                "has_staple": case["has_staple"],
            })
        except Exception as e:
            logger.warning(f"Error {case['case_id']}: {e}")

    return pd.DataFrame(records)


# ─── 4. Distribución de clases ────────────────────────────────────────────────

def analyze_class_distribution(cases: list, label_values: list, max_cases=None):
    logger.info("Analizando distribución de clases...")
    records = []
    hu_agg = {v: [] for v in label_values if v > 0}
    subset = cases[:max_cases] if max_cases else cases

    for case in tqdm(subset, desc="Class distribution"):
        try:
            image_arr, _ = load_nifti_nib(case["image"])

            # Fuente de labels: STAPLE duro → primer rater
            if case["has_staple"]:
                arr, _ = load_nifti_nib(case["staple"])
                if len(np.unique(arr)) > 10:   # es soft → usar rater
                    if not case["raters"]:
                        continue
                    arr, _ = load_nifti_nib(next(iter(case["raters"].values())))
            elif case["raters"]:
                arr, _ = load_nifti_nib(next(iter(case["raters"].values())))
            else:
                continue

            label_arr = arr.astype(np.int32)

            import SimpleITK as sitk
            img_s = sitk.ReadImage(str(case["image"]))
            sp = img_s.GetSpacing()
            vox_vol = sp[0] * sp[1] * sp[2]

            row = {"case_id": case["case_id"], "num_raters": case["num_raters"]}
            for v in label_values:
                mask  = label_arr == v
                count = int(mask.sum())
                row[f"cls{v}_present"] = count > 0
                row[f"cls{v}_vol_cm3"] = count * vox_vol / 1000
                if count > 0 and v > 0:
                    hu_vals = image_arr[mask]
                    if len(hu_vals) > 5000:
                        hu_vals = hu_vals[np.random.choice(len(hu_vals), 5000, replace=False)]
                    hu_agg[v].extend(hu_vals.tolist())

            records.append(row)
        except Exception as e:
            logger.warning(f"Error {case['case_id']}: {e}")

    return pd.DataFrame(records), hu_agg


def compute_hu_stats(hu_agg: dict) -> pd.DataFrame:
    records = []
    for v, vals in hu_agg.items():
        if not vals:
            continue
        arr = np.array(vals)
        records.append({
            "class_val":  v,
            "class_name": CLASS_NAMES.get(v, f"Class {v}"),
            "n_voxels":   len(arr),
            "mean":       float(arr.mean()),
            "std":        float(arr.std()),
            "p5":         float(np.percentile(arr, 5)),
            "p25":        float(np.percentile(arr, 25)),
            "median":     float(np.median(arr)),
            "p75":        float(np.percentile(arr, 75)),
            "p95":        float(np.percentile(arr, 95)),
        })
    return pd.DataFrame(records)


# ─── 5. Acuerdo inter-rater ───────────────────────────────────────────────────

def dice_score(a: np.ndarray, b: np.ndarray, v: int) -> float:
    ma = a == v; mb = b == v
    inter = (ma & mb).sum()
    denom = ma.sum() + mb.sum()
    return 1.0 if denom == 0 else float(2 * inter / denom)


def analyze_interrater(cases: list, label_values: list, max_cases=None) -> pd.DataFrame:
    logger.info("Calculando acuerdo inter-rater...")
    records = []
    subset = [c for c in (cases[:max_cases] if max_cases else cases)
              if c["num_raters"] >= 2]

    for case in tqdm(subset, desc="Inter-rater"):
        try:
            arrs = {rid: load_nifti_nib(rp)[0].astype(np.int32)
                    for rid, rp in case["raters"].items()}
            for r1, r2 in combinations(arrs, 2):
                for v in label_values:
                    if v == 0:
                        continue
                    d = dice_score(arrs[r1], arrs[r2], v)
                    any_pos = bool((arrs[r1] == v).any() or (arrs[r2] == v).any())
                    records.append({
                        "case_id":     case["case_id"],
                        "pair":        f"r{r1} vs r{r2}",
                        "class_val":   v,
                        "class_name":  CLASS_NAMES.get(v, f"Cls{v}"),
                        "dice":        d,
                        "any_positive": any_pos,
                    })
        except Exception as e:
            logger.warning(f"Error {case['case_id']}: {e}")

    return pd.DataFrame(records)


# ─── 6. Incertidumbre ─────────────────────────────────────────────────────────

def analyze_uncertainty(cases: list, max_cases: int = 50) -> pd.DataFrame:
    logger.info("Analizando mapas de incertidumbre...")
    records = []
    for case in tqdm(cases[:max_cases], desc="Uncertainty"):
        if not case["uncertainty"]:
            continue
        try:
            arr, _ = load_nifti_nib(case["uncertainty"])
            nz = arr[arr > 0]
            if len(nz) == 0:
                continue
            records.append({
                "case_id":        case["case_id"],
                "mean_all":       float(arr.mean()),
                "mean_nonzero":   float(nz.mean()),
                "max":            float(arr.max()),
                "std":            float(arr.std()),
                "pct_uncertain":  float((arr > 0.3).mean() * 100),
            })
        except Exception as e:
            logger.warning(f"Error {case['case_id']}: {e}")
    return pd.DataFrame(records)


# ─── 7. Figuras ───────────────────────────────────────────────────────────────

def plot_volume_metadata(df: pd.DataFrame, out_dir: Path):
    fig = plt.figure(figsize=(16, 9))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.38)
    fig.suptitle("Volume Metadata — MBH-Seg25 Training Set", fontsize=13, fontweight="bold")

    ax = fig.add_subplot(gs[0, 0])
    ax.hist(df["shape_z"].dropna(), bins=20, color="#457B9D", edgecolor="white", lw=0.5)
    ax.axvline(df["shape_z"].mean(), color="red", ls="--", lw=1.5,
               label=f"mean={df['shape_z'].mean():.0f}")
    ax.set_xlabel("N Slices (Z)"); ax.set_ylabel("Cases")
    ax.set_title("Axial Slices per Volume"); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    ax.hist(df["shape_x"].dropna(), bins=15, color="#E63946", alpha=0.7, label="X",
            edgecolor="white", lw=0.5)
    ax.hist(df["shape_y"].dropna(), bins=15, color="#2A9D8F", alpha=0.7, label="Y",
            edgecolor="white", lw=0.5)
    ax.set_xlabel("Voxels"); ax.set_title("In-plane Shape (X & Y)"); ax.legend()

    ax = fig.add_subplot(gs[0, 2])
    ax.hist(df["spacing_z"].dropna(), bins=25, color="#F4A261", edgecolor="white", lw=0.5)
    ax.axvline(df["spacing_z"].median(), color="red", ls="--", lw=1.5,
               label=f"median={df['spacing_z'].median():.2f} mm")
    ax.set_xlabel("mm"); ax.set_title("Slice Thickness (Z Spacing)"); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 0])
    ax.hist(df["spacing_x"].dropna(), bins=20, color="#A8DADC", edgecolor="white", lw=0.5)
    ax.axvline(df["spacing_x"].median(), color="red", ls="--", lw=1.5,
               label=f"median={df['spacing_x'].median():.3f} mm")
    ax.set_xlabel("mm"); ax.set_title("In-plane Spacing (X=Y)"); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 1])
    ratio = df["spacing_z"] / df["spacing_x"]
    ax.hist(ratio.dropna(), bins=25, color="#E63946", edgecolor="white", lw=0.5)
    ax.axvline(ratio.median(), color="navy", ls="--", lw=1.5,
               label=f"median={ratio.median():.1f}x")
    ax.set_xlabel("Z / XY spacing"); ax.set_title("Anisotropy Ratio"); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 2])
    rater_counts = df["num_raters"].value_counts().sort_index()
    colors_bar = ["#AAAAAA", "#F4A261", "#2A9D8F", "#457B9D"]
    bars = ax.bar(rater_counts.index.astype(str), rater_counts.values,
                  color=colors_bar[:len(rater_counts)], edgecolor="white", lw=0.7)
    for bar, val in zip(bars, rater_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xlabel("Number of Raters"); ax.set_ylabel("Cases")
    ax.set_title("Rater Coverage per Case")

    out = out_dir / "01_volume_metadata.png"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    logger.info(f"  Guardada: {out}")


def plot_class_distribution(df_cases: pd.DataFrame, label_values: list, out_dir: Path):
    fig = plt.figure(figsize=(16, 9))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.4)
    fig.suptitle("Class Distribution — MBH-Seg25", fontsize=13, fontweight="bold")
    hem = [v for v in label_values if v > 0]

    # Prevalencia
    ax = fig.add_subplot(gs[0, 0])
    names, pcts, colors = [], [], []
    for v in hem:
        col = f"cls{v}_present"
        if col in df_cases.columns:
            names.append(CLASS_NAMES.get(v, f"Cls{v}"))
            pcts.append(df_cases[col].mean() * 100)
            colors.append(CLASS_COLORS.get(v, "#888"))
    if names:
        bars = ax.bar(names, pcts, color=colors, edgecolor="white", lw=0.7)
        for bar, pct in zip(bars, pcts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{pct:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_ylabel("% of Cases"); ax.set_title("Class Prevalence")
    ax.set_ylim(0, 110); ax.tick_params(axis="x", rotation=20)

    # Volumen medio
    ax = fig.add_subplot(gs[0, 1])
    vm_n, vm_m, vm_s, vm_c = [], [], [], []
    for v in hem:
        col = f"cls{v}_vol_cm3"
        if col in df_cases.columns:
            pos = df_cases[df_cases[col] > 0][col]
            if len(pos) >= 3:
                vm_n.append(CLASS_NAMES.get(v, f"Cls{v}"))
                vm_m.append(pos.mean()); vm_s.append(pos.std())
                vm_c.append(CLASS_COLORS.get(v, "#888"))
    if vm_m:
        ax.bar(vm_n, vm_m, yerr=vm_s, color=vm_c, edgecolor="white", lw=0.7, capsize=4)
    ax.set_ylabel("cm³"); ax.set_title("Mean Lesion Volume (positive cases)")
    ax.tick_params(axis="x", rotation=20)

    # Boxplot volumen log
    ax = fig.add_subplot(gs[0, 2])
    bx_d, bx_l, bx_c = [], [], []
    for v in hem:
        col = f"cls{v}_vol_cm3"
        if col in df_cases.columns:
            pos = df_cases[df_cases[col] > 0][col].values
            if len(pos) >= 3:
                bx_d.append(pos); bx_l.append(CLASS_NAMES.get(v, f"Cls{v}"))
                bx_c.append(CLASS_COLORS.get(v, "#888"))
    if bx_d:
        bp = ax.boxplot(bx_d, labels=bx_l, patch_artist=True, widths=0.5,
                        flierprops={"markersize": 3, "alpha": 0.5})
        for patch, color in zip(bp["boxes"], bx_c):
            patch.set_facecolor(color); patch.set_alpha(0.75)
        ax.set_yscale("log")
    ax.set_ylabel("cm³ (log)"); ax.set_title("Volume Distribution")
    ax.tick_params(axis="x", rotation=20)

    # Co-ocurrencia
    ax = fig.add_subplot(gs[1, :2])
    cooc_names = [CLASS_NAMES.get(v, f"Cls{v}") for v in hem]
    n = len(hem)
    cooc = np.zeros((n, n))
    for i, vi in enumerate(hem):
        for j, vj in enumerate(hem):
            ci, cj = f"cls{vi}_present", f"cls{vj}_present"
            if ci in df_cases.columns and cj in df_cases.columns:
                cooc[i, j] = (df_cases[ci] & df_cases[cj]).mean() * 100
    im = ax.imshow(cooc, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(n)); ax.set_xticklabels(cooc_names, rotation=30, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(cooc_names)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cooc[i,j]:.0f}%", ha="center", va="center",
                    fontsize=9, color="white" if cooc[i,j] > 50 else "black")
    plt.colorbar(im, ax=ax, label="% of Cases")
    ax.set_title("Class Co-occurrence (%)")

    # Pie raters
    ax = fig.add_subplot(gs[1, 2])
    rd = df_cases["num_raters"].value_counts().sort_index()
    ax.pie(rd.values,
           labels=[f"{k} rater{'s' if k>1 else ''}" for k in rd.index],
           autopct="%1.1f%%",
           colors=["#AAAAAA","#F4A261","#2A9D8F","#457B9D"][:len(rd)],
           startangle=90)
    ax.set_title("Cases by Rater Count")

    out = out_dir / "02_class_distribution.png"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    logger.info(f"  Guardada: {out}")


def plot_hu_distributions(hu_agg: dict, label_values: list, out_dir: Path):
    hem = [v for v in label_values if v > 0 and len(hu_agg.get(v, [])) > 0]
    if not hem:
        return
    cols = 3
    rows = (len(hem) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4))
    fig.suptitle("HU Value Distributions per Hemorrhage Type", fontsize=13, fontweight="bold")
    axes = np.array(axes).flatten()

    for i, v in enumerate(hem):
        ax = axes[i]
        vals = np.array(hu_agg[v])
        color = CLASS_COLORS.get(v, "#888")
        ax.hist(vals, bins=80, color=color, alpha=0.85, edgecolor="none", density=True)
        ax.axvline(float(np.median(vals)), color="white", lw=2, ls="--",
                   label=f"Median {np.median(vals):.0f} HU")
        ax.axvline(float(vals.mean()), color="yellow", lw=1.5, ls=":",
                   label=f"Mean {vals.mean():.0f} HU")
        ax.axvspan(30, 80, alpha=0.12, color="cyan", label="Blood 30-80 HU")
        ax.set_xlim(-200, 300)
        ax.set_title(f"{CLASS_NAMES.get(v, f'Cls{v}')}  (n={len(vals):,})")
        ax.set_xlabel("HU"); ax.set_ylabel("Density"); ax.legend(fontsize=7)

    for ax in axes[len(hem):]:
        ax.axis("off")

    out = out_dir / "03_hu_distributions.png"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    logger.info(f"  Guardada: {out}")


def plot_interrater(df_ir: pd.DataFrame, out_dir: Path):
    if df_ir.empty:
        logger.warning("Sin datos inter-rater.")
        return
    fig = plt.figure(figsize=(16, 7))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.4)
    fig.suptitle("Inter-Rater Agreement (Dice) — Positive Cases Only",
                 fontsize=13, fontweight="bold")

    df_pos = df_ir[df_ir["any_positive"]].copy()

    ax = fig.add_subplot(gs[0, 0])
    bx_d, bx_l, bx_c = [], [], []
    for v in sorted(df_pos["class_val"].unique()):
        sub = df_pos[df_pos["class_val"] == v]["dice"].values
        if len(sub) >= 2:
            bx_d.append(sub)
            bx_l.append(CLASS_NAMES.get(v, f"Cls{v}"))
            bx_c.append(CLASS_COLORS.get(v, "#888"))
    if bx_d:
        bp = ax.boxplot(bx_d, labels=bx_l, patch_artist=True, widths=0.5,
                        flierprops={"markersize": 3, "alpha": 0.4})
        for patch, color in zip(bp["boxes"], bx_c):
            patch.set_facecolor(color); patch.set_alpha(0.75)
        for i, (data, color) in enumerate(zip(bx_d, bx_c)):
            jitter = np.random.uniform(-0.18, 0.18, len(data))
            ax.scatter(np.ones(len(data))*(i+1)+jitter, data,
                       alpha=0.35, s=12, color=color, zorder=3)
    ax.axhline(0.7, color="green",  ls="--", lw=1.2, label="Good (0.7)")
    ax.axhline(0.5, color="orange", ls="--", lw=1.2, label="Moderate (0.5)")
    ax.set_ylim(-0.05, 1.05); ax.set_ylabel("Dice Score")
    ax.set_title("Pairwise Dice by Class"); ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=20)

    ax = fig.add_subplot(gs[0, 1])
    pivot = df_pos.groupby(["pair", "class_name"])["dice"].mean().unstack(fill_value=np.nan)
    if not pivot.empty:
        sns.heatmap(pivot, ax=ax, cmap="RdYlGn", vmin=0, vmax=1,
                    annot=True, fmt=".2f", linewidths=0.5,
                    cbar_kws={"label": "Mean Dice"})
        ax.set_title("Mean Dice: Rater Pair × Class")
        ax.tick_params(axis="x", rotation=30); ax.tick_params(axis="y", rotation=0)

    out = out_dir / "04_interrater_agreement.png"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    logger.info(f"  Guardada: {out}")


def plot_uncertainty(df_unc: pd.DataFrame, out_dir: Path):
    if df_unc.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Label Uncertainty Map Analysis", fontsize=13, fontweight="bold")

    axes[0].hist(df_unc["mean_nonzero"].dropna(), bins=25, color="#E63946",
                 edgecolor="white", lw=0.5)
    axes[0].set_xlabel("Mean Uncertainty (non-zero)"); axes[0].set_ylabel("Cases")
    axes[0].set_title("Uncertainty Distribution")

    axes[1].hist(df_unc["pct_uncertain"].dropna(), bins=25, color="#457B9D",
                 edgecolor="white", lw=0.5)
    axes[1].set_xlabel("% Voxels with uncertainty > 0.3")
    axes[1].set_title("High-Uncertainty Fraction")

    axes[2].scatter(df_unc["mean_nonzero"], df_unc["pct_uncertain"],
                    alpha=0.6, s=20, color="#2A9D8F")
    axes[2].set_xlabel("Mean Uncertainty"); axes[2].set_ylabel("% High-Unc. Voxels")
    axes[2].set_title("Mean vs Extent of Disagreement")

    out = out_dir / "05_uncertainty.png"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    logger.info(f"  Guardada: {out}")


def plot_sample_slices(cases: list, label_values: list, out_dir: Path, n_cases: int = 4):
    max_val = max(label_values) if label_values else 5
    hem_colors = [CLASS_COLORS.get(v, "#888") for v in range(max_val + 1)]
    hem_colors[0] = "black"
    cmap = matplotlib.colors.ListedColormap(hem_colors)

    n_cols = 5
    fig, axes = plt.subplots(n_cases, n_cols, figsize=(n_cols*3.5, n_cases*3.2))
    if n_cases == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle("Sample Axial Slices — CT + Annotations + STAPLE",
                 fontsize=12, fontweight="bold")

    for row, case in enumerate(cases[:n_cases]):
        try:
            image_arr, _ = load_nifti_nib(case["image"])
            src = case["staple"] if case["has_staple"] else (
                next(iter(case["raters"].values())) if case["raters"] else None)
            if src is None:
                continue
            ref, _ = load_nifti_nib(src)
            ref = ref.astype(np.int32)
            counts = [(ref[:,:,z] > 0).sum() for z in range(ref.shape[2])]
            best_z = int(np.argmax(counts)) if max(counts) > 0 else ref.shape[2]//2

            img_sl = image_arr[:, :, best_z]
            vmin, vmax = np.percentile(img_sl, 2), np.percentile(img_sl, 98)

            # Col 0: CT solo
            axes[row, 0].imshow(img_sl.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
            axes[row, 0].set_title(f"CT (z={best_z})" if row == 0 else f"z={best_z}", fontsize=8)
            axes[row, 0].axis("off")

            # Cols 1-3: raters
            col = 1
            for rid, rp in list(case["raters"].items())[:3]:
                if col >= n_cols - 1:
                    break
                arr, _ = load_nifti_nib(rp)
                sl = arr[:, :, best_z].astype(np.int32)
                axes[row, col].imshow(img_sl.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
                masked = np.ma.masked_where(sl == 0, sl)
                axes[row, col].imshow(masked.T, cmap=cmap, alpha=0.6,
                                      vmin=0, vmax=max_val, origin="lower")
                axes[row, col].set_title(f"Annot {rid}" if row == 0 else "", fontsize=8)
                axes[row, col].axis("off")
                col += 1

            # Última col: STAPLE
            if case["has_staple"]:
                arr, _ = load_nifti_nib(case["staple"])
                arr = arr.astype(np.int32)
                if len(np.unique(arr)) <= max_val + 1:  # hard
                    sl = arr[:, :, best_z]
                    axes[row, n_cols-1].imshow(img_sl.T, cmap="gray", origin="lower",
                                               vmin=vmin, vmax=vmax)
                    masked = np.ma.masked_where(sl == 0, sl)
                    axes[row, n_cols-1].imshow(masked.T, cmap=cmap, alpha=0.6,
                                               vmin=0, vmax=max_val, origin="lower")
                    axes[row, n_cols-1].set_title("STAPLE" if row == 0 else "", fontsize=8)
                else:
                    axes[row, n_cols-1].imshow(arr[:, :, best_z].T, cmap="hot", origin="lower")
                    axes[row, n_cols-1].set_title("STAPLE (soft)" if row == 0 else "", fontsize=8)
                axes[row, n_cols-1].axis("off")

            for c in range(col, n_cols - 1):
                axes[row, c].axis("off")

        except Exception as e:
            logger.warning(f"Error graficando {case['case_id']}: {e}")
            for ax in axes[row]:
                ax.axis("off")

    patches = [mpatches.Patch(color=CLASS_COLORS.get(v, "#888"),
                               label=CLASS_NAMES.get(v, f"Cls{v}"))
               for v in label_values if v > 0]
    fig.legend(handles=patches, loc="lower center", ncol=len(patches),
               bbox_to_anchor=(0.5, -0.02), fontsize=8)

    out = out_dir / "06_sample_slices.png"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    logger.info(f"  Guardada: {out}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_eda(config_path: str = "config.yaml", max_cases=None):
    cfg = load_config(config_path)
    set_seed(cfg["project"]["seed"])

    voxel_dir    = Path(cfg["data"]["voxel_label_dir"])
    rater_prefix = cfg["data"]["rater_prefix"]
    rater_ids    = cfg["data"]["raters_train"]
    output_dir   = Path("experiments/eda")
    figs_dir     = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(exist_ok=True)

    logger.info("=" * 62)
    logger.info("  MBH-Seg25 — Exploratory Data Analysis")
    logger.info("=" * 62)
    logger.info(f"Dataset: {voxel_dir}")
    if max_cases:
        logger.info(f"Modo rápido: {max_cases} casos")

    if not voxel_dir.exists():
        logger.error(f"Directorio no encontrado: {voxel_dir}")
        return

    # Escaneo
    cases = scan_dataset(voxel_dir, rater_prefix, rater_ids)
    if not cases:
        logger.error("No se encontraron casos.")
        return

    # Valores de labels
    n_disc = min(30, len(cases)) if not max_cases else min(max_cases, 30)
    label_info = discover_label_values(cases, max_cases=n_disc)
    label_values = label_info["rater_label_values"]
    with open(output_dir / "label_values.json", "w") as f:
        json.dump(label_info, f, indent=2)

    # Metadata
    df_vol = analyze_volumes(cases, max_cases=max_cases)
    df_vol.to_csv(output_dir / "volume_metadata.csv", index=False)

    logger.info("\n── Shape & Spacing ──")
    for dim in ["shape_z", "shape_y", "shape_x"]:
        s = df_vol[dim]
        logger.info(f"  {dim}: mean={s.mean():.1f}  range=[{s.min():.0f}, {s.max():.0f}]")
    for sp in ["spacing_x", "spacing_y", "spacing_z"]:
        s = df_vol[sp]
        logger.info(f"  {sp}: mean={s.mean():.3f}  range=[{s.min():.3f}, {s.max():.3f}] mm")
    ratio = df_vol["spacing_z"] / df_vol["spacing_x"]
    logger.info(f"  Anisotropy (Z/XY): median={ratio.median():.1f}x  max={ratio.max():.1f}x")

    # Clases
    df_cases, hu_agg = analyze_class_distribution(cases, label_values, max_cases=max_cases)
    df_hu = compute_hu_stats(hu_agg)
    df_cases.to_csv(output_dir / "class_distribution.csv", index=False)
    df_hu.to_csv(output_dir / "hu_statistics.csv", index=False)

    logger.info("\n── Class prevalence ──")
    for v in label_values:
        if v == 0:
            continue
        col = f"cls{v}_present"
        if col in df_cases.columns:
            logger.info(f"  {CLASS_NAMES.get(v,f'Cls{v}'):22s}: "
                        f"{df_cases[col].mean()*100:.1f}%")

    logger.info("\n── HU per class ──")
    for _, row in df_hu.iterrows():
        logger.info(f"  {row['class_name']:22s}: "
                    f"median={row['median']:.1f}  IQR=[{row['p25']:.0f},{row['p75']:.0f}] HU")

    # Inter-rater
    df_ir = analyze_interrater(cases, label_values, max_cases=max_cases)
    df_ir.to_csv(output_dir / "interrater_agreement.csv", index=False)
    if not df_ir.empty:
        logger.info("\n── Inter-rater Dice (positive cases) ──")
        for v in label_values:
            if v == 0:
                continue
            sub = df_ir[(df_ir["class_val"] == v) & df_ir["any_positive"]]
            if len(sub) > 0:
                logger.info(f"  {CLASS_NAMES.get(v,f'Cls{v}'):22s}: "
                            f"mean={sub['dice'].mean():.3f}  "
                            f"median={sub['dice'].median():.3f}")

    # Incertidumbre
    n_unc = min(50, len(cases)) if not max_cases else max_cases
    df_unc = analyze_uncertainty(cases, max_cases=n_unc)
    df_unc.to_csv(output_dir / "uncertainty_stats.csv", index=False)
    if not df_unc.empty:
        logger.info(f"\n── Uncertainty ──")
        logger.info(f"  Mean (non-zero): {df_unc['mean_nonzero'].mean():.3f}")
        logger.info(f"  High-uncertainty cases: {(df_unc['pct_uncertain']>10).sum()}")

    # Figuras
    logger.info("\n── Generando figuras... ──")
    plot_volume_metadata(df_vol, figs_dir)
    plot_class_distribution(df_cases, label_values, figs_dir)
    plot_hu_distributions(hu_agg, label_values, figs_dir)
    plot_interrater(df_ir, figs_dir)
    plot_uncertainty(df_unc, figs_dir)
    plot_sample_slices(cases[:4], label_values, figs_dir)

    # Resumen JSON
    summary = {
        "total_cases":         len(cases),
        "cases_with_staple":   int(sum(1 for c in cases if c["has_staple"])),
        "label_values_found":  label_values,
        "staple_is_prob_map":  label_info["staple_is_prob_map"],
        "mean_shape":          {k: float(df_vol[f"shape_{k}"].mean()) for k in ["x","y","z"]},
        "mean_spacing_mm":     {k: float(df_vol[f"spacing_{k}"].mean()) for k in ["x","y","z"]},
        "anisotropy_ratio":    float(ratio.median()),
        "class_prevalence":    {
            CLASS_NAMES.get(v, f"Cls{v}"): float(
                df_cases.get(f"cls{v}_present", pd.Series([0])).mean()*100)
            for v in label_values if v > 0},
        "interrater_dice":     {
            CLASS_NAMES.get(v, f"Cls{v}"): float(
                df_ir[(df_ir["class_val"]==v)&df_ir["any_positive"]]["dice"].mean())
            if not df_ir.empty else None
            for v in label_values if v > 0},
        "preprocessing_recommendation": {
            "original_spacing_xy": float(df_vol["spacing_x"].median()),
            "original_spacing_z":  float(df_vol["spacing_z"].median()),
            "original_shape_xy":   int(df_vol["shape_x"].median()),
            "original_shape_z":    int(df_vol["shape_z"].median()),
            "note": ("Resample a 1x1x1mm. Shape resultante ~250x250x143. "
                     "Patches de 128x128x128 para entrenamiento."),
        },
    }
    with open(output_dir / "eda_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("\n" + "=" * 62)
    logger.info("  EDA completado ✓")
    logger.info(f"  Resultados en: {output_dir.resolve()}")
    logger.info("=" * 62)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    type=str, default="config.yaml")
    parser.add_argument("--max_cases", type=int, default=None)
    args = parser.parse_args()
    run_eda(args.config, args.max_cases)
