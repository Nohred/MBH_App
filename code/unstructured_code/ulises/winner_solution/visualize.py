"""
visualize.py  —  Adaptado de visualize_inference.py para la UNet 2D (nvauto approach)
Genera una cuadrícula inference_grid.png con N pacientes que tengan hemorragia visible.
Cada fila: CT (soft tissue window) | Ground Truth overlay | Predicción overlay
"""

import os
import glob
import random
import torch
import numpy as np
import SimpleITK as sitk
import matplotlib
from triton import Config
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from monai.networks.nets import UNet, SegResNet
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete

# =========================================================================
# 1. CONFIGURACIÓN
# =========================================================================
class VisConfig:
    val_dir    = "/media/ulises-calderon/SteamDisk/Recuperacion/Escuela/8vo Semestre/ProyectoIntegrador1/data/MBH_Val_2025_voxel-label"
    # model_path = "./models/best_model_majority_fold0.pth"
    model_path = "./models/best_model_segresnet_staple_fold0.pth"

    
    output_img = "inference_grid.png"

    # Arquitectura idéntica a train.py
    spatial_dims = 2
    in_channels  = 3
    out_channels = 6

    # Inferencia sliding window (igual que evaluate.py)
    # CORRECTION: Check crop roi_size 384
    roi_size      = (384, 384)
    sw_batch_size = 4
    overlap       = 0.5

    # Cuántas filas quieres en la cuadrícula (pacientes con hemorragia)
    num_samples = 6

    # Window Level / Width para visualización soft tissue
    WL = 40    # center HU  (subdural/cerebro)
    WW = 120   # ancho HU

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================================
# 2. COLORES POR CLASE
# =========================================================================
COLORS = {
    1: [1.0, 0.15, 0.15],   # Rojo      — Epidural
    2: [0.15, 1.0, 0.15],   # Verde     — Subdural
    3: [0.15, 0.45, 1.0],   # Azul      — Subarachnoid
    4: [1.0,  1.0,  0.0 ],  # Amarillo  — Intraventricular
    5: [0.0,  1.0,  1.0 ],  # Cian      — Intracerebral
}
CLASS_NAMES = {
    1: "Epidural",
    2: "Subdural",
    3: "Subarachnoid",
    4: "Intraventricular",
    5: "Intracerebral",
}

# =========================================================================
# 3. FUNCIONES AUXILIARES
# =========================================================================
def window_image(img_np, wl, ww):
    """Aplica Window Level / Width sobre HU raw y devuelve imagen normalizada [0,1]."""
    lower = wl - ww / 2.0
    upper = wl + ww / 2.0
    clipped = np.clip(img_np, lower, upper)
    return (clipped - lower) / (upper - lower)

def normalize_for_model(img_np):
    """
    Normalización basada en soft-tissue/brain window, idéntica a train.py y transforms.py:
    ScaleIntensityRanged(a_min=-15, a_max=200, b_min=0.0, b_max=1.0, clip=True)
    """
    a_min = -15.0
    a_max = 200.0
    img_np = np.clip(img_np, a_min, a_max)
    return ((img_np - a_min) / (a_max - a_min)).astype(np.float32)

def apply_color_mask(img_vis, mask):
    """
    Superpone máscara de colores sobre imagen 2D [0,1] en escala de grises.
    40 % color de clase + 60 % CT original (igual que visualize_inference.py).
    """
    colored = np.stack([img_vis, img_vis, img_vis], axis=-1)
    for cls_id, color in COLORS.items():
        px = (mask == cls_id)
        if px.any():
            colored[px] = colored[px] * 0.6 + np.array(color) * 0.4
    return colored

def build_input_tensor(norm_vol, z_idx):
    """
    Toma el volumen normalizado [Z, H, W] y construye el tensor de entrada 2D
    con 3 canales contiguos (z-1, z, z+1), igual que Extract3Slice2Dd en transforms.py.
    Devuelve tensor [1, 3, H, W].
    """
    z_dim = norm_vol.shape[0]
    z0 = max(z_idx - 1, 0)
    z2 = min(z_idx + 1, z_dim - 1)
    s0 = torch.tensor(norm_vol[z0])
    s1 = torch.tensor(norm_vol[z_idx])
    s2 = torch.tensor(norm_vol[z2])
    return torch.stack([s0, s1, s2], dim=0).unsqueeze(0)   # [1, 3, H, W]

# =========================================================================
# 4. CARGA DEL MODELO
# =========================================================================
print("Cargando modelo...")

# model = UNet(
#         spatial_dims=VisConfig.spatial_dims, 
#         in_channels=VisConfig.in_channels,   
#         out_channels=VisConfig.out_channels, 
#         channels=(32, 64, 128, 256, 512), 
#         strides=(2, 2, 2, 2),             
#         num_res_units=2,                  
#         dropout=0.2
#     ).to(VisConfig.device)

model = SegResNet(
    spatial_dims=VisConfig.spatial_dims,
    init_filters=32,
    in_channels=VisConfig.in_channels,
    out_channels=VisConfig.out_channels,
    blocks_down=(1, 2, 2, 4),
    blocks_up=(1, 1, 1),
    norm="batch",
    act="relu"
).to(VisConfig.device)


ckpt = torch.load(VisConfig.model_path, map_location=VisConfig.device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"  Epoch {ckpt.get('epoch','?')}  |  Val Dice: {ckpt.get('val_dice', 0):.4f}")

post_pred = AsDiscrete(argmax=True, to_onehot=VisConfig.out_channels)

# =========================================================================
# 5. BUSCAR PACIENTES CON HEMORRAGIA VISIBLE
# =========================================================================
subject_dirs = sorted(glob.glob(os.path.join(VisConfig.val_dir, "*")))
subject_dirs = [d for d in subject_dirs if os.path.isdir(d)]
random.shuffle(subject_dirs)

samples_found = []
print(f"\nBuscando {VisConfig.num_samples} pacientes con hemorragia...")

for subj_path in subject_dirs:
    if len(samples_found) >= VisConfig.num_samples:
        break

    img_path = os.path.join(subj_path, "image.nii.gz")

    # Preferir label_consensus_majority; si no, tomar cualquier anotador
    gt_path = os.path.join(subj_path, "label_consensus_majority.nii.gz")
    if not os.path.exists(gt_path):
        annot_files = sorted(glob.glob(os.path.join(subj_path, "label_annot_*.nii.gz")))
        if annot_files:
            gt_path = annot_files[0]
        else:
            continue

    if not os.path.exists(img_path):
        continue

    # Cargar GT y encontrar el slice con más hemorragia
    gt_sitk = sitk.ReadImage(gt_path, sitk.sitkUInt8)
    gt_np   = sitk.GetArrayFromImage(gt_sitk)   # [Z, H, W] (sitk invierte ejes)

    lesion_per_slice = (gt_np > 0).sum(axis=(1, 2))
    best_z = int(np.argmax(lesion_per_slice))

    if lesion_per_slice[best_z] < 50:   # lesión demasiado pequeña → skip
        continue

    subj_name = os.path.basename(subj_path)
    print(f"  [{len(samples_found)+1}] {subj_name}  (slice Z={best_z}, "
          f"pxl lesión={lesion_per_slice[best_z]})")

    # Cargar imagen raw
    img_sitk = sitk.ReadImage(img_path)
    raw_np   = sitk.GetArrayFromImage(img_sitk).astype(np.float32)  # [Z, H, W]

    # Normalizar para el modelo (percentiles, igual que transforms.py)
    norm_vol = normalize_for_model(raw_np)

    # Construir tensor de entrada 2D [1, 3, H, W]
    input_tensor = build_input_tensor(norm_vol, best_z).to(VisConfig.device)

    # Inferencia
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=VisConfig.device.type == "cuda"):
            outputs = sliding_window_inference(
                inputs=input_tensor,
                roi_size=VisConfig.roi_size,
                sw_batch_size=VisConfig.sw_batch_size,
                predictor=model,
                overlap=VisConfig.overlap,
            )

    pred_mask = torch.argmax(outputs[0], dim=0).cpu().numpy()   # [H, W]

    # Slice 2D para visualización
    img_2d_raw = raw_np[best_z]
    gt_2d      = gt_np[best_z]

    samples_found.append({
        "img_raw":   img_2d_raw,
        "gt":        gt_2d,
        "pred":      pred_mask,
        "name":      subj_name,
        "z":         best_z,
    })

if not samples_found:
    print("No se encontraron pacientes con hemorragia suficiente.")
    exit(1)

# =========================================================================
# 6. GENERAR CUADRÍCULA
# =========================================================================
n = len(samples_found)
print(f"\nGenerando cuadrícula ({n} filas × 3 columnas)...")

fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n))
if n == 1:
    axes = [axes]

for i, s in enumerate(samples_found):
    # Windowing para visualización (soft tissue: WL=40 WW=120)
    img_vis = window_image(s["img_raw"], VisConfig.WL, VisConfig.WW)

    gt_overlay   = apply_color_mask(img_vis, s["gt"])
    pred_overlay = apply_color_mask(img_vis, s["pred"])

    # Panel 1: CT solo
    axes[i][0].imshow(img_vis, cmap="gray", vmin=0, vmax=1)
    axes[i][0].set_title(f"CT  —  {s['name']}\nSlice Z={s['z']}  |  WL={VisConfig.WL} WW={VisConfig.WW}")
    axes[i][0].axis("off")

    # Panel 2: Ground Truth
    axes[i][1].imshow(gt_overlay)
    axes[i][1].set_title("Ground Truth")
    axes[i][1].axis("off")

    # Panel 3: Predicción
    axes[i][2].imshow(pred_overlay)
    axes[i][2].set_title("Predicción (SegResNetDS 2D)")
    axes[i][2].axis("off")

# Leyenda global de clases
legend_handles = [
    Patch(facecolor=color, label=CLASS_NAMES[cls_id])
    for cls_id, color in COLORS.items()
]
fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=5,
    fontsize=11,
    bbox_to_anchor=(0.5, 0.01),
)
plt.tight_layout(rect=[0, 0.04, 1, 1])

plt.savefig(VisConfig.output_img, dpi=200, bbox_inches="tight")
print(f"\n✓ Listo → {VisConfig.output_img}")