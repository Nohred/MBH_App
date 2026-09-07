import os
import glob
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from monai.inferers import sliding_window_inference
from viola_unet import ViolaUNet
from transforms import get_transforms

# =========================================================================
# 1. CONFIGURACIÓN
# =========================================================================
class VisConfig:
    val_dir    = "/home/ccdatos/Escritorio/ulisescalderon/proyecto_integrador/data/MBH_Val_2025_voxel-label"
    model_path = "./models/final_viola_multiclass_majority_fold0.pth"
    output_img = "inference_grid.png"

    spatial_dims  = 3
    in_channels   = 3
    out_channels  = 6
    patch_size    = (160, 160, 32)
    target_spacing = (1.0, 1.0, 3.0)
    sw_batch_size = 1
    overlap       = 0.5
    num_samples   = 4
    device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Colores para las 5 clases de hemorragia
COLORS = {
    1: [1.0, 0.0, 0.0],   # Rojo   — Clase 1
    2: [0.0, 1.0, 0.0],   # Verde  — Clase 2
    3: [0.0, 0.5, 1.0],   # Azul   — Clase 3
    4: [1.0, 1.0, 0.0],   # Amarillo — Clase 4
    5: [0.0, 1.0, 1.0],   # Cian   — Clase 5
}
CLASS_NAMES = {1: "Clase 1", 2: "Clase 2", 3: "Clase 3", 4: "Clase 4", 5: "Clase 5"}

# =========================================================================
# 2. UTILIDADES DE VISUALIZACIÓN
# =========================================================================
def window_for_plot(img_np, wl=40, ww=80):
    """Ventana visual cerebro/subdural para el plot (asume que ya esta en 0-1, 
    pero como tenemos canales ya normalizados reconstruiremos para el visual)."""
    # Como la entrada img_np (si usamos img_np[0] de la red, es `C0: cerebro`) ya está normalizada en [0,1],
    # usaremos ese canal directamente y no necesitamos hacer windowing de nuevo, pero si lo pasamos a 255 es mas facil.
    return img_np

def apply_overlay(img_gray, mask):
    """Superpone máscara de clases sobre imagen en escala de grises."""
    rgb = np.stack([img_gray, img_gray, img_gray], axis=-1)
    for cls_idx, color in COLORS.items():
        idx = (mask == cls_idx)
        if idx.any():
            rgb[idx] = rgb[idx] * 0.55 + np.array(color) * 0.45
    return rgb

# =========================================================================
# 4. CARGA DEL MODELO
# =========================================================================
print("Cargando modelo...")
model = ViolaUNet(
    spatial_dims=VisConfig.spatial_dims,
    in_channels=VisConfig.in_channels,
    out_channels=VisConfig.out_channels,
    kernel_size=[[3, 3, 1], [3, 3, 1], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
    strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
    upsample_kernel_size=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
    filters=(16, 32, 32, 64, 128),
    dec_filters=(16, 16, 32, 64),
    norm_name=("BATCH", {"affine": True}),
    act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
    dropout=0.2,
    deep_supervision=True,
    deep_supr_num=2,
    res_block=True,
    trans_bias=True,
    viola_att=True,
    gated_att=False,
    sum_deep_supr=False,
).to(VisConfig.device)

state_dict = torch.load(VisConfig.model_path, map_location=VisConfig.device, weights_only=True)
model.load_state_dict(state_dict, strict=False)
model.eval()
print(f"Modelo cargado desde: {VisConfig.model_path}")

# =========================================================================
# 5. BÚSQUEDA DE MUESTRAS E INFERENCIA
# =========================================================================
subject_dirs = glob.glob(os.path.join(VisConfig.val_dir, "*"))
random.shuffle(subject_dirs)

samples_found = []
print("Buscando pacientes con hemorragia visible...")

_, val_transforms = get_transforms()

for subj_path in subject_dirs:
    if len(samples_found) >= VisConfig.num_samples:
        break

    if not os.path.isdir(subj_path):
        continue

    img_path    = os.path.join(subj_path, "image.nii.gz")
    annot_files = sorted(glob.glob(os.path.join(subj_path, "label_annot_*.nii.gz")))

    if not os.path.exists(img_path) or not annot_files:
        continue

    subj_name = os.path.basename(subj_path)

    # Preprocesar imagen y label con el mismo pipeline de MONAI
    data_dict = {"image": img_path, "label": annot_files[0]}
    processed = val_transforms(data_dict)
    
    input_tensor = processed["image"].unsqueeze(0).to(VisConfig.device)
    gt_np = processed["label"].numpy()
    
    # Extraer el canal de cerebro ([0]) preprocesado para la visualización en 0-1
    # Y transponemos los ejes espaciales de (X, Y, Z) a (Z, Y, X) para la visualización axial clásica
    img_np_processed = processed["image"][0].numpy().transpose((2, 1, 0))

    # Buscar el slice axial con más hemorragia (sobre el label resamplado)
    if gt_np.ndim == 4:
        gt_np = gt_np[0]
    gt_np = gt_np.transpose((2, 1, 0))
        
    lesion_per_slice = (gt_np > 0).sum(axis=(1, 2))
    best_z = int(np.argmax(lesion_per_slice))

    if lesion_per_slice[best_z] < 50:
        continue

    print(f"  {subj_name} — slice Z={best_z} ({lesion_per_slice[best_z]} px lesión)")

    # Inferencia en GPU, resultado acumulado en CPU para evitar OOM
    with torch.no_grad():
        with torch.amp.autocast("cuda"):
            outputs = sliding_window_inference(
                inputs=input_tensor,
                roi_size=VisConfig.patch_size,
                sw_batch_size=VisConfig.sw_batch_size,
                predictor=model,
                overlap=VisConfig.overlap,
                sw_device=VisConfig.device,
                device="cpu",
            )

    if isinstance(outputs, torch.Tensor) and outputs.dim() == 6:
        outputs = outputs[:, 0]   # tomar la resolución principal

    # Pasamos argmax de (X, Y, Z) a (Z, Y, X)
    pred_classes = torch.argmax(outputs[0], dim=0).numpy().transpose((2, 1, 0)) 

    # Asegurarse de que gt y pred tengan el mismo número de slices
    n_slices = min(img_np_processed.shape[0], gt_np.shape[0], pred_classes.shape[0])
    if best_z >= n_slices:
        best_z = int(np.argmax((gt_np[:n_slices] > 0).sum(axis=(1, 2))))

    samples_found.append((
        img_np_processed[best_z],   # HU resamplados para ventana visual
        gt_np[best_z],
        pred_classes[best_z],
        subj_name,
        best_z,
    ))

    del outputs, input_tensor
    torch.cuda.empty_cache()

if not samples_found:
    print("No se encontraron pacientes con hemorragia suficiente en val_dir.")
    exit(1)

# =========================================================================
# 6. GENERACIÓN DE LA CUADRÍCULA
# =========================================================================
print(f"\nGenerando cuadrícula con {len(samples_found)} pacientes...")

fig, axes = plt.subplots(len(samples_found), 3, figsize=(15, 5 * len(samples_found)))
if len(samples_found) == 1:
    axes = [axes]

for i, (img_hu, gt_mask, pred_mask, name, z_idx) in enumerate(samples_found):
    img_visual   = window_for_plot(img_hu, wl=40, ww=80)
    gt_colored   = apply_overlay(img_visual, gt_mask)
    pred_colored = apply_overlay(img_visual, pred_mask)

    # Clases presentes en GT y predicción
    gt_classes   = [c for c in range(1, 6) if (gt_mask == c).any()]
    pred_classes_present = [c for c in range(1, 6) if (pred_mask == c).any()]

    axes[i][0].imshow(img_visual, cmap="gray")
    axes[i][0].set_title(f"CT Original\n{name}", fontsize=9)
    axes[i][0].axis("off")

    axes[i][1].imshow(gt_colored)
    axes[i][1].set_title(f"Ground Truth\nClases: {gt_classes}", fontsize=9)
    axes[i][1].axis("off")

    axes[i][2].imshow(pred_colored)
    axes[i][2].set_title(f"Predicción (ViolaUNet)\nClases: {pred_classes_present}", fontsize=9)
    axes[i][2].axis("off")

legend_elements = [
    Patch(facecolor=color, label=CLASS_NAMES[i])
    for i, color in COLORS.items()
]
fig.legend(handles=legend_elements, loc="lower center", ncol=5,
           fontsize=11, bbox_to_anchor=(0.5, 0.01))
plt.suptitle("Inferencia ViolaUNet — Validación", fontsize=13, y=1.01)
plt.tight_layout(rect=[0, 0.05, 1, 1])

output_path = os.path.join(VisConfig.output_img)
plt.savefig(output_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"\n✅ Cuadrícula guardada en: {output_path}")
