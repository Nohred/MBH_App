import os
import glob
import warnings
import torch
import torch.nn.functional as F
import numpy as np
import SimpleITK as sitk
from collections import defaultdict
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDiceMetric
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    ScaleIntensityRangePercentilesd, AsDiscrete, DivisiblePadd
)
from monai.networks.nets import SegResNet, UNet
from monai.inferers import sliding_window_inference
class Config:
    data_dir = "/media/ulises-calderon/SteamDisk/Recuperacion/Escuela/8vo Semestre/ProyectoIntegrador1/data/MBH_Train_2025_voxel-label"
    models_dir = "./models"
    plots_dir = "./plots"
    
    # Parámetros del experimento
    model_type = "segresnet" 
    label_type = "staple"  # Cambiar a "majority" para el otro experimento
    fold = 0
    num_folds = 5
    
    # Hiperparámetros de la red
    spatial_dims = 2
    in_channels = 3       # 3 slices contiguos
    out_channels = 6      # 1 Fondo (0) + 5 Tipos de Hemorragia (1-5)
    
    epochs = 1
    batch_size = 8  # Reducido de 8 a 4 para prevenir OOM (toma en cuenta que num_samples=2)
    learning_rate = 1
    weight_decay = 1e-5   
    val_interval = 5     
    use_cache = False     

    # Parametros de checkpointing
    checkpoint_interval = 50
    multiplicador_ciclo = 2


    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Suprimir warnings de clases vacías en HD95/NSD (esperado en segmentación multiclase)
warnings.filterwarnings("ignore", category=UserWarning, module="monai.metrics")

# =========================================================================
# 1. CONFIGURACIÓN
# =========================================================================
class TestConfig:
    test_dir   = "/media/ulises-calderon/SteamDisk/Recuperacion/Escuela/8vo Semestre/ProyectoIntegrador1/data/MBH_Val_2025_voxel-label"
    model_path = "./models/best_model_segresnet_staple_fold0.pth"

    spatial_dims = 2
    in_channels  = 3
    out_channels = 6

    type = "segresnet"  # "unet" o "segresnet"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================================
# 2. MÉTRICA RVD
# =========================================================================
def calculate_rvd(pred_onehot, gt_onehot):
    dims_to_sum = tuple(range(2, pred_onehot.dim()))
    pred_vol = pred_onehot[:, 1:].float().sum(dim=dims_to_sum)  # excluir fondo
    gt_vol   = gt_onehot[:, 1:].float().sum(dim=dims_to_sum)
    denom    = torch.clamp(gt_vol + pred_vol, min=1.0)          # evita /0 y explosión
    rvd      = torch.abs(pred_vol - gt_vol) / denom
    return rvd.mean().item()


def normalize_for_model(arr):
    """
    Normalización basada en soft-tissue/brain window, idéntica a train.py y transforms.py:
    ScaleIntensityRanged(a_min=-15, a_max=200, b_min=0.0, b_max=1.0, clip=True)
    """
    a_min = -15.0
    a_max = 200.0
    arr = np.clip(arr, a_min, a_max)
    return ((arr - a_min) / (a_max - a_min)).astype(np.float32)

def load_volume_sitk(img_path, lbl_path):
    """Carga imagen y label con SimpleITK → orden correcto [Z, H, W]"""
    img_np = sitk.GetArrayFromImage(sitk.ReadImage(img_path)).astype(np.float32)  # [Z, H, W]
    lbl_np = sitk.GetArrayFromImage(sitk.ReadImage(lbl_path, sitk.sitkUInt8))     # [Z, H, W]
    return img_np, lbl_np

# =========================================================================
# 3. CARGA DEL MODELO
# =========================================================================
def load_model():
    
    if TestConfig.type == "unet":
        model = UNet(
            spatial_dims=TestConfig.spatial_dims, 
            in_channels=TestConfig.in_channels,   
            out_channels=TestConfig.out_channels, 
            channels=(32, 64, 128, 256, 512), 
            strides=(2, 2, 2, 2),             
            num_res_units=2,                  
            dropout=0.2
        ).to(TestConfig.device)
    else:
        model = SegResNet(
            spatial_dims=TestConfig.spatial_dims,
            init_filters=32,
            in_channels=TestConfig.in_channels,
            out_channels=TestConfig.out_channels,
            blocks_down=(1, 2, 2, 4), # 4 downsampling stages para que las hemorragias no desaparezcan
            blocks_up=(1, 1, 1), 
            norm="batch", 
            act="relu" 
        ).to(TestConfig.device)


    ckpt = torch.load(TestConfig.model_path, map_location=TestConfig.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Modelo cargado | Epoch: {ckpt.get('epoch','?')} | Val Dice: {ckpt.get('val_dice','?'):.4f}")
    return model


# =========================================================================
# 4. INFERENCIA DE UN SLICE (3 canales contiguos, igual que en train)
# =========================================================================
def predict_slice(model, img_tensor, z_idx, post_pred):
    """
    img_tensor : [1, Z, H, W] normalizado 0-1 y paddeado
    Retorna pred_onehot [6, H, W] en CPU
    """
    z_dim = img_tensor.shape[0]
    z0 = max(z_idx - 1, 0)
    z2 = min(z_idx + 1, z_dim - 1)

    inp = torch.stack([
        torch.tensor(img_tensor[z0]),
        torch.tensor(img_tensor[z_idx]),
        torch.tensor(img_tensor[z2]),
    ], dim=0).unsqueeze(0).to(TestConfig.device)   # [1, 3, H, W]

    with torch.no_grad():
        out = sliding_window_inference(
            inputs=inp,
            # CORRECTION: Check crop roi_size 384
            roi_size=(384, 384),
            sw_batch_size=4,
            predictor=model,
            overlap=0.5,
        )
    return post_pred(out[0]).cpu()   # [6, H, W]


# =========================================================================
# 5. EVALUACIÓN PRINCIPAL
# =========================================================================
def run_testing():
    print(f"Device: {TestConfig.device}\n")
    model = load_model()

    post_pred  = AsDiscrete(argmax=True, to_onehot=TestConfig.out_channels)
    # post_label = AsDiscrete(to_onehot=TestConfig.out_channels)

    dice_metric = DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=True)
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
    nsd_metric  = SurfaceDiceMetric(include_background=False, class_thresholds=[1.0]*5, reduction="mean")

    eval_tfm = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityRangePercentilesd(
            keys=["image"], lower=1.0, upper=99.0,
            b_min=0.0, b_max=1.0, clip=True
        ),
        DivisiblePadd(keys=["image"], k=16),
    ])

    label_tfm = Compose([
        LoadImaged(keys=["label"]),
        EnsureChannelFirstd(keys=["label"]),
        DivisiblePadd(keys=["label"], k=16),
    ])

    subject_dirs = sorted([
        d for d in glob.glob(os.path.join(TestConfig.test_dir, "*"))
        if os.path.isdir(d)
    ])

    global_metrics        = defaultdict(list)
    results_per_annotator = defaultdict(list)

    print(f"--- EVALUANDO {len(subject_dirs)} PACIENTES (TODOS LOS SLICES) ---\n")

    for subj_path in subject_dirs:
        subj_name = os.path.basename(subj_path)
        img_path  = os.path.join(subj_path, "image.nii.gz")

        if not os.path.exists(img_path):
            continue

        annotator_files = sorted(glob.glob(os.path.join(subj_path, "label_annot_*.nii.gz")))
        if not annotator_files:
            continue

        # Cargar imagen normalizada una sola vez por paciente
        img_np, _ = load_volume_sitk(img_path, annotator_files[0]) # Load image using the first annotator's path just to get structural alignment
        norm_vol = normalize_for_model(img_np)   # [Z, H, W]
        z_dim    = norm_vol.shape[0]

        case_dices, case_hd95s, case_nsds, case_rvds = [], [], [], []

        for annot_file in annotator_files:
            annotator_id = os.path.basename(annot_file).split(".")[0]

            _, gt_np = load_volume_sitk(img_path, annot_file)  # Load the specific ground truth

            slice_dices, slice_hd95s, slice_nsds, slice_rvds = [], [], [], []

            # Estamos acumulando las predicciones 3D y labels para hacer Volumetric Dice como en Train
            pred_vol = torch.zeros((TestConfig.out_channels, norm_vol.shape[1], norm_vol.shape[2], z_dim))
            gt_vol = torch.zeros((1, norm_vol.shape[1], norm_vol.shape[2], z_dim))

            # ── TODOS los slices del volumen ──────────────────────────────
            for z_idx in range(z_dim):

                pred_onehot = predict_slice(model, norm_vol, z_idx, post_pred)
                gt_slice    = torch.tensor(gt_np[z_idx].astype(np.int64))   # [H, W] long
                
                pred_vol[..., z_idx] = pred_onehot
                gt_vol[0, ..., z_idx] = gt_slice
                
                # gt_onehot   = F.one_hot(gt_slice, num_classes=TestConfig.out_channels).permute(2,0,1).float()  # [6, H, W]
                # # gt_onehot = gt_onehot.permute(2, 0, 1).float() 

                # pred_batch = pred_onehot.unsqueeze(0).cpu()              # [1, 6, H, W]
                # gt_batch   = gt_onehot.unsqueeze(0).cpu()                # [1, 6, H, W]

            # AHORA EVALUAMOS DE MANERA VOLUMÉTRICA (COMO EN TRAIN)
            # Primero convertimos el ground truth a one-hot para las métricas
            gt_vol_onehot = F.one_hot(gt_vol.squeeze(0).to(torch.int64), num_classes=TestConfig.out_channels).permute(3, 0, 1, 2).float() # [6, H, W, Z]
            
            pred_batch = pred_vol.unsqueeze(0).cpu()              # [1, 6, H, W, Z]
            gt_batch   = gt_vol_onehot.unsqueeze(0).cpu()         # [1, 6, H, W, Z]

            has_lesion = (gt_batch[:, 1:].sum() > 0) or (pred_batch[:, 1:].sum() > 0)

            if has_lesion:
                # --- Dice (Volumétrico) ---
                dice_metric.reset()
                dice_metric(y_pred=pred_batch, y=gt_batch)
                dice_vals, not_nans = dice_metric.aggregate()
                valid_d = [
                    v.item()
                    for v, n in zip(dice_vals.flatten(), not_nans.flatten())
                    if n > 0
                ]
                if valid_d:
                    slice_dices.append(np.mean(valid_d))

                # --- HD95 ---
                try:
                    hd95_metric.reset()
                    hd95_metric(y_pred=pred_batch, y=gt_batch)
                    val = hd95_metric.aggregate().item()
                    if np.isfinite(val):
                        slice_hd95s.append(val)
                except Exception:
                    pass

                # --- NSD ---
                try:
                    nsd_metric.reset()
                    nsd_metric(y_pred=pred_batch, y=gt_batch)
                    val = nsd_metric.aggregate().item()
                    if np.isfinite(val):
                        slice_nsds.append(val)
                except Exception:
                    pass

                # --- RVD ---
                slice_rvds.append(calculate_rvd(pred_batch, gt_batch))

            # ----- FIN DEL FOR z_idx -----
            
            # Promediar sobre todos los slices informativos para este anotador
            if slice_dices:
                annotator_dice = np.mean(slice_dices)
            else:
                annotator_dice = 0.0  # paciente sin lesión y sin predicción errónea

            case_dices.append(annotator_dice)
            if slice_hd95s: case_hd95s.append(np.mean(slice_hd95s))
            if slice_nsds:  case_nsds.append(np.mean(slice_nsds))
            if slice_rvds:  case_rvds.append(np.mean(slice_rvds))

            results_per_annotator[annotator_id].append(annotator_dice)

        # ----- FIN DEL FOR annot_file -----
        # Promediar sobre los anotadores del paciente
        mean_dsc = np.mean(case_dices)
        global_metrics["DSC"].append(mean_dsc)
        if case_hd95s: global_metrics["HD95"].append(np.mean(case_hd95s))
        if case_nsds:  global_metrics["NSD"].append(np.mean(case_nsds))
        if case_rvds:  global_metrics["RVD"].append(np.mean(case_rvds))

        print(f"  {subj_name} ({len(annotator_files)} anotadores) → DSC: {mean_dsc:.4f}")

    # =========================================================================
    # 6. RESUMEN FINAL
    # =========================================================================
    print("\n=======================================================")
    print(" RESUMEN FINAL DE TESTING")
    print("=======================================================")
    print(f"Total de pacientes evaluados : {len(global_metrics['DSC'])}")
    print(f"\nMétricas Modelo vs Panel de Expertos (Promedio):")
    print(f"  - Dice  (DSC) : {np.mean(global_metrics['DSC']):.4f}   (↑ mejor)")
    print(f"  - NSD         : {np.mean(global_metrics['NSD']):.4f}   (↑ mejor)")
    print(f"  - RVD         : {np.mean(global_metrics['RVD']):.4f}   (↓ mejor, cercano a 0)")
    print(f"  - HD95        : {np.mean(global_metrics['HD95']):.4f} mm  (↓ mejor)")
    print("\nDesglose Dice por Anotador:")
    for annot_id, scores in sorted(results_per_annotator.items()):
        print(f"  {annot_id}: {np.mean(scores):.4f}  (en {len(scores)} casos)")


if __name__ == "__main__":
    run_testing()
