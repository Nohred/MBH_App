import os
import glob
import torch
import numpy as np

from collections import defaultdict

from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDiceMetric
from monai.transforms import AsDiscrete

from viola_unet import ViolaUNet
from transforms import get_transforms



# =========================================================================
# 1. CONFIGURACIÓN DEL TEST
# =========================================================================
class Config:
    data_dir = "/home/ccdatos/Escritorio/ulisescalderon/proyecto_integrador/data/MBH_Val_2025_voxel-label"
    output_dir = "./runs"
    models_dir = "./models"
    plots_dir = "./plots"

    # Experimento
    label_type = "majority"   # 'staple' o "majority"
    fold = 0
    num_folds = 5

    # Datos / entrada
    spatial_dims = 3
    in_channels = 3         # 3 ventanas HU
    out_channels = 6        # fondo + 5 tipos de hemorragia

    # Viola / T6
    patch_size = (160, 160, 32)
    spacing = (1.0, 1.0, 3.0)

    # Entrenamiento
    epochs = 15
    batch_size = 1
    learning_rate = 1e-3
    weight_decay = 1e-5
    val_interval = 5

    # Sistema
    use_cache = True  
    cache_rate = 0.5 #1.0
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class TestConfig:
    # Ruta de validación/test
    test_dir = "/media/ulises-calderon/SteamDisk/Recuperacion/Escuela/8vo Semestre/ProyectoIntegrador1/data/MBH_Val_2025_voxel-label"

    # Ruta al mejor modelo entrenado
    model_path = "./models/best_viola_multiclass_majority_fold0.pth"

    # Parámetros del modelo (deben coincidir con train.py)
    spatial_dims = 3
    in_channels = 3
    out_channels = 6

    # Parámetros de inferencia
    patch_size = (160, 160, 32)
    sw_batch_size = 1
    overlap = 0.5

    # Bandera para ignorar casos/anotadores sanos (sin hemorragia)
    ignore_healthy_cases = True

    # Métrica NSD: umbral por clase (5 clases de hemorragia, sin fondo)
    class_thresholds = [1.0, 1.0, 1.0, 1.0, 1.0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================================
# 2. UTILIDADES
# =========================================================================
def calculate_rvd(pred_onehot, gt_onehot):
    """
    Relative Volume Difference (RVD)
    pred_onehot, gt_onehot: [B, NumClasses, X, Y, Z]
    """
    # IGNORAR LA CLASE 0 (fondo). Solo sumar volumen de las clases 1 a 5.
    pred_lesion = pred_onehot[:, 1:, ...]  # [B, 5, X, Y, Z]
    gt_lesion   = gt_onehot[:, 1:, ...]    # [B, 5, X, Y, Z]
    
    # Sumar todos los píxeles (las dimensiones 2, 3, 4 son X, Y, Z)
    # y sumar también a través de los canales de clase (dim 1)
    pred_vol = pred_lesion.sum(dim=(1, 2, 3, 4)).float()
    gt_vol   = gt_lesion.sum(dim=(1, 2, 3, 4)).float()

    # Prevenir división por cero si el ground truth no tiene ninguna hemorragia
    rvd = torch.abs(pred_vol - gt_vol) / (gt_vol + 1e-5)
    return rvd.mean().item()


def build_model(device):
    model = ViolaUNet(
    spatial_dims=Config.spatial_dims,
    in_channels=Config.in_channels,
    out_channels=Config.out_channels,
    kernel_size=[[3, 3, 1], [3, 3, 1], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
    strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
    upsample_kernel_size=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
    filters=(16, 32, 32, 64, 128),
    dec_filters=(16, 16, 32, 64),
    norm_name=("BATCH", {"affine": True}),
    act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
    dropout=0.2,
    deep_supervision=False, # True
    deep_supr_num=2,
    res_block=True,
    trans_bias=True,
    viola_att=False, # True
    gated_att=False,
    sum_deep_supr=False,
    ).to(Config.device)

    state_dict = torch.load(TestConfig.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

# =========================================================================
# 3. EVALUACIÓN MULTI-ANOTADOR
# =========================================================================
def run_testing():
    print("Iniciando evaluación con ViolaUNet...")
    print(f"Modelo: {TestConfig.model_path}")
    print(f"Device: {TestConfig.device}")

    model = build_model(TestConfig.device)

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hd95_metric = HausdorffDistanceMetric(
        include_background=False,
        percentile=95,
        reduction="mean",
    )
    nsd_metric = SurfaceDiceMetric(
        include_background=False,
        class_thresholds=TestConfig.class_thresholds,
        reduction="mean",
    )

    post_pred = AsDiscrete(argmax=True, to_onehot=TestConfig.out_channels)
    post_label = AsDiscrete(to_onehot=TestConfig.out_channels)
    
    _, val_transforms = get_transforms()

    subject_dirs = sorted(glob.glob(os.path.join(TestConfig.test_dir, "*")))

    global_metrics = defaultdict(list)
    results_per_annotator = defaultdict(list)

    print("\n--- EVALUANDO PACIENTE POR PACIENTE ---")

    with torch.no_grad():
        for subj_path in subject_dirs:
            if not os.path.isdir(subj_path):
                continue

            subj_name = os.path.basename(subj_path)
            img_path = os.path.join(subj_path, "image.nii.gz")

            if not os.path.exists(img_path):
                print(f"[WARN] {subj_name}: no existe image.nii.gz")
                continue

            annotator_files = sorted(glob.glob(os.path.join(subj_path, "label_annot_*.nii.gz")))
            if not annotator_files:
                print(f"[WARN] {subj_name}: no se encontraron anotadores")
                continue

            # A. Imagen y anotador base para la transformación
            # Usamos el primer anotador solo para que transforms.py pueda ejecutar el pipeline completo
            base_dict = {"image": img_path, "label": annotator_files[0]}
            processed_base = val_transforms(base_dict)
            
            input_tensor = processed_base["image"].unsqueeze(0).to(TestConfig.device)

            # B. Inferencia 3D por sliding window
            outputs = sliding_window_inference(
                inputs=input_tensor,
                roi_size=TestConfig.patch_size,
                sw_batch_size=TestConfig.sw_batch_size,
                predictor=model,
                overlap=TestConfig.overlap,
                sw_device=TestConfig.device, 
                device="cpu",                
            )

            pred_onehot = post_pred(outputs[0])  
            pred_batch = pred_onehot.unsqueeze(0)  

            case_dices, case_hd95s, case_nsds, case_rvds = [], [], [], []

            # C. Comparación contra cada anotador
            for annot_file in annotator_files:
                annotator_id = os.path.basename(annot_file).split(".")[0]

                # Ejecutamos el pipeline de nuevo solo para obtener el GT ya preprocesado 
                # (crop_foreground y spacing operan igual porque la 'image' es la misma en todos)
                annot_dict = {"image": img_path, "label": annot_file}
                processed_annot = val_transforms(annot_dict)
                
                annot_tensor = processed_annot["label"].unsqueeze(0)

                # Ignorar si es un caso sano (sin hemorragia en el label de este anotador)
                if TestConfig.ignore_healthy_cases and annot_tensor.max() == 0:
                    print(f"    -> [INFO] Anotador {annotator_id} ignorado (GT vacío/sano)")
                    continue

                gt_onehot    = post_label(annot_tensor[0])   
                gt_batch     = gt_onehot.unsqueeze(0).cpu()
                # Dice
                dice_metric.reset()
                dice_metric(y_pred=pred_batch, y=gt_batch)
                dice_val = dice_metric.aggregate().item()

                # HD95
                try:
                    hd95_metric.reset()
                    hd95_metric(y_pred=pred_batch, y=gt_batch)
                    hd95_val = hd95_metric.aggregate().item()
                except Exception:
                    hd95_val = float("nan")

                # NSD
                try:
                    nsd_metric.reset()
                    nsd_metric(y_pred=pred_batch, y=gt_batch)
                    nsd_val = nsd_metric.aggregate().item()
                except Exception:
                    nsd_val = float("nan")

                # RVD
                rvd_val = calculate_rvd(pred_batch, gt_batch)

                case_dices.append(dice_val)
                case_hd95s.append(hd95_val)
                case_nsds.append(nsd_val)
                case_rvds.append(rvd_val)
                

                results_per_annotator[annotator_id].append(dice_val)

            # D. Promedio por paciente
            if len(case_dices) == 0:
                print(f"{subj_name} ignorado completamente (todos los anotadores están limpios)")
            else:
                global_metrics["DSC"].append(np.mean(case_dices))
                global_metrics["RVD"].append(np.mean(case_rvds))

                clean_hd95 = [v for v in case_hd95s if not np.isnan(v)]
                clean_nsd = [v for v in case_nsds if not np.isnan(v)]

                if clean_hd95:
                    global_metrics["HD95"].append(np.mean(clean_hd95))
                if clean_nsd:
                    global_metrics["NSD"].append(np.mean(clean_nsd))

                print(
                    f"{subj_name} ({len(case_dices)}/{len(annotator_files)} anotadores evaluados) -> "
                    f"DSC: {np.mean(case_dices):.4f} | "
                    f"RVD: {np.mean(case_rvds):.4f}"
                )

            # E. === LIMPIEZA OBLIGATORIA DE MEMORIA GPU ===
            # Borrar variables masivas de la memoria
            del input_tensor, outputs, pred_onehot, pred_batch
            
            # Borrar variables del bucle interno si existen
            
            if 'annot_tensor' in locals():
                try:
                    del annot_tensor
                except:
                    pass
            if 'gt_onehot' in locals():
                try:                    
                    del gt_onehot
                except:
                    pass
            if 'gt_batch' in locals():
                try:
                    del gt_batch
                except:
                    pass
                
            # Vaciar el caché de CUDA
            torch.cuda.empty_cache()

    # =========================================================================
    # 4. RESUMEN FINAL
    # =========================================================================
    print("\n=======================================================")
    print("RESUMEN FINAL DE TESTING")
    print("=======================================================")
    print(f"Total de pacientes evaluados: {len(global_metrics['DSC'])}")

    if global_metrics["DSC"]:
        print(f" - Dice (DSC): {np.mean(global_metrics['DSC']):.4f}")
    if global_metrics["NSD"]:
        print(f" - NSD       : {np.mean(global_metrics['NSD']):.4f}")
    if global_metrics["RVD"]:
        print(f" - RVD       : {np.mean(global_metrics['RVD']):.4f}")
    if global_metrics["HD95"]:
        print(f" - HD95      : {np.mean(global_metrics['HD95']):.4f} mm")

    print("\nDesglose de Dice por anotador:")
    for annot_id, scores in sorted(results_per_annotator.items()):
        print(f" - {annot_id}: Dice promedio = {np.mean(scores):.4f} (n={len(scores)})")


if __name__ == "__main__":
    run_testing()