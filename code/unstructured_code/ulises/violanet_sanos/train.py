import os
import torch
import matplotlib.pyplot as plt
from dataset import get_dataloaders
from transforms import get_transforms
from viola_unet import ViolaUNet
from monai.losses import DiceCELoss, DiceFocalLoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.data import decollate_batch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from monai.inferers import sliding_window_inference
import time
from collections import defaultdict
import numpy as np


# =========================================================================
# PARTE 1: CONFIGURACIÓN E HIPERPARÁMETROS
# =========================================================================
class Config:
    data_dir = "/home/ccdatos/Escritorio/ulisescalderon/proyecto_integrador/data/MBH_Train_2025_voxel-label"
    models_dir = "./models"
    plots_dir = "./plots"

    # Experimento
    label_type = "staple"   # 'staple' o "majority"
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
    epochs = 100
    batch_size = 1
    learning_rate = 2e-4
    weight_decay = 1e-5
    val_interval = 10

    # Sistema
    use_cache = True  
    cache_rate = 1.0 
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



os.makedirs(Config.models_dir, exist_ok=True)
os.makedirs(Config.plots_dir, exist_ok=True)

print(f"Iniciando experimento Viola multiclase | Label: {Config.label_type} | Fold: {Config.fold}")
print(f"Device: {Config.device}")


# =========================================================================
# PARTE 2: CARGA DE DATOS
# =========================================================================
train_transforms, val_transforms = get_transforms()

train_loader, val_loader = get_dataloaders(
    data_dir=Config.data_dir,
    label_type=Config.label_type,
    fold=Config.fold,
    num_folds=Config.num_folds,
    batch_size=Config.batch_size,
    train_transforms=train_transforms,
    val_transforms=val_transforms,
    use_cache=Config.use_cache,
    cache_rate=Config.cache_rate,
    num_workers=Config.num_workers,
)

print("DataLoaders creados correctamente.")
print(f"Batch size train: {Config.batch_size}")
print(f"Batch size val: {Config.batch_size}")


# =========================================================================
# PARTE 3: DEFINICIÓN DEL MODELO
# =========================================================================
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
    deep_supervision=True, # True
    deep_supr_num=2,
    res_block=True,
    trans_bias=True,
    viola_att=True, # True
    gated_att=False,
    sum_deep_supr=False,
).to(Config.device)





print("Modelo cargado correctamente.")
print(f"Input channels: {Config.in_channels}")
print(f"Output classes: {Config.out_channels}")
print(f"Patch size esperado: {Config.patch_size}")
print(f"Spacing esperado: {Config.spacing}")




# =========================================================================
# PARTE 4: FUNCIÓN DE PÉRDIDA, OPTIMIZADOR, SCHEDULER Y VALIDACIÓN
# =========================================================================
# Calcular class_weights del dataset


class_counts = defaultdict(int)
for batch_data in train_loader:
    labels = batch_data["label"].numpy().astype(int)
    unique, counts = np.unique(labels, return_counts=True)
    for cls, cnt in zip(unique, counts):
        class_counts[cls] += cnt

class_counts_arr = np.array([class_counts.get(i, 1) for i in range(Config.out_channels)])
total = class_counts_arr.sum()

log_weights = np.log1p(total / (Config.out_channels * class_counts_arr))
log_weights = np.clip(log_weights, 0.5, 3.0)

class_weights = torch.tensor(log_weights, dtype=torch.float32).to(Config.device)
print(f"Class weights (log inverso): {class_weights.tolist()}")
print(f"Class weights calculated: {class_weights}")


# loss_function = DiceCELoss(
#     to_onehot_y=True,
#     softmax=True,
#     include_background=False,
#     weight=class_weights,  # Castigamos más si falla en las hemorragias
#     squared_pred=True         # Ayuda con el Dice cuando los targets son muy pequeños
# )

loss_function = DiceFocalLoss(
    to_onehot_y=True,
    softmax=True,
    include_background=False,
    lambda_dice=0.5,
    lambda_focal=0.5,
    gamma=2.0,
    weight=class_weights[1:]
)

optimizer = AdamW(
    model.parameters(),
    lr=Config.learning_rate,
    weight_decay=Config.weight_decay,
)

# Configuración recomendada para 100 épocas (escalable a más)
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=100,        # El LR bajará y hará su primer "reinicio" en la época 50
    T_mult=2,      # El siguiente ciclo durará otras 50 épocas (hasta la 100)
    eta_min=1e-6   # El valor mínimo al que caerá el LR antes de rebotar
)

dice_metric = DiceMetric(include_background=False,reduction="mean")
post_pred = AsDiscrete(argmax=True, to_onehot=Config.out_channels)
post_label = AsDiscrete(to_onehot=Config.out_channels)


def validate_epoch(model, val_loader, device):
    model.eval()
    dice_metric.reset()
    val_loss = 0.0
    val_step = 0

    with torch.no_grad():
        for batch_data in val_loader:
            val_step += 1

            inputs = batch_data["image"].to(device)
            # labels = batch_data["label"].to(device)
            labels = batch_data["label"].long().to(Config.device)

            # outputs = model(inputs)

            outputs = sliding_window_inference(
                inputs=inputs,
                roi_size=Config.patch_size,
                sw_batch_size=1,
                predictor=model,
                overlap=0.25,
            )

            loss = loss_function(outputs, labels)
            val_loss += loss.item()

            outputs_list = [post_pred(x) for x in decollate_batch(outputs)]
            labels_list = [post_label(x) for x in decollate_batch(labels)]

            dice_metric(y_pred=outputs_list, y=labels_list)

    mean_val_loss = val_loss / val_step
    mean_val_dice = dice_metric.aggregate().item()
    dice_metric.reset()

    return mean_val_loss, mean_val_dice


history = {
    "train_loss": [],
    "val_loss": [],
    "train_dice": [],
    "val_dice": [],
    "learning_rate": [],
}



best_val_dice = 0.0
best_model_path = os.path.join(
    Config.models_dir,
    f"best_viola_multiclass_{Config.label_type}_fold{Config.fold}.pth"
)

# =========================================================================
# PARTE 5: LOOP DE ENTRENAMIENTO Y VALIDACIÓN
# =========================================================================
# =========================================================================
# PARTE 5: LOOP DE ENTRENAMIENTO Y VALIDACIÓN
# =========================================================================
print("Comenzando entrenamiento...")
start_time = time.time()

accumulation_steps = 4  # Ajusta este valor si quieres simular un batch_size mayor (ej. 4 u 8)
scaler = torch.amp.GradScaler("cuda")
for epoch in range(Config.epochs):
    model.train()
    dice_metric.reset()

    epoch_loss = 0.0
    step = 0
    optimizer.zero_grad() # Limpieza inicial por época

    for batch_data in train_loader:
        step += 1

        inputs = batch_data["image"].to(Config.device)
        labels = batch_data["label"].long().to(Config.device)

        with torch.amp.autocast("cuda"):
            raw_output = model(inputs)

            # Manejo robusto: lista de tensores (MONAI) o tensor 6D
            if isinstance(raw_output, (list, tuple)):
                outputs_unbind = list(raw_output)
            elif isinstance(raw_output, torch.Tensor) and raw_output.dim() == 6:
                outputs_unbind = list(torch.unbind(raw_output, dim=1))
            else:
                outputs_unbind = None  # deep_supervision desactivado

            if outputs_unbind is not None:
                weights = [1.0 / (2**i) for i in range(len(outputs_unbind))]
                weights = [w / sum(weights) for w in weights]
                loss = sum(w * loss_function(out, labels) for w, out in zip(weights, outputs_unbind))
                outputs_for_metric = outputs_unbind[0]
            else:
                loss = loss_function(raw_output, labels)
                outputs_for_metric = raw_output

            loss = loss / accumulation_steps

        # Backward con Scaler
        scaler.scale(loss).backward()
        epoch_loss += loss.item() * accumulation_steps # Recuperar valor real para logging

        # Usar outputs_for_metric en lugar de outputs para el cálculo del Dice
        outputs_list = [post_pred(x) for x in decollate_batch(outputs_for_metric)]
        labels_list = [post_label(x) for x in decollate_batch(labels)]
        dice_metric(y_pred=outputs_list, y=labels_list)

        # Paso de optimización simulando batch mayor
        if step % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad() # Limpiamos los gradientes tras el paso

    # Asegurar que se haga el paso si el loader no es múltiplo de accumulation_steps
    if step % accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    epoch_loss /= step
    epoch_train_dice = dice_metric.aggregate().item()
    dice_metric.reset()

    history["train_loss"].append(epoch_loss)
    history["train_dice"].append(epoch_train_dice)
    history["learning_rate"].append(scheduler.get_last_lr()[0])

    print(
        f"Época {epoch + 1}/{Config.epochs} | "
        f"Train Loss: {epoch_loss:.4f} | "
        f"Train Dice: {epoch_train_dice:.4f} | "
        f"LR: {history['learning_rate'][-1]:.6e}"
    )

    if (epoch + 1) % Config.val_interval == 0:
        val_loss, val_mean_dice = validate_epoch(model, val_loader, Config.device)

        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_mean_dice)

        print(
            f"   => Validación | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_mean_dice:.4f}"
        )

        if val_mean_dice > best_val_dice:
            best_val_dice = val_mean_dice
            torch.save(model.state_dict(), best_model_path)
            print(f"   => Nuevo mejor modelo guardado en: {best_model_path}")

    scheduler.step(epoch + 1)

end_time = time.time()
total_time = end_time - start_time
hours, rem = divmod(total_time, 3600)
minutes, seconds = divmod(rem, 60)
#Save last model checkpoint
final_model_path = os.path.join(
    Config.models_dir,
    f"final_viola_multiclass_{Config.label_type}_fold{Config.fold}.pth"
)
torch.save(model.state_dict(), final_model_path)
print(f"Último modelo guardado en: {final_model_path}")

print(f"\nEntrenamiento finalizado en {int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s.")
print(f"\nEntrenamiento finalizado. Mejor Dice de validación: {best_val_dice:.4f}")


# =========================================================================
# PARTE 6: GENERACIÓN Y GUARDADO DE CURVAS
# =========================================================================
print("\nGenerando gráficas de entrenamiento...")

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

train_epochs = list(range(1, Config.epochs + 1))
val_epochs = [i * Config.val_interval for i in range(1, len(history["val_loss"]) + 1)]

# 1. Loss: Train vs Validation
axes[0].plot(
    train_epochs,
    history["train_loss"],
    label="Train Loss",
    color="red",
    linewidth=2,
)
axes[0].plot(
    val_epochs,
    history["val_loss"],
    label="Val Loss",
    color="orange",
    marker="o",
    linestyle="--",
    linewidth=2,
)
axes[0].set_title("Loss: Train vs Validation", fontsize=14)
axes[0].set_xlabel("Epoch", fontsize=12)
axes[0].set_ylabel("Loss", fontsize=12)
axes[0].grid(True, linestyle="--", alpha=0.7)
axes[0].legend()

# 2. Dice: Train vs Validation
axes[1].plot(
    train_epochs,
    history["train_dice"],
    label="Train Dice",
    color="blue",
    linewidth=2,
)
axes[1].plot(
    val_epochs,
    history["val_dice"],
    label="Val Dice",
    color="cyan",
    marker="o",
    linestyle="--",
    linewidth=2,
)
axes[1].set_title("Dice: Train vs Validation", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=12)
axes[1].set_ylabel("Dice", fontsize=12)
axes[1].grid(True, linestyle="--", alpha=0.7)
axes[1].legend()

# 3. Learning Rate
axes[2].plot(
    train_epochs,
    history["learning_rate"],
    label="Learning Rate",
    color="green",
    linewidth=2,
)
axes[2].set_title("Learning Rate Schedule", fontsize=14)
axes[2].set_xlabel("Epoch", fontsize=12)
axes[2].set_ylabel("LR", fontsize=12)
axes[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
axes[2].grid(True, linestyle="--", alpha=0.7)
axes[2].legend()

plt.tight_layout()

plot_filename = os.path.join(
    Config.plots_dir,
    f"learning_curves_viola_multiclass_{Config.label_type}_fold{Config.fold}.png"
)
plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
plt.close()

print(f"Gráficas guardadas en: {plot_filename}")