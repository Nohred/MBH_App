import argparse
import os
import torch
import time
import math
import gc
import numpy as np
from monai.networks.nets import SegResNet, UNet

from monai.losses import DiceCELoss, DiceFocalLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from dataset import get_dataloaders
from transforms import get_transforms
import matplotlib.pyplot as plt
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference


# get cuda device and clear cache before starting

cuda_available = torch.cuda.is_available()
print(f"CUDA disponible: {cuda_available}")

if cuda_available:
    print(f"CUDA version de PyTorch: {torch.version.cuda}")
    print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
else:
    print(f"PyTorch fue compilado con CUDA: {torch.version.cuda}")
    print("PyTorch no ve ninguna GPU en este entorno; el entrenamiento seguirá en CPU si el resto del script lo permite.")

gc.collect()
if cuda_available:
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    print(torch.cuda.memory_summary())
else:
    print("Saltando limpieza/memory summary de CUDA porque no hay GPU disponible.")

# Dead code validate_epoch removed

# =========================================================================
# PARSER DE ARGUMENTOS
# =========================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="segresnet", choices=["unet", "segresnet"])
parser.add_argument("--epochs", type=int, default=200) # Modificado a 200 epochs (~4 horas)
parser.add_argument("--lr", type=float, default=2e-4)
args = parser.parse_args()

# =========================================================================
# PARTE 1: CONFIGURACIÓN E HIPERPARÁMETROS
# =========================================================================
class Config:
    data_dir = "/media/ulises-calderon/SteamDisk/Recuperacion/Escuela/8vo Semestre/ProyectoIntegrador1/data/MBH_Train_2025_voxel-label"
    models_dir = "./models"
    plots_dir = "./plots"
    
    # Parámetros del experimento
    model_type = args.model
    label_type = "staple"  # Cambiar a "majority" para el otro experimento
    fold = 0
    num_folds = 5
    
    # Hiperparámetros de la red
    spatial_dims = 2
    in_channels = 3       # 3 slices contiguos
    out_channels = 6      # 1 Fondo (0) + 5 Tipos de Hemorragia (1-5)
    
    epochs = args.epochs 
    batch_size = 8  # Reducido de 8 a 4 para prevenir OOM (toma en cuenta que num_samples=2)
    learning_rate = args.lr
    weight_decay = 1e-5   
    val_interval = 5     
    use_cache = False     

    # Parametros de checkpointing
    checkpoint_interval = 50
    multiplicador_ciclo = 2


    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Crear carpetas de salida si no existen
os.makedirs(Config.models_dir, exist_ok=True)
os.makedirs(Config.plots_dir, exist_ok=True)

# =========================================================================
# PARTE 2: CARGA DE DATOS
# =========================================================================
print(f"Iniciando experimento con label: {Config.label_type} | Fold: {Config.fold}, epocas: {Config.epochs}")

train_transforms, val_transforms = get_transforms()

train_loader, val_loader = get_dataloaders(
    data_dir=Config.data_dir,
    label_type=Config.label_type,
    fold=Config.fold,
    num_folds=Config.num_folds,
    batch_size=Config.batch_size,
    train_transforms=train_transforms,
    val_transforms=val_transforms,
    use_cache=Config.use_cache 
)

# =========================================================================
# PARTE 3: DEFINICIÓN DEL MODELO 
# =========================================================================
if Config.model_type == "unet":
    model = UNet(
        spatial_dims=Config.spatial_dims, 
        in_channels=Config.in_channels,   
        out_channels=Config.out_channels, 
        channels=(32, 64, 128, 256, 512), 
        strides=(2, 2, 2, 2),             
        num_res_units=2,                  
        dropout=0.2
    ).to(Config.device)
else:
    model = SegResNet(
        spatial_dims=Config.spatial_dims,
        init_filters=32,
        in_channels=Config.in_channels,
        out_channels=Config.out_channels,
        blocks_down=(1, 2, 2, 4), # 4 downsampling stages para que las hemorragias no desaparezcan
        blocks_up=(1, 1, 1), 
        norm="batch", 
        act="relu" 
    ).to(Config.device)

print(f"Modelo {Config.model_type.upper()} cargado en {Config.device}.")
print(f"Input channels: {Config.in_channels} | Output classes: {Config.out_channels}")



# =========================================================================
# PARTE 4: FUNCIÓN DE PÉRDIDA, OPTIMIZADOR Y SCHEDULER
# =========================================================================
# CORRECTION: Use DiceFocalLoss to handle extreme class imbalance
loss_function = DiceFocalLoss(
    to_onehot_y=True, 
    softmax=True, 
    include_background=False
)

optimizer = AdamW(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
# CORRECTION: CosineAnnealingLR decaying to 0 at final epoch (no warm restarts)
scheduler = CosineAnnealingLR(optimizer, T_max=Config.epochs, eta_min=0)

scaler = torch.cuda.amp.GradScaler() # Agregado para usar Mixed Precision y ahorrar memoria VRAM

def save_checkpoint(path, epoch, val_dice, train_loss, train_dice, val_loss=None):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_loss": train_loss,
        "train_dice": train_dice,
        "val_dice": val_dice,
        "val_loss": val_loss,
    }
    torch.save(checkpoint, path)


cycle_end_epochs = []
cycle_length = Config.checkpoint_interval
accumulated_epochs = 0
while accumulated_epochs < Config.epochs:
    accumulated_epochs += cycle_length
    cycle_end_epochs.append(accumulated_epochs)
    cycle_length *= Config.multiplicador_ciclo
print(f"Ciclos definidos con final en épocas: {cycle_end_epochs}")


# Configurar métricas y post-procesamiento para el Dice
dice_metric = DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=True)
post_pred = AsDiscrete(argmax=True, to_onehot=Config.out_channels)
post_label = AsDiscrete(to_onehot=Config.out_channels)

# Historial completo (manzanas con manzanas)
history = {
    "train_loss": [],
    "val_loss": [],
    "train_dice": [],
    "val_dice": [],
    "learning_rate": []
}

best_val_dice = 0.0
best_model_path = os.path.join(Config.models_dir, f"best_model_{Config.model_type}_{Config.label_type}_fold{Config.fold}.pth")
best_cycle_dice = 0.0
current_cycle_index = 0
current_cycle_end_epoch = cycle_end_epochs[current_cycle_index] if len(cycle_end_epochs) > 0 else None
current_cycle_checkpoint_path = None

# =========================================================================
# PARTE 5 & 6: LOOP DE ENTRENAMIENTO Y VALIDACIÓN
# =========================================================================
print("Comenzando entrenamiento...")
start_time = time.time()

for epoch in range(Config.epochs):
    # --- TRAINING ---
    model.train()
    epoch_loss = 0
    step = 0
    
    for batch_data in train_loader:
        step += 1
        inputs = batch_data["image"].to(Config.device)
        labels = batch_data["label"].to(Config.device)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            
            # CORRECTION: Handle deep supervision outputs (tuple of tensors)
            if isinstance(outputs, tuple) or isinstance(outputs, list):
                loss = sum([loss_function(out, labels) for out in outputs]) / len(outputs)
                main_output = outputs[0]
            else:
                loss = loss_function(outputs, labels)
                main_output = outputs
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
        
        # Calcular Train Dice en vivo
        # Hacemos detach().cpu() para liberar la memoria de la GPU (VRAM) y no acumular gráficos
        outputs_list = [post_pred(i.detach().cpu()) for i in decollate_batch(main_output)]
        labels_list = [post_label(i.cpu()) for i in decollate_batch(labels)]
        dice_metric(y_pred=outputs_list, y=labels_list)
        
    epoch_loss /= step
    
    # CORRECTION: Fix DiceMetric tuple unpacking bug
    dice_vals, not_nans = dice_metric.aggregate()
    valid_train = [v.item() for v, n in zip(dice_vals.flatten(), not_nans.flatten()) if n > 0]
    epoch_train_dice = float(np.mean(valid_train)) if valid_train else 0.0
    
    # PARCHE ANTI-NaN: Train Dice
    if math.isnan(epoch_train_dice):
        epoch_train_dice = 0.0
        
    dice_metric.reset() # Limpiar métrica para la siguiente fase
    
    history["train_loss"].append(epoch_loss)
    history["train_dice"].append(epoch_train_dice)
    history["learning_rate"].append(scheduler.get_last_lr()[0])
    
    scheduler.step()
    
    print(f"Época {epoch + 1}/{Config.epochs} | Train Loss: {epoch_loss:.4f} | Train Dice: {epoch_train_dice:.4f} | LR: {history['learning_rate'][-1]:.2e}")
    
    # --- VALIDATION & CHECKPOINTING ---
    if (epoch + 1) % Config.val_interval == 0 or epoch == 0:
        model.eval()
        val_loss = 0
        val_step = 0
        
        with torch.no_grad():
            for val_data in val_loader:
                val_step += 1
                
                # CORRECTION: Validation must run on the full volume using 2.5D slice-by-slice inference
                # Data is [B=1, C=1, H, W, Z]. Assuming batch_size=1 for validation volumetric.
                img_vol = val_data["image"][0, 0] # [H, W, Z]
                lbl_vol = val_data["label"][0]    # [1, H, W, Z]
                
                z_dim = img_vol.shape[-1]
                pred_vol = torch.zeros((Config.out_channels, img_vol.shape[0], img_vol.shape[1], z_dim), device=Config.device)
                
                for z in range(z_dim):
                    z0 = max(z - 1, 0)
                    z2 = min(z + 1, z_dim - 1)
                    
                    inp = torch.stack([img_vol[..., z0], img_vol[..., z], img_vol[..., z2]], dim=0)
                    inp = inp.unsqueeze(0).to(Config.device) # [1, 3, H, W]
                    
                    with torch.cuda.amp.autocast():
                        outputs = sliding_window_inference(
                            inputs=inp,
                            roi_size=(384, 384),
                            sw_batch_size=4,
                            predictor=model,
                            overlap=0.5
                        )
                    pred_vol[..., z] = outputs[0] # [6, H, W]
                
                val_labels = lbl_vol.unsqueeze(0).to(Config.device) # [1, 1, H, W, Z]
                pred_vol_b = pred_vol.unsqueeze(0) # [1, 6, H, W, Z]
                
                v_loss = loss_function(pred_vol_b, val_labels)
                val_loss += v_loss.item()
                
                # Dice de Validación
                # Movemos a CPU para no dejar megabytes de memoria bloqueando la GPU entre pacientes
                val_outputs_list = [post_pred(i.cpu()) for i in decollate_batch(pred_vol_b)]
                val_labels_list = [post_label(i.cpu()) for i in decollate_batch(val_labels)]
                dice_metric(y_pred=val_outputs_list, y=val_labels_list)
                
                # Liberamos explícitamente el volumen pesado de predicción de la GPU
                del pred_vol, pred_vol_b, val_labels
                
        val_loss /= val_step
        
        # CORRECTION: Fix DiceMetric tuple unpacking bug for Validation
        val_dice_vals, val_not_nans = dice_metric.aggregate()
        valid_val = [v.item() for v, n in zip(val_dice_vals.flatten(), val_not_nans.flatten()) if n > 0]
        val_mean_dice = float(np.mean(valid_val)) if valid_val else 0.0
        
        dice_metric.reset()
        
        # PARCHE ANTI-NaN: Validation Dice
        if math.isnan(val_mean_dice):
            val_mean_dice = 0.0
            
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_mean_dice)
        
        print(f"   => Validación | Val Loss: {val_loss:.4f} | Val Dice: {val_mean_dice:.4f}")
        
        # GUARDADO ROBUSTO
        if val_mean_dice > best_val_dice or (epoch == 0 and not os.path.exists(best_model_path)):
            if val_mean_dice > best_val_dice:
                best_val_dice = val_mean_dice
                
            save_checkpoint(
                best_model_path,
                epoch=epoch + 1,
                val_dice=val_mean_dice,
                train_loss=epoch_loss,
                train_dice=epoch_train_dice,
                val_loss=val_loss,
            )
            print(f"   => ¡Nuevo mejor modelo guardado (Récord: {best_val_dice:.4f})!")

        if val_mean_dice > best_cycle_dice or current_cycle_checkpoint_path is None:
            best_cycle_dice = val_mean_dice
            if current_cycle_end_epoch is not None:
                current_cycle_checkpoint_path = os.path.join(
                    Config.models_dir,
                    f"cycle_{current_cycle_index + 1:02d}_best_until_epoch_{current_cycle_end_epoch}.pth",
                )
                save_checkpoint(
                    current_cycle_checkpoint_path,
                    epoch=epoch + 1,
                    val_dice=val_mean_dice,
                    train_loss=epoch_loss,
                    train_dice=epoch_train_dice,
                    val_loss=val_loss,
                )


        if current_cycle_end_epoch is not None and (epoch + 1) >= current_cycle_end_epoch:
            cycle_checkpoint_path = os.path.join(
                Config.models_dir,
                f"cycle_{current_cycle_index + 1:02d}_final_epoch_{current_cycle_end_epoch}.pth",
            )
            save_checkpoint(
                cycle_checkpoint_path,
                epoch=current_cycle_end_epoch,
                val_dice=best_cycle_dice,
                train_loss=epoch_loss,
                train_dice=epoch_train_dice,
                val_loss=val_loss,
            )
            print(
                f"   => Fin de ciclo {current_cycle_index + 1}: checkpoint guardado en la época {current_cycle_end_epoch} "
                f"(mejor Val Dice del ciclo: {best_cycle_dice:.4f})."
            )

            current_cycle_index += 1
            best_cycle_dice = 0.0
            current_cycle_end_epoch = cycle_end_epochs[current_cycle_index] if current_cycle_index < len(cycle_end_epochs) else None
            current_cycle_checkpoint_path = None


end_time = time.time()
total_time = end_time - start_time
hours, rem = divmod(total_time, 3600)
minutes, seconds = divmod(rem, 60)
#Save last model checkpoint
final_model_path = os.path.join(Config.models_dir, f"final_model_{Config.model_type}_{Config.label_type}_fold{Config.fold}.pth")
save_checkpoint(
    final_model_path,
    epoch=Config.epochs,
    val_dice=history["val_dice"][-1],
    train_loss=history["train_loss"][-1],
    train_dice=history["train_dice"][-1],
    val_loss=history["val_loss"][-1],
)

print(f"\nEntrenamiento finalizado en {int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s.")
print(f"Mejor Dice de validación: {best_val_dice:.4f}")

# =========================================================================
# PARTE 7: GENERACIÓN Y GUARDADO DE CURVAS DE APRENDIZAJE
# =========================================================================
print("\nGenerando gráficas de las curvas de entrenamiento...")

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Ejes X (Train se grafica cada época, Val cada Config.val_interval)
train_epochs = list(range(1, Config.epochs + 1))
val_epochs = [i * Config.val_interval for i in range(1, len(history["val_loss"]) + 1)]

# 1. Curva de Pérdida (Train vs Val)
axes[0].plot(train_epochs, history["train_loss"], label='Train Loss', color='blue', linewidth=2)
axes[0].plot(val_epochs, history["val_loss"], label='Val Loss', color='red', linestyle='--', marker='o', linewidth=2)
axes[0].set_title('Loss: Train vs Validation', fontsize=14)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].grid(True, linestyle='--', alpha=0.7)
axes[0].legend()

# 2. Curva de Dice (Train vs Val)
axes[1].plot(train_epochs, history["train_dice"], label='Train Dice', color='blue', linewidth=2)
axes[1].plot(val_epochs, history["val_dice"], label='Val Dice', color='red', linestyle='--', marker='o', linewidth=2)
axes[1].set_title('Dice Score: Train vs Validation', fontsize=14)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Dice', fontsize=12)
axes[1].grid(True, linestyle='--', alpha=0.7)
axes[1].legend()

# 3. Curva del Learning Rate
axes[2].plot(train_epochs, history["learning_rate"], label='Learning Rate', color='green', linewidth=2)
axes[2].set_title('Learning Rate Schedule', fontsize=14)
axes[2].set_xlabel('Epoch', fontsize=12)
axes[2].set_ylabel('LR', fontsize=12)
axes[2].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
axes[2].grid(True, linestyle='--', alpha=0.7)
axes[2].legend()

plt.tight_layout()

plot_filename = os.path.join(Config.plots_dir, f"learning_curves_{Config.model_type}_{Config.label_type}_fold{Config.fold}.png")
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
plt.close()

print(f"Gráficas guardadas exitosamente en: {plot_filename}")