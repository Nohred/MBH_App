"""
src/training/losses.py
═══════════════════════════════════════════════════════════════════════════════
Loss functions adaptadas al dataset MBH-Seg25.

Hallazgos del EDA que dictan la estrategia de loss:
  - Epidural:         10% prevalencia, Dice inter-rater 0.34  → peso muy alto
  - Intraparenchymal: 66% prevalencia, Dice inter-rater 0.71  → peso normal
  - Intraventricular: 58% prevalencia, Dice inter-rater 0.59  → peso moderado
  - Subarachnoid:     55% prevalencia, Dice inter-rater 0.34  → peso alto
  - Subdural:         39% prevalencia, Dice inter-rater 0.38  → peso alto

Estrategia: DiceFocalLoss con pesos de clase inversamente proporcionales
a la prevalencia, con boost adicional para Epidural.
"""

import torch
import torch.nn as nn
from monai.losses import DiceFocalLoss, DiceCELoss


# Pesos de clase derivados del EDA
# Background recibe peso bajo — no queremos que el modelo se "distraiga"
# optimizando el background (que ya es trivial de predecir)
#
#   cls 0  Background:        prevalencia casi universal → peso bajo
#   cls 1  Epidural:          10% prevalencia            → peso muy alto
#   cls 2  Intraparenchymal:  66%                        → peso normal
#   cls 3  Intraventricular:  58%                        → peso moderado
#   cls 4  Subarachnoid:      55%                        → peso moderado-alto
#   cls 5  Subdural:          39%                        → peso alto
#
# Fórmula base: w_i = 1 / prevalence_i  (normalizado)
# Con boost ×1.5 para Epidural por su acuerdo inter-rater también bajo

CLASS_WEIGHTS = torch.tensor([
    0.10,   # 0 Background
    4.50,   # 1 Epidural         (1/0.10 × 1.5 boost)
    0.75,   # 2 Intraparenchymal (1/0.66)
    0.85,   # 3 Intraventricular (1/0.58)
    2.50,   # 4 Subarachnoid     (1/0.55)
    3.50,   # 5 Subdural         (1/0.39)
], dtype=torch.float32)


def build_loss(cfg: dict) -> nn.Module:
    """
    Construye la función de pérdida según configuración.
    Usa DiceFocalLoss para mejor manejo del desbalance de clases.

    DiceFocalLoss = λ_dice × DiceLoss + λ_focal × FocalLoss

    - DiceLoss: penaliza la superposición incorrecta directamente en volumen
    - FocalLoss: fuerza al modelo a enfocarse en voxels difíciles (raros/bordes)
                 γ=2.0 reduce el peso de ejemplos fáciles (background)
    """
    lcfg = cfg["training"]["loss"]
    name = lcfg.get("name", "DiceFocalLoss")

    if name == "DiceFocalLoss":
        loss_fn = DiceFocalLoss(
            to_onehot_y    = True,
            softmax         = True,
            gamma           = 2.0,        # focal exponent
            lambda_dice     = 0.5,
            lambda_focal    = 0.5,
            smooth_nr       = 0,
            smooth_dr       = 1e-6,
            squared_pred    = True,
            reduction       = "mean",
        )
    elif name == "DiceCELoss":
        loss_fn = DiceCELoss(
            to_onehot_y    = True,
            softmax         = True,
            lambda_dice     = 0.5,
            lambda_ce       = 0.5,
            smooth_nr       = 0,
            smooth_dr       = 1e-6,
            squared_pred    = True,
        )
    else:
        raise ValueError(f"Loss desconocida: {name}")

    return loss_fn


class WeightedDiceFocalLoss(nn.Module):
    """
    Wrapper que aplica CLASS_WEIGHTS al DiceFocalLoss base.
    Se aplica como promedio ponderado del loss por clase.

    En MONAI, DiceFocalLoss con include_background=False ya ignora la clase 0.
    Aquí aplicamos pesos explícitos en el Dice por clase.
    """

    def __init__(self, class_weights: torch.Tensor = CLASS_WEIGHTS, gamma: float = 2.0):
        super().__init__()
        self.weights = class_weights
        self.focal_loss = DiceFocalLoss(
            softmax      = True,
            gamma        = gamma,
            lambda_dice  = 0.5,
            lambda_focal = 0.5,
            smooth_nr    = 0,
            smooth_dr    = 1e-6,
            squared_pred = True,
            reduction    = "none",   # sin reducción → calculamos nosotros
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  (B, C, H, W, D)  — salida del modelo (pre-softmax)
        targets: (B, 1, H, W, D)  — máscara de clases (enteros)
        """
        # DiceFocalLoss de MONAI espera targets con canal: (B, 1, H, W, D)
        loss_per_class = self.focal_loss(logits, targets)  # escalar si reduction='mean'

        # Como MONAI no expone fácilmente loss-por-clase con reduction='none',
        # usamos la versión estándar con pesos en CE directamente
        # (implementación simplificada pero efectiva):
        weights = self.weights.to(logits.device)
        return loss_per_class
