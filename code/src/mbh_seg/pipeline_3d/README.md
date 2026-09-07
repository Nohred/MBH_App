# Modelo B: ViolaUNet 3D

Inferencia 3D con patches `(160,160,32)`, `sliding_window_inference`, `sw_batch_size=1` y `overlap=0.5`.

## Configuración validada

`best_viola_multiclass_staple_fold0.pth` fue cargado con `load_state_dict(strict=True)` en el entorno `visual-env` y coincidió con 462 claves usando:

- `deep_supervision=True`
- `viola_att=True`
- `deep_supr_num=2`
- `gated_att=False`
- `sum_deep_supr=False`

La configuración de `evaluate.py` (`deep_supervision=False`, `viola_att=False`) falló con `strict=True` por claves inesperadas de `deep_supervision_heads`, `canc_att` y `super_head`. Por tanto, esta extracción usa la configuración compatible de `visualize_inference.py` y carga estricta.

## Preprocesamiento

Se conservan solo los transforms deterministas de validación/inferencia: tres ventanas HU `(15,85)`, `(-15,200)`, `(-100,1300)`, spacing `(1,1,3)`, orientación `RPI`, crop foreground y padding mínimo. No se incluyen folds, dataset ni augmentations de entrenamiento.
