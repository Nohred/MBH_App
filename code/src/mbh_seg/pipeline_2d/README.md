# Modelo A: SegResNet 2D

Inferencia 2D slice-por-slice con tres canales contiguos (`z-1`, `z`, `z+1`). Usa `SegResNet` MONAI con `roi_size=(384,384)`, `sw_batch_size=4` y `overlap=0.5`.

## Preprocesamiento

La ruta de inferencia de `winner_solution/evaluate.py` carga el volumen directamente con SimpleITK y solo aplica windowing HU `[-15,200]` a `[0,1]`. No aplica explícitamente `Spacingd`, `Orientationd` ni `CropForegroundd`.

En cambio, `winner_solution/transforms.py` sí aplica spacing `(1,1,3)`, orientación `RPI`, crop de foreground y padding/crop de `384x384x3`; esa ruta pertenece al entrenamiento/validación histórica y no debe mezclarse con la ruta actual sin una decisión explícita.

La extracción aquí conserva la ruta de `evaluate.py` para mantener el comportamiento de inferencia existente.
