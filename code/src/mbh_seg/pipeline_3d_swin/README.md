# Modelo C: SwinUNETR

Este directorio es únicamente un placeholder. No contiene código de inferencia ni pesos copiados.

Antes de integrarlo hay que confirmar con Jonathan:

- checkpoint exacto y formato real (`state_dict` o diccionario con `model_state_dict`);
- configuración completa de SwinUNETR: `in_channels`, `out_channels`, `feature_size`, `img_size`, `use_checkpoint` y `use_v2`;
- spacing, orientación, clipping, normalización y tamaño de patch usados en inferencia;
- estrategia de `sliding_window_inference` y dispositivo;
- convención de ejes de entrada y salida;
- significado de las seis clases;
- versiones compatibles de PyTorch y MONAI;
- si existe una función de inferencia validada o un caso de prueba reproducible.
