# MBH Inference API

Servidor FastAPI para inferencia aislada de los Modelos A y B.

## Ejecutar

Desde la raíz del repositorio:

```bash
conda activate visual-env
cd code
PYTHONPATH=src uvicorn api.main:app --host 127.0.0.1 --port 8000
```

## Endpoint

`POST /api/v1/predict` recibe `multipart/form-data` con:

- `file`: `.nii` o `.nii.gz`;
- `model_id`: `model_a` o `model_b`.

La respuesta exitosa es una máscara NIfTI comprimida (`application/gzip`). El Modelo C responde `501 MODEL_NOT_IMPLEMENTED`.

Timeouts configurados:

- `model_a`: 60 segundos;
- `model_b`: 300 segundos.

Ambos valores están definidos en `MODEL_TIMEOUTS_SECONDS` y son configurables.

## Regla para el frontend

El frontend debe revisar `response.ok` o `response.status` antes de decidir cómo interpretar el cuerpo:

- HTTP `200`: parsear como blob binario NIfTI;
- HTTP `4xx/5xx`: parsear como JSON y mostrar `error.code` y `error.message`.

No debe intentar parsear una respuesta de error como máscara binaria.

## Modelo B y espacio original

El pipeline 3D conserva la metadata `MetaTensor` generada por `Spacingd`,
`Orientationd`, `CropForegroundd` y `SpatialPadd`. Después del `argmax`,
`Invertd` revierte esas transformaciones usando `nearest_interp=True`, por lo
que la máscara conserva labels discretos y vuelve al espacio del NIfTI de
entrada. El backend valida shape, spacing, orientación, origen/affine y rango
de clases `0..5` antes de responder; si alguna comprobación falla devuelve
`INFERENCE_ERROR`.


