import asyncio
import os
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from mbh_seg.common.nifti_io import load_nifti, save_mask
from mbh_seg.pipeline_2d import load_model as load_model_a
from mbh_seg.pipeline_3d import load_model as load_model_b


MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024
MODEL_TIMEOUTS_SECONDS = {"model_a": 60, "model_b": 300}
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]
CHECKPOINTS = {
    "model_a": Path(__file__).parents[1] / "outputs/checkpoints/model_a_segresnet/segresnet_staple_best_fold0.pth",
    "model_b": Path(__file__).parents[1] / "outputs/checkpoints/model_b_violaunet/violaunet_staple_best_fold0.pth",
}


class InvalidNiftiError(Exception):
    pass


app = FastAPI(title="MBH Inference API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["*"]
)


def error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=2)
def get_model(model_id: str):
    checkpoint = CHECKPOINTS[model_id]
    if not checkpoint.is_file():
        raise FileNotFoundError(str(checkpoint))
    selected_device = device()
    if model_id == "model_a":
        return load_model_a(str(checkpoint), selected_device), selected_device
    return load_model_b(str(checkpoint), selected_device), selected_device


def validate_mask(mask: np.ndarray, reference: sitk.Image) -> None:
    expected_shape = tuple(reversed(reference.GetSize()))
    if tuple(mask.shape) != expected_shape:
        raise RuntimeError(
            f"La máscara tiene shape {tuple(mask.shape)}; se esperaba {expected_shape}."
        )
    if not np.issubdtype(mask.dtype, np.integer):
        raise RuntimeError(f"La máscara no es entera: dtype={mask.dtype}.")
    unique_values = np.unique(mask)
    if np.any((unique_values < 0) | (unique_values > 5)):
        raise RuntimeError(
            f"La máscara contiene valores fuera del rango 0-5: {unique_values.tolist()}."
        )


def validate_saved_metadata(output_path: str, reference: sitk.Image) -> None:
    output = sitk.ReadImage(output_path)
    if output.GetSize() != reference.GetSize():
        raise RuntimeError(f"La máscara guardada tiene size {output.GetSize()} distinto de {reference.GetSize()}.")
    if not np.allclose(output.GetSpacing(), reference.GetSpacing(), atol=1e-6):
        raise RuntimeError("El spacing de la máscara no coincide con el volumen original.")
    if not np.allclose(output.GetDirection(), reference.GetDirection(), atol=1e-6):
        raise RuntimeError("La orientación de la máscara no coincide con el volumen original.")
    if not np.allclose(output.GetOrigin(), reference.GetOrigin(), atol=1e-6):
        raise RuntimeError("El origen/affine de la máscara no coincide con el volumen original.")


def infer_file(model_id: str, image_path: str, output_path: str) -> None:
    model, selected_device = get_model(model_id)
    try:
        image, reference = load_nifti(image_path, pixel_type=sitk.sitkFloat32)
    except Exception as exc:
        raise InvalidNiftiError from exc

    if model_id == "model_a":
        normalized = model.preprocess(image)
        prediction = torch.stack([
            model.predict(normalized, z_index)
            for z_index in range(normalized.shape[0])
        ])
    else:
        processed = model.preprocess(image_path)
        prediction = model.predict(processed)

    mask = model.postprocess(prediction).numpy()
    if model_id == "model_b":
        # MONAI returns spatial tensors as (X, Y, Z); SimpleITK arrays use (Z, Y, X).
        mask = np.transpose(mask, (2, 1, 0))
        validate_mask(mask, reference)
    save_mask(mask, reference, output_path)
    if model_id == "model_b":
        validate_saved_metadata(output_path, reference)


async def read_upload(upload: UploadFile) -> bytes:
    chunks = []
    total_size = 0
    while chunk := await upload.read(1024 * 1024):
        total_size += len(chunk)
        if total_size > MAX_FILE_SIZE_BYTES:
            raise ValueError("FILE_TOO_LARGE")
        chunks.append(chunk)
    return b"".join(chunks)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "cuda_available": torch.cuda.is_available()}


@app.post("/api/v1/predict")
async def predict(file: UploadFile = File(...), model_id: str = Form(...)):
    if model_id not in MODEL_TIMEOUTS_SECONDS:
        if model_id == "model_c":
            return error_response(501, "MODEL_NOT_IMPLEMENTED", "El Modelo C todavía no está implementado.")
        return error_response(400, "INVALID_REQUEST", "model_id debe ser model_a o model_b.")

    filename = (file.filename or "").lower()
    if not (filename.endswith(".nii") or filename.endswith(".nii.gz")):
        return error_response(415, "UNSUPPORTED_FILE_TYPE", "Solo se aceptan archivos NIfTI .nii o .nii.gz.")

    try:
        upload_bytes = await read_upload(file)
    except ValueError:
        return error_response(413, "FILE_TOO_LARGE", "El archivo supera el límite de 100 MB.")

    suffix = ".nii.gz" if filename.endswith(".nii.gz") else ".nii"
    with tempfile.TemporaryDirectory(prefix="mbh_api_") as temp_dir:
        input_path = os.path.join(temp_dir, "input" + suffix)
        output_path = os.path.join(temp_dir, "mask.nii.gz")
        Path(input_path).write_bytes(upload_bytes)

        try:
            await asyncio.wait_for(
                run_in_threadpool(infer_file, model_id, input_path, output_path),
                timeout=MODEL_TIMEOUTS_SECONDS[model_id],
            )
        except FileNotFoundError:
            return error_response(404, "MODEL_NOT_FOUND", "No se encontró el checkpoint solicitado.")
        except asyncio.TimeoutError:
            return error_response(504, "INFERENCE_TIMEOUT", f"La inferencia superó el timeout de {MODEL_TIMEOUTS_SECONDS[model_id]} segundos.")
        except InvalidNiftiError:
            return error_response(422, "INVALID_NIFTI", "El archivo no es un volumen NIfTI válido.")
        except Exception as exc:
            return error_response(500, "INFERENCE_ERROR", str(exc))

        response_bytes = Path(output_path).read_bytes()
        return Response(
            content=response_bytes,
            media_type="application/gzip",
            headers={
                "Content-Disposition": f'attachment; filename="mask_{model_id}.nii.gz"',
                "X-Inference-Device": device().type,
            },
        )
