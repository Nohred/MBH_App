import torch
from monai.inferers import sliding_window_inference

from .transforms import PATCH_SIZE, get_inference_transforms
from .model import load_model

SW_BATCH_SIZE = 1
OVERLAP = 0.5


def preprocess_case(image_path: str):
    processed = get_inference_transforms()({"image": image_path})
    return processed["image"]


def predict_volume(model, image_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    inputs = image_tensor.unsqueeze(0).to(device)
    with torch.inference_mode():
        outputs = sliding_window_inference(
            inputs=inputs,
            roi_size=PATCH_SIZE,
            sw_batch_size=SW_BATCH_SIZE,
            predictor=model,
            overlap=OVERLAP,
            sw_device=device,
            device="cpu",
        )
    if outputs.ndim == 6:
        outputs = outputs[:, 0]
    return torch.argmax(outputs[0], dim=0)
