import torch
from monai.inferers import sliding_window_inference
from monai.data import MetaTensor
from monai.transforms import Invertd

from .transforms import PATCH_SIZE, get_inference_transforms

SW_BATCH_SIZE = 1
OVERLAP = 0.5


def preprocess_case(image_path: str):
    transform = get_inference_transforms()
    return transform({"image": image_path}), transform


def predict_volume(model, preprocessed, device: torch.device) -> torch.Tensor:
    processed_case, processed_transform = preprocessed
    image_tensor = processed_case["image"]
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

    pred_classes = torch.argmax(outputs[0], dim=0).to(torch.uint8)
    processed_case["pred"] = MetaTensor(
        pred_classes.unsqueeze(0),
        meta=processed_case["image"].meta,
    )
    inverse = Invertd(
        keys="pred",
        transform=processed_transform,
        orig_keys="image",
        nearest_interp=True,
        to_tensor=True,
    )
    return inverse(processed_case)["pred"][0].to(torch.uint8)
