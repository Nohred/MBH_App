import torch
from monai.inferers import sliding_window_inference

from mbh_seg.common.preprocessing import normalize_ct_for_model
from .model import load_model


ROI_SIZE = (384, 384)
SW_BATCH_SIZE = 4
OVERLAP = 0.5


def build_input(volume, z_index: int) -> torch.Tensor:
    z_dim = volume.shape[0]
    previous_index = max(z_index - 1, 0)
    next_index = min(z_index + 1, z_dim - 1)
    return torch.stack(
        [torch.as_tensor(volume[previous_index]), torch.as_tensor(volume[z_index]), torch.as_tensor(volume[next_index])]
    ).unsqueeze(0)


def predict_slice(model, normalized_volume, z_index: int, device: torch.device) -> torch.Tensor:
    input_tensor = build_input(normalized_volume, z_index).to(device)
    with torch.inference_mode():
        logits = sliding_window_inference(
            inputs=input_tensor,
            roi_size=ROI_SIZE,
            sw_batch_size=SW_BATCH_SIZE,
            predictor=model,
            overlap=OVERLAP,
        )
    return torch.argmax(logits[0], dim=0).cpu()


def predict_volume(model, volume, device: torch.device) -> torch.Tensor:
    normalized_volume = model.preprocess(volume)
    predictions = [predict_slice(model, normalized_volume, z, device) for z in range(normalized_volume.shape[0])]
    return torch.stack(predictions)
