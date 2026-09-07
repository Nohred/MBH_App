import torch

from ..common.checkpoint import load_state_dict
from ..common.model import InferenceModel
from .viola_unet import ViolaUNet


MODEL_KWARGS = {
    "spatial_dims": 3,
    "in_channels": 3,
    "out_channels": 6,
    "kernel_size": [[3, 3, 1], [3, 3, 1], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
    "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
    "upsample_kernel_size": [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
    "filters": (16, 32, 32, 64, 128),
    "dec_filters": (16, 16, 32, 64),
    "norm_name": ("BATCH", {"affine": True}),
    "act_name": ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
    "dropout": 0.2,
    "deep_supervision": True,
    "deep_supr_num": 2,
    "res_block": True,
    "trans_bias": True,
    "viola_att": True,
    "gated_att": False,
    "sum_deep_supr": False,
}


class ViolaInference(InferenceModel):
    def __init__(self, network: ViolaUNet, device: torch.device) -> None:
        self.network = network
        self.device = device

    def preprocess(self, image_path: str):
        from .inference import preprocess_case

        return preprocess_case(image_path)

    def predict(self, image_tensor):
        from .inference import predict_volume

        return predict_volume(self.network, image_tensor, self.device)

    def postprocess(self, prediction):
        return prediction


def load_model(checkpoint_path: str, device: torch.device) -> ViolaInference:
    model = ViolaUNet(**MODEL_KWARGS).to(device)
    state_dict = load_state_dict(checkpoint_path, device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return ViolaInference(model, device)
