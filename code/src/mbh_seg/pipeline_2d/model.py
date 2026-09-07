import torch
from monai.networks.nets import SegResNet

from ..common.model import InferenceModel
from ..common.preprocessing import normalize_ct_for_model


class SegResNet2D(SegResNet):
    def __init__(self) -> None:
        super().__init__(
            spatial_dims=2,
            init_filters=32,
            in_channels=3,
            out_channels=6,
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
            norm="batch",
            act="relu",
        )


class SegResNetInference(InferenceModel):
    def __init__(self, network: SegResNet2D, device: torch.device) -> None:
        self.network = network
        self.device = device

    def preprocess(self, image):
        return normalize_ct_for_model(image)

    def predict(self, image, z_index: int):
        from .inference import predict_slice

        return predict_slice(self.network, image, z_index, self.device)

    def postprocess(self, prediction):
        return prediction


def load_model(checkpoint_path: str, device: torch.device) -> SegResNetInference:
    model = SegResNet2D().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return SegResNetInference(model, device)
