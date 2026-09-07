from pathlib import Path
from typing import Any

import torch


def load_state_dict(path: str | Path, device: torch.device, checkpoint_key: str | None = None) -> Any:
    checkpoint = torch.load(str(path), map_location=device, weights_only=True)
    if checkpoint_key is not None:
        return checkpoint[checkpoint_key]
    return checkpoint
