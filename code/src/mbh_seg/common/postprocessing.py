import torch


def logits_to_mask(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits, dim=1)
