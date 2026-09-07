import numpy as np


def window_hu(array: np.ndarray, a_min: float, a_max: float) -> np.ndarray:
    clipped = np.clip(array, a_min, a_max)
    return ((clipped - a_min) / (a_max - a_min)).astype(np.float32)


def normalize_ct_for_model(array: np.ndarray) -> np.ndarray:
    return window_hu(array, -15.0, 200.0)
