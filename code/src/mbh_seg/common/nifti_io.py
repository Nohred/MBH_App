from pathlib import Path
from typing import Tuple

import numpy as np
import SimpleITK as sitk


def load_nifti(path: str | Path, pixel_type=None) -> Tuple[np.ndarray, sitk.Image]:
    image = sitk.ReadImage(str(path), pixel_type) if pixel_type is not None else sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(image), image


def save_mask(mask: np.ndarray, reference: sitk.Image, path: str | Path) -> None:
    output = sitk.GetImageFromArray(np.asarray(mask, dtype=np.uint8))
    output.SetOrigin(reference.GetOrigin())
    output.SetDirection(reference.GetDirection())
    output.SetSpacing(reference.GetSpacing())
    sitk.WriteImage(output, str(path))
