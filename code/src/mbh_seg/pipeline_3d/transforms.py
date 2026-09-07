from monai.transforms import (
    Compose,
    ConcatItemsd,
    CopyItemsd,
    CropForegroundd,
    DeleteItemsd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    SpatialPadd,
    Spacingd,
)


PATCH_SIZE = (160, 160, 32)
SPACING = (1.0, 1.0, 3.0)


def get_inference_transforms():
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        CopyItemsd(keys=["image"], times=2, names=["img_2", "img_3"]),
        ScaleIntensityRanged(keys=["image"], a_min=15, a_max=85, b_min=0.0, b_max=1.0, clip=True),
        ScaleIntensityRanged(keys=["img_2"], a_min=-15, a_max=200, b_min=0.0, b_max=1.0, clip=True),
        ScaleIntensityRanged(keys=["img_3"], a_min=-100, a_max=1300, b_min=0.0, b_max=1.0, clip=True),
        ConcatItemsd(keys=["image", "img_2", "img_3"], name="image"),
        DeleteItemsd(keys=["img_2", "img_3"]),
        Spacingd(keys=["image"], pixdim=SPACING, mode="bilinear"),
        Orientationd(keys=["image"], axcodes="RPI"),
        CropForegroundd(keys=["image"], source_key="image"),
        SpatialPadd(keys=["image"], spatial_size=PATCH_SIZE),
        EnsureTyped(keys=["image"]),
    ])
