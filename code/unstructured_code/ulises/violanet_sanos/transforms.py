from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    CopyItemsd,
    ScaleIntensityRanged,
    ConcatItemsd,
    DeleteItemsd,
    Spacingd,
    CropForegroundd,
    Orientationd,
    SpatialPadd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotated,
    RandZoomd,
    RandGaussianNoised,
    RandAdjustContrastd,
    EnsureTyped,
)

def get_transforms():
    """
    Transforms 3D para Viola-UNet multiclase mejorado.
    """
    wind_levels = [
        (15, 85),     # brain / soft tissue
        (-15, 200),   # subdural-like
        (-100, 1300), # bone / wide
    ]

    spacing = (1.0, 1.0, 3.0)
    patch_size = (160, 160, 32)

    base_transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),

        # Crear 3 ventanas HU como 3 canales
        CopyItemsd(keys=["image"], times=2, names=["img_2", "img_3"]),

        ScaleIntensityRanged(
            keys=["image"],
            a_min=wind_levels[0][0], a_max=wind_levels[0][1],
            b_min=0.0, b_max=1.0, clip=True,
        ),
        ScaleIntensityRanged(
            keys=["img_2"],
            a_min=wind_levels[1][0], a_max=wind_levels[1][1],
            b_min=0.0, b_max=1.0, clip=True,
        ),
        ScaleIntensityRanged(
            keys=["img_3"],
            a_min=wind_levels[2][0], a_max=wind_levels[2][1],
            b_min=0.0, b_max=1.0, clip=True,
        ),

        ConcatItemsd(keys=["image", "img_2", "img_3"], name="image"),
        DeleteItemsd(keys=["img_2", "img_3"]),

        # Resample a (1.0, 1.0, 3.0)
        Spacingd(
            keys=["image", "label"],
            pixdim=spacing,
            mode=("bilinear", "nearest"),
        ),

        # Orientación consistente
        Orientationd(keys=["image", "label"], axcodes="RPI"),

        # Quitar fondo para que el crop sea más eficiente
        CropForegroundd(
            keys=["image", "label"],
            source_key="image",
        ),

        # Asegurar tamaño mínimo
        SpatialPadd(
            keys=["image", "label"],
            spatial_size=patch_size,
        ),
    ]

    train_transforms = Compose(
        base_transforms + [
            # EL CAMBIO MÁS IMPORTANTE: Forzar a que vea lesiones
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=patch_size,
                pos=1, neg=3, num_samples=4, # Saca 4 parches por imagen
                image_key="image", image_threshold=0,
                allow_smaller=True
            ),
            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
            RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
            RandFlipd(keys=["image", "label"], spatial_axis=2, prob=0.5),
            RandRotated(keys=["image", "label"], prob=0.2, range_x=0.15, range_y=0.15, range_z=0.15, mode=("bilinear", "nearest"), keep_size=True),
            RandZoomd(keys=["image", "label"], prob=0.2, min_zoom=0.9, max_zoom=1.1, mode=("trilinear", "nearest"), keep_size=True),
            RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.01),
            RandAdjustContrastd(keys=["image"], prob=0.15, gamma=(0.9, 1.1)),
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    val_transforms = Compose(
        base_transforms + [
            # En validación no hacemos crop aquí, porque validaremos con sliding_window_inference en todo el volumen
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    return train_transforms, val_transforms
