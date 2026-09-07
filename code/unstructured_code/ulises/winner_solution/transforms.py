from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    RandSpatialCropSamplesd,
    RandCropByPosNegLabeld,
    CenterSpatialCropd,
    RandRotated,
    RandZoomd,
    RandGaussianNoised,
    RandAdjustContrastd,
    RandFlipd,
    RandCoarseShuffled,
    MapTransform,
    SpatialPadd,
    Spacingd,
    Orientationd,
    CropForegroundd,
    EnsureTyped,
)

class Extract3Slice2Dd(MapTransform):
    """
    Transformación personalizada para el enfoque del equipo ganador (nvauto):
    Toma un parche 3D de tamaño [X, Y, 3] y lo convierte en un input 2D de 3 canales.
    - Maneja tensores directamente o listas de 1 elemento provenientes de RandCrop.
    - La imagen pasa a [3, X, Y] (los 3 slices son canales).
    - La etiqueta toma el slice central [1, X, Y].
    """
    def __call__(self, data):
        # Si data es una lista de 1 elemento (por RandCropByPosNegLabeld), saca el diccionario.
        if isinstance(data, list) and len(data) == 1:
            d = dict(data[0])
        else:
            d = dict(data)
            
        for key in self.keys:
            if key == "image":
                # shape original: [1, X, Y, 3] -> removemos el canal original y movemos el eje Z al canal
                # [X, Y, 3] -> permute -> [3, X, Y]
                d[key] = d[key][0].permute(2, 0, 1)
            elif key == "label":
                # La etiqueta solo nos importa para el slice central (índice 1 de los 3 slices)
                d[key] = d[key][:, :, :, 1]
        
        # En caso de que el dataloader espere el mismo formato que recibió (una lista de 1 elemento)
        if isinstance(data, list) and len(data) == 1:
            return [d]
        return d

def get_transforms():
    """
    Retorna las transformaciones de entrenamiento y validación basadas en el paper INSTANCE 2022.
    """
    
    # 1. Preprocesamiento base (común para Train y Val)
    base_transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        
        # Resample a un spacing consistente (mejora la estandarización para la red 2D)
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 3.0),
            mode=("bilinear", "nearest"),
        ),
        
        # Orientación consistente
        Orientationd(keys=["image", "label"], axcodes="RPI"),
        
        # Quitar fondo (aire) para optimizar el recorte
        CropForegroundd(keys=["image", "label"], source_key="image"),
        
        # Escalar intensidades de CT fijas para ver hemorragias (ventana subdural/cerebro)
        ScaleIntensityRanged(
            keys=["image"], a_min=-15, a_max=200, 
            b_min=0.0, b_max=1.0, clip=True
        ),
    ]

    # 2. Transformaciones de Entrenamiento (con Data Augmentation)
    train_transforms = Compose(
        base_transforms + [
            # Pad por si alguna imagen es más pequeña que 384x384 (poco común, pero seguro)
            SpatialPadd(keys=["image", "label"], spatial_size=(384, 384, 3)),
            

            # CORRECTION: Crop size must be 384x384 as per the reference architecture.
            # Cambiado pos=1, neg=3 para que vea más tejido sano que hemorragia y no sobre-segmente.
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(384, 384, 3), 
                pos=1, neg=3,  # Aumentamos la proporción de negativos para evitar sobre-segmentación y mejorar la generalización.             
                num_samples=2, # Aumentado a 2 para extraer más muestras por volumen              
                image_key="image", 
                image_threshold=0,
                allow_smaller=True
            ),
            
            # Convertir el parche 3D a la representación 2D de 3 canales del paper
            Extract3Slice2Dd(keys=["image", "label"]),

            # --- DATA AUGMENTATION (Según el paper nvauto) ---
            RandRotated(keys=["image", "label"], prob=0.4, range_x=0.4), # Rotación
            RandZoomd(keys=["image", "label"], prob=0.4, min_zoom=0.8, max_zoom=1.2), # Zoom
            RandAdjustContrastd(keys=["image"], prob=0.2), # Contraste
            RandGaussianNoised(keys=["image"], prob=0.2), # Ruido
            RandCoarseShuffled(keys=["image"], prob=0.5, holes=8, spatial_size=32), # Coarse shuffle
            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5), # Flip eje X
            RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5), # Flip eje Y
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    # 3. Transformaciones de Validación (SIN Data Augmentation)
    val_transforms = Compose(
        base_transforms + [
            # CORRECTION: Do not crop Z axis. Keep full volume for proper validation.
            # Pad only spatial dimensions X and Y if needed, or crop to 384x384.
            # Use -1 in Z to keep all slices.
            SpatialPadd(keys=["image", "label"], spatial_size=(384, 384, -1)),
            CenterSpatialCropd(
                keys=["image", "label"],
                roi_size=(384, 384, -1),
            ),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
    return train_transforms, val_transforms