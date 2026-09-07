import os
import glob
from sklearn.model_selection import KFold
from monai.data import CacheDataset, DataLoader, Dataset

def get_data_dicts(data_dir, label_type="staple"): # "staple" o "majority"
    """
    Escanea el directorio y construye la lista de diccionarios para MONAI.
    label_type determina si cargamos el consenso STAPLE o el Majority.
    """
    data_dicts = []
    subject_dirs = sorted(glob.glob(os.path.join(data_dir, "*")))
    
    # Seleccionar dinámicamente el archivo de etiqueta
    label_filename = f"label_consensus_{label_type}.nii.gz"
    
    for subj_path in subject_dirs:
        if not os.path.isdir(subj_path):
            continue
            
        img_path = os.path.join(subj_path, "image.nii.gz")
        lbl_path = os.path.join(subj_path, label_filename)
        
        if os.path.exists(img_path) and os.path.exists(lbl_path):
            data_dicts.append({
                "image": img_path,
                "label": lbl_path
            })
            
    return data_dicts

def get_dataloaders(data_dir, label_type="staple", fold=0, num_folds=5, 
                    batch_size=4, train_transforms=None, val_transforms=None, 
                    use_cache=True):
    """
    Divide los datos en train/val usando K-Fold y retorna los DataLoaders.
    """
    data_dicts = get_data_dicts(data_dir, label_type)
    
    if len(data_dicts) == 0:
        raise ValueError(f"No se encontraron datos en {data_dir} con la etiqueta '{label_type}'")

    # 1. K-Fold Cross Validation (seed fija para reproducibilidad)
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    splits = list(kf.split(data_dicts))
    
    train_indices, val_indices = splits[fold]
    
    train_files = [data_dicts[i] for i in train_indices]
    val_files = [data_dicts[i] for i in val_indices]
    
    print(f"-> Preparando Fold {fold + 1}/{num_folds} | Label: {label_type}")
    print(f"   Train: {len(train_files)} casos | Val: {len(val_files)} casos")
    
    # 2. Instanciar Datasets (CacheDataset acelera muchísimo si tienes RAM suficiente)
    DatasetClass = CacheDataset if use_cache else Dataset
    
    train_ds = DatasetClass(
        data=train_files, 
        transform=train_transforms,
        # Si usas CacheDataset, ajusta cache_rate según tu RAM (ej. 0.5 para 50%)
    )
    val_ds = DatasetClass(
        data=val_files, 
        transform=val_transforms
    )
    
    # 3. Crear DataLoaders
    # En entrenamiento barajamos los datos. En val, batch_size=1 es estándar para inferencia volumétrica.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader