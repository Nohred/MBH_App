import os
import glob
from sklearn.model_selection import KFold
from monai.data import CacheDataset, DataLoader, Dataset


def resolve_label_filename(label_type: str) -> str:
    """
    Convierte un nombre corto de etiqueta al nombre real del archivo.
    label_type soportado:
        - "staple"
        - "majority"
    """
    label_type = label_type.lower().strip()

    mapping = {
        "staple": "label_consensus_staple.nii.gz",
        "majority": "label_consensus_majority.nii.gz",
    }

    if label_type not in mapping:
        raise ValueError(
            f"label_type='{label_type}' no soportado. Usa 'staple' o 'majority'."
        )

    return mapping[label_type]


def get_data_dicts(data_dir, label_type="staple"):
    """
    Escanea el dataset y arma la lista de diccionarios para MONAI.

    Estructura esperada:
        data_dir/
            caso_001/
                image.nii.gz
                label_consensus_staple.nii.gz
            caso_002/
                image.nii.gz
                label_consensus_staple.nii.gz
    """
    data_dicts = []
    subject_dirs = sorted(glob.glob(os.path.join(data_dir, "*")))
    label_filename = resolve_label_filename(label_type)

    for subj_path in subject_dirs:
        if not os.path.isdir(subj_path):
            continue

        img_path = os.path.join(subj_path, "image.nii.gz")
        lbl_path = os.path.join(subj_path, label_filename)

        if os.path.exists(img_path) and os.path.exists(lbl_path):
            data_dicts.append(
                {
                    "image": img_path,
                    "label": lbl_path,
                    "case_id": os.path.basename(subj_path),
                }
            )

    return data_dicts


def get_fold_split(data_dicts, fold=0, num_folds=5, seed=42):
    """
    Separa los casos en train/val usando K-Fold.
    """
    if len(data_dicts) < num_folds:
        raise ValueError(
            f"No hay suficientes casos ({len(data_dicts)}) para {num_folds} folds."
        )

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    splits = list(kf.split(data_dicts))

    if fold < 0 or fold >= num_folds:
        raise ValueError(f"fold debe estar entre 0 y {num_folds - 1}, recibido: {fold}")

    train_idx, val_idx = splits[fold]
    train_files = [data_dicts[i] for i in train_idx]
    val_files = [data_dicts[i] for i in val_idx]

    return train_files, val_files


def get_dataloaders(
    data_dir,
    label_type="staple",
    fold=0,
    num_folds=5,
    batch_size=1,
    train_transforms=None,
    val_transforms=None,
    use_cache=True,
    cache_rate=1.0,
    num_workers=4,
    seed=42,
):
    """
    DataLoaders para Viola-UNet 3D.

    Recomendaciones:
    - batch_size=1 para entrenamiento 3D volumétrico.
    - Si falta RAM, bajar cache_rate o usar use_cache=False.
    """
    data_dicts = get_data_dicts(data_dir=data_dir, label_type=label_type)

    if len(data_dicts) == 0:
        raise ValueError(
            f"No se encontraron casos en '{data_dir}' con label_type='{label_type}'."
        )

    train_files, val_files = get_fold_split(
        data_dicts=data_dicts,
        fold=fold,
        num_folds=num_folds,
        seed=seed,
    )

    print(f"-> Preparando Fold {fold + 1}/{num_folds} | Label: {label_type}")
    print(f"   Train: {len(train_files)} casos | Val: {len(val_files)} casos")

    DatasetClass = CacheDataset if use_cache else Dataset

    if use_cache:
        train_ds = DatasetClass(
            data=train_files,
            transform=train_transforms,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )
        val_ds = DatasetClass(
            data=val_files,
            transform=val_transforms,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )
    else:
        train_ds = DatasetClass(data=train_files, transform=train_transforms)
        val_ds = DatasetClass(data=val_files, transform=val_transforms)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    data_dir = "/media/ulises-calderon/SteamDisk/Recuperacion/Escuela/8vo Semestre/ProyectoIntegrador1/data/MBH_Train_2025_voxel-label"
    files = get_data_dicts(data_dir, label_type="staple")
    print(f"Casos encontrados: {len(files)}")
    if files:
        print(files[0])