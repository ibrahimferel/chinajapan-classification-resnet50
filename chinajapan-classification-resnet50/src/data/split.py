from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split


def split_dataset(data_dir: str, test_size: float = 0.2, val_size: float = 0.2, random_state: int = 42):
    """Split dataset into train/val/test using stratified sampling."""
    data_path = Path(data_dir)
    image_paths = []
    labels = []

    for class_dir in sorted(data_path.iterdir()):
        if not class_dir.is_dir():
            continue
        label = class_dir.name
        for image_file in sorted(class_dir.glob('*')):
            if image_file.is_file():
                image_paths.append(str(image_file))
                labels.append(label)

    if len(image_paths) < 3:
        raise ValueError("Dataset terlalu kecil untuk split train/val/test.")

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths,
        labels,
        test_size=test_size + val_size,
        stratify=labels,
        random_state=random_state,
    )

    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=test_size / (test_size + val_size),
        stratify=temp_labels,
        random_state=random_state,
    )

    return {
        "train": (train_paths, train_labels),
        "val": (val_paths, val_labels),
        "test": (test_paths, test_labels),
    }
