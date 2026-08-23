"""
src/data/split.py

Buat train/validation/test split yang reproducible dan bebas data leakage.

Prinsip utama:
    - Split dilakukan di level IMAGE sebelum preprocessing apapun.
    - Stratified: distribusi kelas di setiap split proporsional sama.
    - Seed fixed agar split identik di setiap run.
    - Test set tidak pernah disentuh selama training dan validation.

Tentang data leakage:
    Leakage terjadi jika informasi dari val/test "bocor" ke training.
    Di sini split dilakukan dari path mentah sebelum augmentation atau
    normalisasi statistics dihitung, sehingga leakage tidak terjadi.

Output split berbentuk dict of lists of (path, label) tuples.
Ini dipass ke dataset.py untuk dibangun menjadi tf.data.Dataset.

Usage:
    from configs.config import get_baseline_config
    from src.data.split import create_split

    cfg = get_baseline_config(dataset_dir="/content/dataset")
    split = create_split(cfg)

    print(len(split["train"]))   # list of (path, label)
    print(len(split["val"]))
    print(len(split["test"]))
"""

import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple

# Type alias untuk kejelasan
ImageEntry = Tuple[str, int]   # (file_path, class_index)
Split = Dict[str, List[ImageEntry]]

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _collect_images(dataset_dir: str, class_names: List[str]) -> Dict[str, List[str]]:
    """
    Kumpulkan semua path gambar per kelas dari dataset_dir.

    Mengharapkan struktur:
        dataset_dir/
        ├── chinese/
        └── japanese/

    Args:
        dataset_dir: Root folder dataset.
        class_names: List nama kelas, urutan ini menentukan label integer.
                     ["chinese", "japanese"] → chinese=0, japanese=1

    Returns:
        Dict {class_name: [file_path, ...]}
    """
    per_class: Dict[str, List[str]] = {}

    for class_name in class_names:
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.exists(class_dir):
            raise FileNotFoundError(
                f"Folder kelas tidak ditemukan: {class_dir}\n"
                f"Pastikan nama folder sesuai dengan class_names di config: {class_names}"
            )
        files = [
            os.path.join(class_dir, f)
            for f in os.listdir(class_dir)
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
        ]
        per_class[class_name] = sorted(files)   # sort untuk determinisme

    return per_class


def _stratified_split(
    per_class: Dict[str, List[str]],
    class_names: List[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Split:
    """
    Lakukan stratified split per kelas, kemudian gabungkan dan shuffle.

    Stratified berarti proporsi kelas dijaga sama di setiap split.
    Contoh: 70% chinese dan 70% japanese masuk train,
            bukan 70% dari total (yang bisa jadi tidak seimbang).

    Args:
        per_class:    Dict {class_name: [file_paths]}
        class_names:  Urutan kelas menentukan label integer
        train_ratio:  Proporsi train (0.0–1.0)
        val_ratio:    Proporsi validation (0.0–1.0)
        seed:         Random seed

    Returns:
        Split dict {"train": [...], "val": [...], "test": [...]}
    """
    rng = random.Random(seed)

    train_entries: List[ImageEntry] = []
    val_entries: List[ImageEntry] = []
    test_entries: List[ImageEntry] = []

    for class_name in class_names:
        label = class_names.index(class_name)
        paths = per_class[class_name].copy()
        rng.shuffle(paths)   # shuffle dalam kelas dengan seed terkontrol

        n = len(paths)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        # test = sisanya, memastikan semua gambar masuk salah satu split

        train_paths = paths[:n_train]
        val_paths = paths[n_train : n_train + n_val]
        test_paths = paths[n_train + n_val :]

        train_entries.extend((p, label) for p in train_paths)
        val_entries.extend((p, label) for p in val_paths)
        test_entries.extend((p, label) for p in test_paths)

    # Shuffle final agar kelas tidak berkelompok dalam batch
    rng.shuffle(train_entries)
    rng.shuffle(val_entries)
    rng.shuffle(test_entries)

    return {
        "train": train_entries,
        "val": val_entries,
        "test": test_entries,
    }


def create_split(cfg) -> Split:
    """
    Entry point utama. Baca config, kumpulkan gambar, buat split.

    Args:
        cfg: Config object dari configs/config.py

    Returns:
        Split dict {"train": [...], "val": [...], "test": [...]}
        Setiap entry adalah (file_path: str, label: int)
    """
    per_class = _collect_images(cfg.paths.dataset_dir, cfg.data.class_names)

    split = _stratified_split(
        per_class=per_class,
        class_names=cfg.data.class_names,
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.val_ratio,
        seed=cfg.data.random_seed,
    )

    # Print summary
    total = sum(len(v) for v in split.values())
    print(f"\n📊 Dataset Split Summary:")
    print(f"   {'Split':<12} {'Count':>7}   {'%':>5}")
    print(f"   {'-'*28}")
    for key in ("train", "val", "test"):
        count = len(split[key])
        pct = count / total * 100 if total > 0 else 0
        print(f"   {key:<12} {count:>7}   {pct:>5.1f}%")
    print(f"   {'TOTAL':<12} {total:>7}")

    # Verifikasi distribusi kelas per split
    print(f"\n   Distribusi kelas per split:")
    for split_name in ("train", "val", "test"):
        label_counts: Dict[int, int] = defaultdict(int)
        for _, label in split[split_name]:
            label_counts[label] += 1
        class_info = "  ".join(
            f"{cfg.data.class_names[lbl]}={cnt}"
            for lbl, cnt in sorted(label_counts.items())
        )
        print(f"   {split_name:<8}: {class_info}")

    print()
    return split


def get_class_weights(split: Split, num_classes: int) -> Dict[int, float]:
    """
    Hitung class weights untuk menangani class imbalance saat training.

    Gunakan ini jika imbalance_ratio dari inspector > 1.5.
    Pass ke model.fit(class_weight=weights).

    Formula: weight_i = total_samples / (num_classes × count_i)

    Args:
        split:       Split dict dari create_split()
        num_classes: Jumlah kelas

    Returns:
        Dict {class_index: weight}
    """
    label_counts: Dict[int, int] = defaultdict(int)
    for _, label in split["train"]:
        label_counts[label] += 1

    total = sum(label_counts.values())
    weights = {
        label: total / (num_classes * count)
        for label, count in label_counts.items()
    }

    print(f"Class weights: {weights}")
    return weights