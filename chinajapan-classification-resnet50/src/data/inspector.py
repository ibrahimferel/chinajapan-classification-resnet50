"""
src/data/inspector.py

Inspeksi dataset sebelum training dimulai.
Harus dijalankan PERTAMA (notebook 00_dataset_inspection.ipynb)
sebelum data loader dan split dikonfigurasi.

Fungsi utama inspect_dataset() menghasilkan laporan lengkap:
    - Jumlah total gambar
    - Jumlah gambar per kelas
    - Distribusi ukuran gambar
    - Format file yang ditemukan
    - Gambar corrupt
    - Potensi class imbalance

Dataset harus memiliki struktur folder:
    dataset/
    ├── chinese/
    │   ├── img001.jpg
    │   └── ...
    └── japanese/
        ├── img001.jpg
        └── ...

atau struktur train/val/test:
    dataset/
    ├── train/
    │   ├── chinese/
    │   └── japanese/
    ├── validation/
    │   ├── chinese/
    │   └── japanese/
    └── test/
        ├── chinese/
        └── japanese/

Usage:
    from src.data.inspector import inspect_dataset
    report = inspect_dataset("/content/dataset")
    # report: dict berisi semua statistik
"""

import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _find_images(folder: str) -> List[str]:
    """Rekursif cari semua file gambar di folder dan subfolder-nya."""
    found = []
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if not d.startswith((".","__"))]
        for f in files:
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS:
                found.append(os.path.join(root, f))
    return sorted(found)


def _check_corrupt(path: str) -> bool:
    """Return True jika gambar tidak bisa dibaca (corrupt)."""
    try:
        img = cv2.imread(path)
        return img is None
    except Exception:
        return True


def _get_image_size(path: str) -> Optional[Tuple[int, int]]:
    """Return (height, width) atau None jika corrupt."""
    img = cv2.imread(path)
    if img is None:
        return None
    return img.shape[:2]


def inspect_dataset(
    dataset_dir: str,
    check_corrupt: bool = True,
    sample_sizes: bool = True,
    max_size_samples: int = 500,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Inspeksi dataset dan hasilkan laporan statistik.

    Args:
        dataset_dir:       Root folder dataset.
        check_corrupt:     Jika True, cek setiap file bisa dibaca.
                           Bisa lambat untuk dataset besar.
        sample_sizes:      Jika True, sampling ukuran gambar.
        max_size_samples:  Maksimum gambar yang di-sample untuk cek ukuran.
                           Set lebih kecil agar lebih cepat.
        verbose:           Jika True, print laporan ke console.

    Returns:
        Dict berisi semua statistik. Key:
            "total_images"      : int
            "per_class"         : Dict[str, int]
            "class_names"       : List[str]
            "imbalance_ratio"   : float   (max/min count, 1.0 = perfectly balanced)
            "formats"           : Dict[str, int]
            "corrupt_files"     : List[str]
            "size_stats"        : Dict (min, max, mean height/width dari sample)
            "unique_sizes"      : List[Tuple]
            "dataset_dir"       : str
    """
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset folder tidak ditemukan: {dataset_dir}")

    report: Dict[str, Any] = {"dataset_dir": dataset_dir}

    # ── Deteksi struktur folder ──────────────────────────────────────────
    # Cek apakah ini flat (chinese/, japanese/) atau split (train/, val/, test/)
    top_level = [
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
        and not d.startswith(".")
    ]

    if verbose:
        print(f"\n{'='*55}")
        print(f"  Dataset Inspection")
        print(f"  {dataset_dir}")
        print(f"{'='*55}")
        print(f"Top-level folders: {top_level}")

    # ── Hitung gambar per kelas ──────────────────────────────────────────
    per_class: Dict[str, int] = defaultdict(int)
    all_image_paths: List[str] = []
    format_count: Dict[str, int] = defaultdict(int)

    for top in top_level:
        top_path = os.path.join(dataset_dir, top)
        images = _find_images(top_path)
        per_class[top] = len(images)
        all_image_paths.extend(images)
        for img_path in images:
            ext = os.path.splitext(img_path)[1].lower()
            format_count[ext] += 1

    total = len(all_image_paths)
    report["total_images"] = total
    report["per_class"] = dict(per_class)
    report["class_names"] = sorted(per_class.keys())
    report["formats"] = dict(format_count)

    counts = list(per_class.values())
    if len(counts) >= 2:
        ratio = max(counts) / min(counts) if min(counts) > 0 else float("inf")
    else:
        ratio = 1.0
    report["imbalance_ratio"] = round(ratio, 4)

    if verbose:
        print(f"\n📊 Jumlah gambar:")
        for cls, count in sorted(per_class.items()):
            pct = count / total * 100 if total > 0 else 0
            print(f"   {cls:<20} : {count:>6} gambar ({pct:.1f}%)")
        print(f"   {'TOTAL':<20} : {total:>6} gambar")
        print(f"\n   Imbalance ratio (max/min) : {ratio:.4f}", end="")
        if ratio > 2.0:
            print("  ⚠️  Imbalance tinggi — pertimbangkan class weighting")
        elif ratio > 1.2:
            print("  ⚠️  Sedikit imbalance")
        else:
            print("  ✅ Relatif seimbang")

        print(f"\n📁 Format file: {dict(format_count)}")

    # ── Cek corrupt ──────────────────────────────────────────────────────
    corrupt: List[str] = []
    if check_corrupt:
        if verbose:
            print(f"\n🔍 Mengecek file corrupt ({total} file)...")
        for path in all_image_paths:
            if _check_corrupt(path):
                corrupt.append(path)
        report["corrupt_files"] = corrupt
        if verbose:
            if corrupt:
                print(f"   ⚠️  {len(corrupt)} file corrupt:")
                for c in corrupt[:10]:
                    print(f"      {c}")
                if len(corrupt) > 10:
                    print(f"      ... dan {len(corrupt)-10} lainnya")
            else:
                print("   ✅ Tidak ada file corrupt")
    else:
        report["corrupt_files"] = []
        if verbose:
            print("\n⏭️  Pengecekan corrupt dilewati (check_corrupt=False)")

    # ── Sampling ukuran gambar ───────────────────────────────────────────
    if sample_sizes and all_image_paths:
        sample_paths = all_image_paths[:max_size_samples]
        if verbose:
            print(f"\n📐 Sampling ukuran gambar ({len(sample_paths)} sampel)...")

        heights, widths = [], []
        unique_sizes = set()
        for path in sample_paths:
            size = _get_image_size(path)
            if size:
                h, w = size
                heights.append(h)
                widths.append(w)
                unique_sizes.add((h, w))

        if heights:
            size_stats = {
                "height": {
                    "min": int(np.min(heights)),
                    "max": int(np.max(heights)),
                    "mean": float(np.mean(heights)),
                },
                "width": {
                    "min": int(np.min(widths)),
                    "max": int(np.max(widths)),
                    "mean": float(np.mean(widths)),
                },
                "n_sampled": len(heights),
                "n_unique_sizes": len(unique_sizes),
            }
            report["size_stats"] = size_stats
            report["unique_sizes"] = sorted(unique_sizes)

            if verbose:
                print(f"   Height : min={size_stats['height']['min']}  "
                      f"max={size_stats['height']['max']}  "
                      f"mean={size_stats['height']['mean']:.0f}")
                print(f"   Width  : min={size_stats['width']['min']}  "
                      f"max={size_stats['width']['max']}  "
                      f"mean={size_stats['width']['mean']:.0f}")
                print(f"   Unique sizes: {len(unique_sizes)} "
                      f"({'gambar seragam' if len(unique_sizes)==1 else 'gambar bervariasi'})")
                if len(unique_sizes) <= 5:
                    for s in sorted(unique_sizes):
                        print(f"      {s[0]}×{s[1]}")

    # ── Summary akhir ────────────────────────────────────────────────────
    if verbose:
        print(f"\n{'='*55}")
        print(f"  Inspection selesai.")
        print(f"  Gunakan laporan ini untuk mengkonfigurasi split.py")
        print(f"  dan memastikan class_names di config.py benar.")
        print(f"{'='*55}\n")

    return report