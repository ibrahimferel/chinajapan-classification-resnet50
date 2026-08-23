"""
src/utils/seed.py

Fix random seed untuk semua library yang digunakan.
Harus dipanggil PERTAMA di setiap notebook, sebelum import library lain.

Kenapa satu file tersendiri?
    Karena seed harus diset sebelum TensorFlow menginisialisasi GPU.
    Jika seed diset belakangan, beberapa operasi GPU sudah non-deterministic.

Usage:
    from src.utils.seed import set_seed
    set_seed(42)  # panggil di cell pertama notebook

Catatan:
    Reproducibility penuh di GPU tidak selalu 100% terjamin karena
    beberapa operasi CUDA bersifat inherently non-deterministic.
    set_seed() meminimalkan variasi, tapi mungkin tidak eliminasi sempurna.
    Untuk paper, cukup laporkan seed yang digunakan.
"""

import os
import random


def set_seed(seed: int = 42) -> None:
    """
    Set random seed untuk Python, NumPy, dan TensorFlow secara konsisten.

    Urutan setting penting:
        1. Environment variable dulu (sebelum TF init)
        2. Python built-in random
        3. NumPy
        4. TensorFlow

    Args:
        seed: Integer seed. Default 42.
               Gunakan nilai yang sama di seluruh eksperimen.
    """
    # 1. Environment variable — harus sebelum import TensorFlow
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"   # aktifkan deterministic ops di TF/GPU

    # 2. Python built-in random
    random.seed(seed)

    # 3. NumPy
    import numpy as np
    np.random.seed(seed)

    # 4. TensorFlow
    import tensorflow as tf
    tf.random.set_seed(seed)

    print(f"✅ Seed = {seed} (Python, NumPy, TensorFlow)")