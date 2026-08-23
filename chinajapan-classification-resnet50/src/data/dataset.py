"""
src/data/dataset.py

Bangun tf.data.Dataset yang siap digunakan untuk training dan evaluasi.

Menerima output dari split.py (list of (path, label) tuples) dan
menghasilkan tf.data.Dataset yang sudah di-preprocess, di-batch, dan
di-prefetch.

Pipeline per sample:
    file_path → decode (baca dari disk) → preprocess (resize + normalisasi)
    → (augment jika training) → batch → prefetch ke GPU

Kenapa tf.data bukan ImageDataGenerator?
    tf.data lebih cepat karena:
    - Baca file secara paralel (num_parallel_calls=AUTOTUNE)
    - Prefetch ke GPU saat CPU memproses batch berikutnya
    - Lebih mudah dikontrol dan di-debug

Usage:
    from src.data.dataset import build_dataset
    train_ds = build_dataset(split["train"], cfg, augment=False)
    val_ds   = build_dataset(split["val"],   cfg, augment=False)
    test_ds  = build_dataset(split["test"],  cfg, augment=False)
"""

from typing import List, Tuple

import tensorflow as tf

from src.data.preprocessing import build_decode_fn, build_preprocessing_fn

ImageEntry = Tuple[str, int]


def build_augmentation_fn():
    """
    Augmentasi ringan yang aman untuk lukisan.

    Sengaja dipisah dari preprocessing agar bisa di-toggle independen.
    Hanya diaplikasikan ke training set, TIDAK ke val dan test.

    Augmentasi yang dipilih:
        - Random horizontal flip: aman untuk sebagian besar lukisan
        - Random brightness/contrast kecil: mensimulasikan variasi pencahayaan

    Yang TIDAK digunakan:
        - Vertical flip: tidak natural untuk lukisan
        - Rotation besar: mengubah komposisi secara tidak realistis
        - Color jitter ekstrem: mengubah karakteristik artistik

    Catatan:
        Augmentasi ini TIDAK digunakan untuk baseline murni (use_augmentation=False).
        Hanya diaktifkan jika eksperimen augmentation dijalankan secara terpisah.
    """
    def augment_fn(image: tf.Tensor, label: tf.Tensor):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        return image, label

    return augment_fn


def build_dataset(
    entries: List[ImageEntry],
    cfg,
    augment: bool = False,
    shuffle: bool = None,
) -> tf.data.Dataset:
    """
    Bangun tf.data.Dataset dari list (file_path, label).

    Args:
        entries:  List of (file_path, label_int) dari split.py.
        cfg:      Config object dari configs/config.py.
        augment:  Jika True, aplikasikan augmentasi.
                  Hanya untuk training set. Val dan test selalu False.
        shuffle:  Jika None, otomatis True untuk augment=True (training),
                  False untuk augment=False (val/test).
                  Bisa di-override secara eksplisit.

    Returns:
        tf.data.Dataset siap untuk model.fit() atau model.evaluate()
    """
    if not entries:
        raise ValueError("entries kosong — pastikan split.py sudah dijalankan.")

    # Tentukan apakah perlu shuffle
    should_shuffle = augment if shuffle is None else shuffle

    # Pisahkan paths dan labels
    paths = [e[0] for e in entries]
    labels = [e[1] for e in entries]

    # Buat dataset dari paths dan labels
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    # Shuffle sebelum decode agar urutan file random
    if should_shuffle:
        buffer_size = min(len(entries), 10_000)  # buffer realistis untuk RAM
        ds = ds.shuffle(buffer_size=buffer_size, seed=cfg.data.random_seed)

    # Decode: baca file dari disk secara paralel
    decode_fn = build_decode_fn()
    ds = ds.map(decode_fn, num_parallel_calls=tf.data.AUTOTUNE)

    # Preprocess: resize + normalisasi
    preprocess_fn = build_preprocessing_fn(image_size=cfg.data.image_size)
    ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)

    # Augmentasi (hanya training)
    if augment and cfg.data.use_augmentation:
        augment_fn = build_augmentation_fn()
        ds = ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch dan prefetch
    ds = ds.batch(cfg.data.batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def get_dataset_info(entries: List[ImageEntry], cfg) -> None:
    """
    Print informasi dataset untuk debugging.
    Dipanggil di notebook setelah build_dataset untuk verifikasi.
    """
    n = len(entries)
    label_counts = {}
    for _, label in entries:
        label_counts[label] = label_counts.get(label, 0) + 1

    n_batches = (n + cfg.data.batch_size - 1) // cfg.data.batch_size

    print(f"  Total samples : {n}")
    print(f"  Batch size    : {cfg.data.batch_size}")
    print(f"  Total batches : {n_batches}")
    print(f"  Image size    : {cfg.data.image_size}")
    for label, count in sorted(label_counts.items()):
        cls = cfg.data.class_names[label]
        print(f"  {cls:<12}  : {count} samples (label={label})")