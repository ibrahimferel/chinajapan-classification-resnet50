"""
src/data/preprocessing.py

Preprocessing standar untuk ResNet50.
Tidak ada CLAHE, tidak ada patch extraction, tidak ada custom enhancement.

Yang dilakukan:
    1. Decode file gambar → tensor
    2. Resize ke 224×224 (direct resize, tanpa padding)
    3. Cast ke float32
    4. preprocess_input() dari keras — normalisasi sesuai ImageNet statistics

Kenapa direct resize tanpa padding?
    Untuk baseline, direct resize adalah pilihan paling sederhana dan
    paling banyak dilaporkan di literatur transfer learning standar.
    Padding bisa ditambahkan sebagai ablation tersendiri.

Kenapa preprocess_input() bukan manual normalization?
    ResNet50 pretrained ImageNet menggunakan normalisasi spesifik (bukan [0,1]
    dan bukan zero-mean/unit-variance umum). preprocess_input() memastikan
    distribusi input sesuai dengan yang diharapkan pretrained weights.

    Secara konkret, preprocess_input() melakukan:
        - Konversi RGB → BGR
        - Substraksi mean per channel [103.939, 116.779, 123.68]
        (nilai mean ImageNet training set dalam format BGR)

Usage:
    from src.data.preprocessing import build_preprocessing_fn
    preprocess_fn = build_preprocessing_fn(image_size=(224, 224))

    # Dalam tf.data pipeline:
    dataset = dataset.map(preprocess_fn)
"""

from typing import Callable, Tuple

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input


def build_preprocessing_fn(
    image_size: Tuple[int, int] = (224, 224),
) -> Callable:
    """
    Buat fungsi preprocessing untuk digunakan dalam tf.data pipeline.

    Fungsi yang dihasilkan menerima (image_tensor, label) dan
    mengembalikan (preprocessed_image_tensor, label).

    Args:
        image_size: Target (height, width). Default (224, 224).

    Returns:
        Callable yang kompatibel dengan dataset.map()
    """
    target_h, target_w = image_size

    def preprocess_fn(
        image: tf.Tensor,
        label: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Preprocessing pipeline per gambar.

        Steps:
            1. Resize ke target_h × target_w (bilinear interpolation)
            2. Cast ke float32
            3. preprocess_input() — normalisasi ImageNet

        Args:
            image: Tensor gambar, shape (H, W, 3), dtype uint8 atau float.
            label: Label tensor (tidak diubah).

        Returns:
            Tuple (preprocessed_image, label)
        """
        # Step 1: Resize — bilinear adalah default dan cocok untuk downscaling/upscaling
        image = tf.image.resize(image, [target_h, target_w])

        # Step 2: Cast ke float32 sebelum preprocess_input
        image = tf.cast(image, tf.float32)

        # Step 3: ResNet50 ImageNet normalization
        # preprocess_input mengharapkan float32 dengan range [0, 255]
        image = preprocess_input(image)

        return image, label

    return preprocess_fn


def build_decode_fn() -> Callable:
    """
    Buat fungsi untuk decode file path → image tensor.
    Dipanggil sebelum preprocess_fn dalam pipeline.

    Returns:
        Callable yang menerima (file_path, label) dan mengembalikan
        (image_tensor uint8, label)
    """
    def decode_fn(
        file_path: tf.Tensor,
        label: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Baca file gambar dari disk → tensor RGB uint8.

        Mendukung JPEG dan PNG.
        Output selalu 3 channel (RGB) — grayscale di-convert otomatis.
        """
        raw = tf.io.read_file(file_path)
        # decode_image: auto-detect format, channels=3 paksa RGB
        image = tf.image.decode_image(
            raw,
            channels=3,
            expand_animations=False,  # abaikan GIF frame
        )
        # Set shape secara eksplisit karena decode_image tidak set shape otomatis
        image.set_shape([None, None, 3])
        return image, label

    return decode_fn