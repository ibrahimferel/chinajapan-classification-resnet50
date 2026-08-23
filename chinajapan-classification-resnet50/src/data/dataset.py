from pathlib import Path

import tensorflow as tf

from src.data.preprocessing import load_and_preprocess


def build_dataset(image_paths, labels, batch_size=32, shuffle=True, image_size=(224, 224)):
    """Build tf.data dataset from file paths and labels."""
    labels = tf.convert_to_tensor(labels)
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=max(1000, len(image_paths)))

    ds = ds.map(lambda x, y: load_and_preprocess(x, y, image_size), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def load_class_names(data_dir: str):
    """List class names from dataset directory."""
    return sorted([p.name for p in Path(data_dir).iterdir() if p.is_dir()])
