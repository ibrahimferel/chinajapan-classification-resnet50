import tensorflow as tf


def resize_and_rescale(image, label, image_size=(224, 224)):
    """Resize image and apply ResNet50 preprocessing."""
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image, label


def load_and_preprocess(image_path, label, image_size=(224, 224)):
    """Decode and preprocess a single image."""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, image_size)
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image, label
