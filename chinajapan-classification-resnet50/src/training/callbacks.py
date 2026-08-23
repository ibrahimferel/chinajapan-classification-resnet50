import tensorflow as tf


def build_callbacks(checkpoint_path="checkpoints/best_model.keras", monitor="val_accuracy", mode="max"):
    """Create training callbacks."""
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        mode=mode,
        patience=5,
        restore_best_weights=True,
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor=monitor,
        mode=mode,
        save_best_only=True,
        save_weights_only=False,
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=monitor,
        mode=mode,
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    return [early_stop, checkpoint, reduce_lr]
