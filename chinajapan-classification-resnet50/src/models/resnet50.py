import tensorflow as tf


def build_resnet50_model(input_shape=(224, 224, 3), num_classes=2, freeze_base_model=True, dropout_rate=0.3):
    """Build a binary classification model using ResNet50."""
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        pooling="avg",
    )

    if freeze_base_model:
        base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )
    return model


def unfreeze_model(model, unfreeze_from_layer="conv5_block1_1_conv"):
    """Unfreeze layers for fine-tuning."""
    model.trainable = True
    for layer in model.layers:
        if layer.name == unfreeze_from_layer:
            layer.trainable = True
            break

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )
    return model
