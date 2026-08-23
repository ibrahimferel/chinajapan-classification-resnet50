"""
src/models/resnet50.py

Build ResNet50-based classifier untuk Chinese vs Japanese Painting.

Architecture:
    Input (224 × 224 × 3)
        ↓
    ResNet50 backbone (pretrained ImageNet, include_top=False)
        ↓
    GlobalAveragePooling2D          ← 2048-dim feature vector
        ↓
    Dropout(dropout_rate)
        ↓
    Dense(1, activation='sigmoid')  ← binary output

Kenapa sigmoid + Dense(1) bukan softmax + Dense(2)?
    Keduanya menghasilkan hasil yang ekuivalen untuk binary classification,
    tapi sigmoid + Dense(1) lebih idiomatis dan sedikit lebih efisien.
    Loss yang digunakan: BinaryCrossentropy.

Kenapa GlobalAveragePooling, bukan Flatten?
    GAP menghasilkan vektor 2048-dim terlepas dari ukuran input.
    Flatten menghasilkan vektor besar yang bergantung spatial map size
    dan rentan overfitting. GAP adalah pilihan standar untuk transfer learning.

Fine-tuning strategy:
    Stage 1: backbone.trainable = False
             Hanya classification head yang belajar.
             LR: 1e-3 (relatif tinggi, aman karena backbone frozen)

    Stage 2: Unfreeze dari layer fine_tune_from_layer ke atas
             Seluruh model belajar dengan LR kecil (1e-5)
             untuk menghindari catastrophic forgetting.

Usage:
    from src.models.resnet50 import build_model, unfreeze_for_finetuning
    model = build_model(cfg)                    # Stage 1
    model = unfreeze_for_finetuning(model, cfg) # Stage 2
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam


def build_model(cfg) -> Model:
    """
    Bangun ResNet50 classifier dengan backbone frozen (Stage 1).

    Backbone dibekukan agar classification head bisa dilatih dari nol
    tanpa merusak pretrained ImageNet features.

    Args:
        cfg: Config object dari configs/config.py

    Returns:
        Compiled Keras Model, siap untuk model.fit()
    """
    # ── Backbone ────────────────────────────────────────────────────────
    backbone = ResNet50(
        weights=cfg.model.pretrained_weights,  # "imagenet"
        include_top=False,                     # hapus FC layer asli
        input_shape=(*cfg.data.image_size, 3), # (224, 224, 3)
    )

    # Bekukan semua layer backbone untuk Stage 1
    backbone.trainable = False

    # ── Classification head ─────────────────────────────────────────────
    inputs = backbone.input
    x = backbone.output
    x = GlobalAveragePooling2D(name="gap")(x)
    x = Dropout(cfg.model.dropout_rate, name="head_dropout")(x)
    outputs = Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="resnet50_classifier")

    # ── Compile ─────────────────────────────────────────────────────────
    model.compile(
        optimizer=Adam(learning_rate=cfg.training.stage1_lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    # Print summary ringkas
    total_params = model.count_params()
    trainable_params = sum(
        tf.size(w).numpy() for w in model.trainable_weights
    )
    print(f"Model built.")
    print(f"  Total params     : {total_params:,}")
    print(f"  Trainable params : {trainable_params:,} (head only)")
    print(f"  Frozen params    : {total_params - trainable_params:,} (backbone)")

    return model


def unfreeze_for_finetuning(model: Model, cfg) -> Model:
    """
    Unfreeze backbone untuk Stage 2 fine-tuning.
    Panggil fungsi ini setelah Stage 1 selesai.

    Backbone layer yang di-unfreeze ditentukan oleh
    cfg.training.fine_tune_from_layer:
        -1 → tidak ada fine-tuning (kembalikan model tanpa perubahan)
        143 → unfreeze layer 143+ (partial, conv5_block)
        0 → unfreeze semua (full fine-tuning)

    LR dikecilkan ke stage2_lr untuk menghindari catastrophic forgetting.

    Args:
        model: Model hasil build_model() setelah Stage 1 selesai
        cfg:   Config object

    Returns:
        Model yang sudah di-recompile dengan backbone sebagian/seluruhnya trainable
    """
    fine_tune_from = cfg.training.fine_tune_from_layer

    if fine_tune_from == -1:
        print("fine_tune_from_layer = -1 → backbone tetap frozen, tidak ada Stage 2.")
        return model

    # Cari backbone layer (layer pertama yang bukan Input)
    backbone = model.layers[1]  # ResNet50 adalah layer index 1

    # Aktifkan trainable untuk backbone secara keseluruhan dulu
    backbone.trainable = True

    if fine_tune_from > 0:
        # Bekukan layer sebelum fine_tune_from_layer
        for layer in backbone.layers[:fine_tune_from]:
            layer.trainable = False

        n_frozen = fine_tune_from
        n_unfrozen = len(backbone.layers) - fine_tune_from
        print(f"Partial fine-tuning:")
        print(f"  Frozen  : {n_frozen} layers (0 to {fine_tune_from - 1})")
        print(f"  Unfrozen: {n_unfrozen} layers ({fine_tune_from} onwards)")
    else:
        # fine_tune_from = 0: semua layer trainable
        print(f"Full fine-tuning: semua {len(backbone.layers)} backbone layers trainable")

    # Recompile dengan LR kecil
    model.compile(
        optimizer=Adam(learning_rate=cfg.training.stage2_lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    trainable_params = sum(
        tf.size(w).numpy() for w in model.trainable_weights
    )
    total_params = model.count_params()
    print(f"  Trainable params : {trainable_params:,}")
    print(f"  Frozen params    : {total_params - trainable_params:,}")
    print(f"  Stage 2 LR      : {cfg.training.stage2_lr}")

    return model


def load_model(checkpoint_path: str) -> Model:
    """
    Load model dari checkpoint untuk inference atau lanjut training.

    Args:
        checkpoint_path: Path ke file .keras atau .h5

    Returns:
        Loaded Keras Model
    """
    model = tf.keras.models.load_model(checkpoint_path)
    print(f"Model loaded dari: {checkpoint_path}")
    return model