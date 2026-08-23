"""
src/training/callbacks.py

Keras callbacks untuk training.

Tiga callback yang digunakan:
    1. ModelCheckpoint  — simpan model terbaik berdasarkan val_loss
    2. EarlyStopping    — hentikan training jika val_loss tidak membaik
    3. ReduceLROnPlateau — turunkan LR jika val_loss stagnan

Kenapa monitor val_loss bukan val_accuracy?
    val_loss lebih sensitif dan lebih stabil sebagai signal.
    val_accuracy bisa stagnan meski loss masih turun (terutama dengan
    class imbalance). Untuk paper, semua metrik tetap dilaporkan,
    tapi checkpoint dan stopping menggunakan val_loss.

Usage:
    from src.training.callbacks import build_callbacks
    callbacks = build_callbacks(cfg, stage=1)
    model.fit(..., callbacks=callbacks)
"""

import os
from typing import List

from tensorflow.keras.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)


def build_callbacks(cfg, stage: int = 1) -> List[Callback]:
    """
    Bangun list callback untuk satu stage training.

    Args:
        cfg:   Config object dari configs/config.py
        stage: 1 untuk frozen backbone training,
               2 untuk fine-tuning.
               Nama checkpoint file berbeda per stage.

    Returns:
        List of Keras Callback objects
    """
    checkpoint_dir = cfg.paths.experiment_checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"best_model_stage{stage}.keras",  # .keras format lebih modern dari .h5
    )

    callbacks = [
        # ── 1. Simpan model terbaik ────────────────────────────────────
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,       # hanya simpan jika ada improvement
            save_weights_only=False,   # simpan arsitektur + weights
            verbose=1,
        ),

        # ── 2. Hentikan jika val_loss tidak membaik ────────────────────
        EarlyStopping(
            monitor="val_loss",
            patience=cfg.training.early_stopping_patience,
            restore_best_weights=True,  # kembalikan ke checkpoint terbaik saat stop
            verbose=1,
        ),

        # ── 3. Turunkan LR jika val_loss stagnan ──────────────────────
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=cfg.training.reduce_lr_factor,   # LR baru = LR lama × factor
            patience=cfg.training.reduce_lr_patience,
            min_lr=cfg.training.min_lr,
            verbose=1,
        ),
    ]

    print(f"Callbacks siap (stage {stage}):")
    print(f"  Checkpoint → {checkpoint_path}")
    print(f"  EarlyStopping patience : {cfg.training.early_stopping_patience}")
    print(f"  ReduceLR patience      : {cfg.training.reduce_lr_patience}")

    return callbacks