"""
src/evaluation/visualization.py

Semua fungsi plotting untuk training history dan evaluation results.

Menghasilkan file gambar ke results/experiment_name/:
    - loss_curve.png          ← training + validation loss per epoch
    - accuracy_curve.png      ← training + validation accuracy per epoch
    - confusion_matrix.png    ← confusion matrix heatmap

Usage:
    from src.evaluation.visualization import plot_history, plot_confusion_matrix
    plot_history(history_s1, history_s2, cfg)
    plot_confusion_matrix(results["confusion_matrix"], cfg)
"""

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tensorflow.keras.callbacks import History


def plot_history(
    history_stage1: Optional[History],
    history_stage2: Optional[History],
    cfg,
) -> None:
    """
    Plot training dan validation loss + accuracy.

    Jika ada dua stage, keduanya digabung dalam satu grafik
    dengan garis vertikal pemisah di antara stage 1 dan stage 2.

    Args:
        history_stage1: Keras History dari Stage 1 (bisa None)
        history_stage2: Keras History dari Stage 2 (bisa None)
        cfg:            Config object
    """
    output_dir = cfg.paths.experiment_results_dir
    os.makedirs(output_dir, exist_ok=True)

    # Gabungkan history stage 1 dan stage 2
    def _get(h, key):
        return h.history.get(key, []) if h else []

    train_loss = _get(history_stage1, "loss") + _get(history_stage2, "loss")
    val_loss   = _get(history_stage1, "val_loss") + _get(history_stage2, "val_loss")
    train_acc  = _get(history_stage1, "accuracy") + _get(history_stage2, "accuracy")
    val_acc    = _get(history_stage1, "val_accuracy") + _get(history_stage2, "val_accuracy")

    epochs = range(1, len(train_loss) + 1)
    s1_len = len(_get(history_stage1, "loss"))  # untuk garis pemisah

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Training History — {cfg.paths.experiment_name}",
        fontsize=13,
        fontweight="bold",
    )

    for ax, train_vals, val_vals, metric in zip(
        axes,
        [train_loss, train_acc],
        [val_loss, val_acc],
        ["Loss", "Accuracy"],
    ):
        ax.plot(epochs, train_vals, label=f"Train {metric}", linewidth=1.5)
        ax.plot(epochs, val_vals,   label=f"Val {metric}",   linewidth=1.5, linestyle="--")

        # Garis pemisah stage 1 / stage 2
        if history_stage2 and s1_len > 0:
            ax.axvline(
                x=s1_len + 0.5,
                color="gray",
                linestyle=":",
                linewidth=1,
                label="Stage 1 / Stage 2",
            )

        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Training curves disimpan → {save_path}")


def plot_confusion_matrix(cm: np.ndarray, cfg) -> None:
    """
    Plot confusion matrix sebagai heatmap dan simpan ke file.

    Args:
        cm:  numpy array (2×2) dari sklearn.metrics.confusion_matrix
        cfg: Config object
    """
    output_dir = cfg.paths.experiment_results_dir
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=cfg.data.class_names,
        yticklabels=cfg.data.class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(
        f"Confusion Matrix — {cfg.paths.experiment_name}",
        fontweight="bold",
    )

    plt.tight_layout()
    save_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Confusion matrix disimpan → {save_path}")