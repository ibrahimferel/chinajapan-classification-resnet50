"""
src/evaluation/metrics.py

Evaluasi model pada test set.
Menghasilkan semua metrik yang diperlukan untuk tabel paper.

Metrik yang dihasilkan:
    - accuracy
    - precision (weighted)
    - recall (weighted)
    - F1-score (weighted)
    - classification report (per class)
    - confusion matrix (numpy array)

Weighted average digunakan agar adil untuk class imbalance.

Usage:
    from src.evaluation.metrics import evaluate
    results = evaluate(model, test_ds, cfg)
    print(results["accuracy"])
    print(results["classification_report"])
"""

from typing import Any, Dict

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate(model: tf.keras.Model, test_ds, cfg) -> Dict[str, Any]:
    """
    Evaluasi model pada test set dan kembalikan semua metrik.

    Args:
        model:   Trained Keras model
        test_ds: tf.data.Dataset test (dari build_dataset)
        cfg:     Config object

    Returns:
        Dict berisi semua metrik:
            "accuracy"               : float
            "precision"              : float  (weighted)
            "recall"                 : float  (weighted)
            "f1"                     : float  (weighted)
            "classification_report"  : str
            "confusion_matrix"       : np.ndarray (2×2)
            "y_true"                 : np.ndarray
            "y_pred"                 : np.ndarray
            "y_prob"                 : np.ndarray  (probabilitas sigmoid)
    """
    print("🔍 Mengevaluasi model pada test set...")

    # ── Kumpulkan prediksi ───────────────────────────────────────────────
    y_true_list = []
    y_prob_list = []

    for batch_images, batch_labels in test_ds:
        probs = model(batch_images, training=False).numpy()  # shape (B, 1)
        y_prob_list.extend(probs.flatten())
        y_true_list.extend(batch_labels.numpy())

    y_true = np.array(y_true_list, dtype=int)
    y_prob = np.array(y_prob_list)

    # Threshold 0.5 untuk binary classification
    y_pred = (y_prob >= 0.5).astype(int)

    # ── Hitung metrik ────────────────────────────────────────────────────
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        target_names=cfg.data.class_names,
        zero_division=0,
    )

    # ── Print hasil ──────────────────────────────────────────────────────
    print(f"\n📊 Hasil Evaluasi — {cfg.paths.experiment_name}")
    print(f"   Accuracy  : {acc:.4f}")
    print(f"   Precision : {prec:.4f}")
    print(f"   Recall    : {rec:.4f}")
    print(f"   F1-score  : {f1:.4f}")
    print(f"\nClassification Report:")
    print(report)

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "classification_report": report,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }