"""
src/utils/logger.py

Menyimpan seluruh output eksperimen ke disk secara terstruktur.
Dipanggil di akhir setiap notebook setelah evaluation selesai.

Struktur output per eksperimen:
    results/
    └── exp_a_frozen/
        ├── config.json               ← konfigurasi lengkap
        ├── history_stage1.json       ← training history stage 1
        ├── history_stage2.json       ← training history stage 2 (jika ada)
        ├── metrics.json              ← accuracy, precision, recall, F1
        ├── classification_report.txt ← laporan per kelas
        ├── confusion_matrix.png      ← disimpan oleh confusion_matrix.py
        └── loss_curve.png            ← disimpan oleh visualization.py

Usage:
    from src.utils.logger import save_experiment
    save_experiment(cfg, results, history_s1, history_s2)
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np


def _make_serializable(obj: Any) -> Any:
    """
    Rekursif konversi objek ke tipe yang JSON-serializable.
    Menangani numpy types dan nested dict/list.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    return obj


def _save_json(data: Any, path: str) -> None:
    """Simpan dictionary ke file JSON dengan pretty print."""
    with open(path, "w") as f:
        json.dump(_make_serializable(data), f, indent=2)


def save_experiment(
    cfg,
    results: Dict[str, Any],
    history_stage1: Optional[Any] = None,
    history_stage2: Optional[Any] = None,
) -> str:
    """
    Simpan semua output eksperimen ke satu folder terstruktur.

    Args:
        cfg:            Config object dari configs/config.py
        results:        Dict dari src/evaluation/metrics.py:
                            {
                                "accuracy": float,
                                "precision": float,
                                "recall": float,
                                "f1": float,
                                "classification_report": str,
                                "confusion_matrix": np.ndarray
                            }
        history_stage1: Keras History object dari stage 1 training.
                        Bisa None jika hanya ada satu stage.
        history_stage2: Keras History object dari stage 2 fine-tuning.
                        Bisa None jika tidak ada fine-tuning.

    Returns:
        Path ke folder eksperimen yang dibuat.
    """
    output_dir = cfg.paths.experiment_results_dir
    os.makedirs(output_dir, exist_ok=True)

    # 1. Simpan config
    cfg.save(os.path.join(output_dir, "config.json"))

    # 2. Simpan training history stage 1
    if history_stage1 is not None:
        _save_json(
            history_stage1.history,
            os.path.join(output_dir, "history_stage1.json"),
        )
        print(f"History stage 1 disimpan → {output_dir}/history_stage1.json")

    # 3. Simpan training history stage 2 (fine-tuning)
    if history_stage2 is not None:
        _save_json(
            history_stage2.history,
            os.path.join(output_dir, "history_stage2.json"),
        )
        print(f"History stage 2 disimpan → {output_dir}/history_stage2.json")

    # 4. Simpan metrics numerik (tanpa confusion matrix dan report string)
    metrics_to_save = {
        k: v
        for k, v in results.items()
        if k not in ("classification_report", "confusion_matrix")
    }
    metrics_to_save["timestamp"] = datetime.now().isoformat()
    metrics_to_save["experiment_name"] = cfg.paths.experiment_name
    _save_json(metrics_to_save, os.path.join(output_dir, "metrics.json"))
    print(f"Metrics disimpan → {output_dir}/metrics.json")

    # 5. Simpan classification report sebagai plain text
    if "classification_report" in results:
        report_path = os.path.join(output_dir, "classification_report.txt")
        with open(report_path, "w") as f:
            f.write(f"Experiment : {cfg.paths.experiment_name}\n")
            f.write(f"Timestamp  : {datetime.now().isoformat()}\n")
            f.write("-" * 60 + "\n")
            f.write(results["classification_report"])
        print(f"Classification report disimpan → {report_path}")

    print(f"\n📁 Semua output eksperimen tersimpan di: {output_dir}")
    return output_dir