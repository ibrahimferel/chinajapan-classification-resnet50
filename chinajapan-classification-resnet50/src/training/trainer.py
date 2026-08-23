"""
src/training/trainer.py

Mengelola training loop untuk kedua stage.

Trainer menyederhanakan notebook: satu panggilan fungsi untuk
train_stage1() dan train_stage2(), tanpa logic training di notebook.

Stage 1 (Frozen):
    - Backbone frozen, hanya head yang dilatih
    - LR: 1e-3, epochs: cfg.training.stage1_epochs
    - Simpan history untuk plotting

Stage 2 (Fine-tuning):
    - Backbone sebagian/seluruhnya di-unfreeze
    - LR: 1e-5 (jauh lebih kecil)
    - Lanjut dari model terbaik Stage 1 (bukan dari scratch)

Usage:
    from src.training.trainer import Trainer
    trainer = Trainer(model, cfg)

    history_s1 = trainer.train_stage1(train_ds, val_ds)
    history_s2 = trainer.train_stage2(train_ds, val_ds)  # opsional
"""

from typing import Optional

from tensorflow.keras import Model
from tensorflow.keras.callbacks import History

from src.models.resnet50 import unfreeze_for_finetuning
from src.training.callbacks import build_callbacks


class Trainer:
    """
    Wrapper training untuk ResNet50 classifier.

    Attributes:
        model: Keras Model dari resnet50.py
        cfg:   Config object
    """

    def __init__(self, model: Model, cfg) -> None:
        self.model = model
        self.cfg = cfg

    def train_stage1(
        self,
        train_ds,
        val_ds,
        class_weight: Optional[dict] = None,
    ) -> History:
        """
        Stage 1: train classification head dengan backbone frozen.

        Args:
            train_ds:     tf.data.Dataset untuk training
            val_ds:       tf.data.Dataset untuk validasi
            class_weight: Optional dict {class_idx: weight} dari split.get_class_weights()
                          Gunakan jika dataset imbalanced.

        Returns:
            Keras History object (history.history berisi dict loss/acc per epoch)
        """
        print(f"\n{'='*50}")
        print(f"  Stage 1: Training classification head")
        print(f"  Epochs : {self.cfg.training.stage1_epochs}")
        print(f"  LR     : {self.cfg.training.stage1_lr}")
        print(f"{'='*50}\n")

        callbacks = build_callbacks(self.cfg, stage=1)

        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.cfg.training.stage1_epochs,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1,
        )

        print(f"\nStage 1 selesai.")
        self._print_best_epoch(history, stage=1)

        return history

    def train_stage2(
        self,
        train_ds,
        val_ds,
        class_weight: Optional[dict] = None,
    ) -> Optional[History]:
        """
        Stage 2: fine-tuning backbone (sebagian atau seluruhnya).

        Hanya dijalankan jika fine_tune_from_layer != -1.
        Model di-unfreeze dulu, kemudian dilatih dengan LR kecil.

        Args:
            train_ds:     tf.data.Dataset untuk training
            val_ds:       tf.data.Dataset untuk validasi
            class_weight: Optional class weights

        Returns:
            Keras History object, atau None jika fine-tuning tidak aktif
        """
        if self.cfg.training.fine_tune_from_layer == -1:
            print("fine_tune_from_layer = -1 → Stage 2 dilewati.")
            return None

        print(f"\n{'='*50}")
        print(f"  Stage 2: Fine-tuning backbone")
        print(f"  Unfreeze dari layer : {self.cfg.training.fine_tune_from_layer}")
        print(f"  Epochs              : {self.cfg.training.stage2_epochs}")
        print(f"  LR                  : {self.cfg.training.stage2_lr}")
        print(f"{'='*50}\n")

        # Unfreeze dan recompile dengan LR kecil
        self.model = unfreeze_for_finetuning(self.model, self.cfg)

        callbacks = build_callbacks(self.cfg, stage=2)

        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.cfg.training.stage2_epochs,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1,
        )

        print(f"\nStage 2 selesai.")
        self._print_best_epoch(history, stage=2)

        return history

    @staticmethod
    def _print_best_epoch(history: History, stage: int) -> None:
        """Print epoch dengan val_loss terbaik dari history."""
        val_losses = history.history.get("val_loss", [])
        if not val_losses:
            return
        best_epoch = val_losses.index(min(val_losses)) + 1
        best_val_loss = min(val_losses)
        best_val_acc = history.history.get("val_accuracy", [])[best_epoch - 1]
        print(f"  Stage {stage} best → epoch {best_epoch}  "
              f"val_loss={best_val_loss:.4f}  val_acc={best_val_acc:.4f}")