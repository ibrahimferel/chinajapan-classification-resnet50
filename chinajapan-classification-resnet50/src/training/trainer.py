import json
from pathlib import Path

import tensorflow as tf


class ModelTrainer:
    def __init__(self, model, train_dataset, val_dataset, callbacks=None):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.callbacks = callbacks or []

    def fit(self, epochs=10, verbose=1):
        history = self.model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=epochs,
            callbacks=self.callbacks,
            verbose=verbose,
        )
        return history

    def save_history(self, history, output_path="results/history.json"):
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            json.dump(history.history, f, indent=2)

    def save_model(self, model_path="checkpoints/final_model.keras"):
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
