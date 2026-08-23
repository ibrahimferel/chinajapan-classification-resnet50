import json
from datetime import datetime
from pathlib import Path


class ExperimentLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log(self, payload: dict):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = self.log_dir / f"experiment_{timestamp}.json"
        with log_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return str(log_path)
