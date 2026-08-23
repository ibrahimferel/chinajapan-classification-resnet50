from pathlib import Path


def inspect_dataset(data_dir: str):
    """Return a summary of dataset structure and class counts."""
    data_path = Path(data_dir)
    summary = {}

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    for class_dir in sorted(data_path.iterdir()):
        if class_dir.is_dir():
            files = [p.name for p in class_dir.iterdir() if p.is_file()]
            summary[class_dir.name] = len(files)

    return summary
