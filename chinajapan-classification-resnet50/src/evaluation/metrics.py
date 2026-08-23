from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def compute_metrics(y_true, y_pred, labels=None):
    """Compute accuracy, precision, recall, and F1 for classification."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    if labels is not None:
        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels)
    return metrics


def classification_report(y_true, y_pred, labels=None, target_names=None):
    """Generate a sklearn-style classification report summary."""
    from sklearn.metrics import classification_report as sk_report

    return sk_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
    )
