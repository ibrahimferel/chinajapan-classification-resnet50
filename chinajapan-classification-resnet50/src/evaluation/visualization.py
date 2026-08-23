from pathlib import Path

import matplotlib.pyplot as plt


def plot_training_history(history, save_path="results/training_history.png"):
    """Plot training and validation loss/accuracy curves."""
    history_dict = history.history

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history_dict.get("loss", []), label="train_loss")
    plt.plot(history_dict.get("val_loss", []), label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history_dict.get("accuracy", []), label="train_accuracy")
    plt.plot(history_dict.get("val_accuracy", []), label="val_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    output = Path(save_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def plot_prediction_samples(images, labels, predictions, class_names, save_path="results/prediction_samples.png"):
    """Plot sample predictions for inspection."""
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < len(images):
            ax.imshow(images[i])
            true_label = class_names[labels[i]]
            pred_label = class_names[predictions[i]]
            ax.set_title(f"True: {true_label}\nPred: {pred_label}")
            ax.axis("off")

    output = Path(save_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()
