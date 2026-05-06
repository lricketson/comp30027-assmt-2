import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np


def plot_model_confusion_matrix(
    y_true, y_pred, classes, title="Confusion Matrix", save_path=None
):
    """
    A universal confusion matrix plotter that works for any model.
    """
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=classes)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
    )

    plt.title(title, fontsize=16, pad=15)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved matrix to {save_path}")

    plt.show()
