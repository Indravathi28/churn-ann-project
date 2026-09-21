"""Evaluation helpers. These work on predicted probabilities, so they are
independent of the model library (Keras, scikit-learn, ...)."""
import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)


def best_threshold(y_true, y_prob, low=0.10, high=0.90, step=0.01) -> float:
    """Return the probability threshold that maximises F1 on the given data.

    Use the *validation* set here, never the test set.
    """
    grid = np.arange(low, high + step, step)
    scores = [f1_score(y_true, (y_prob >= t).astype(int), zero_division=0) for t in grid]
    return float(round(grid[int(np.argmax(scores))], 2))


def compute_metrics(y_true, y_prob, threshold=0.5) -> dict:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def text_report(y_true, y_prob, threshold=0.5) -> str:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return classification_report(
        y_true, y_pred, target_names=["Stayed (0)", "Churned (1)"], digits=3
    )


def plot_training_history(history, path):
    """Loss / AUC curves for train vs validation (a Keras History object)."""
    h = history.history
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, title in zip(axes, ["loss", "accuracy", "auc"],
                              ["Loss", "Accuracy", "AUC"]):
        ax.plot(h[key], label="train")
        ax.plot(h[f"val_{key}"], label="validation")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_confusion_matrix(cm, path, title="Confusion matrix (test set)"):
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    im = ax.imshow(cm, cmap="Blues")
    labels = ["Stayed", "Churned"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=13)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_roc_curves(curves: dict, path):
    """curves = {"ANN": (y_true, y_prob), "Logistic regression": (...)}"""
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for name, (y_true, y_prob) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve (test set)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
