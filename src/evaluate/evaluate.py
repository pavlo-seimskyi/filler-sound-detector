from typing import *

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch import Tensor, float32


def _evaluate(
    y_test: Iterable[Any], y_pred_proba: Iterable[float], threshold: float
) -> None:
    """
    Evaluate model performance by computing metrics
        and plotting precision-recall curve and confusion matrix.
    :param y_test: true target variable
    :param y_pred_proba: prediction probabilities for the positive class (0.0 to 1.0)
    :param threshold: cutoff point to determine the final prediction
    :return: None
    """
    y_test, y_pred_proba = dtype_to_tensor(y_test), dtype_to_tensor(y_pred_proba)
    y_test, y_pred_proba = y_test.cpu(), y_pred_proba.cpu()

    metrics = calculate_metrics(y_test, y_pred_proba, threshold)
    print(", ".join([f"{name}: {round(value, 3)}" for name, value in metrics.items()]))
    y_pred = convert_target_to_binary(y_pred_proba, threshold)

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    plt.tight_layout(h_pad=1, w_pad=9)
    plot_confusion_matrix(y_test, y_pred, ax=axs[0])
    plot_precision_recall_curve(y_test, y_pred_proba, threshold, ax=axs[1])


def dtype_to_tensor(x: Iterable[Any]) -> Tensor:
    """Make sure data type is numpy array."""
    if isinstance(x, Tensor):
        return x
    elif isinstance(x, (np.ndarray, list, tuple)):
        return Tensor(x)
    else:
        raise Exception(f"The data type must be a sequence, got {type(x)} instead.")


def calculate_metrics(
    y_test: Tensor, y_pred_proba: Tensor, threshold: float
) -> Dict[str, float]:
    """Calculate all metrics."""
    y_test, y_pred_proba = y_test.cpu(), y_pred_proba.cpu()
    y_pred = convert_target_to_binary(y_pred_proba, threshold=threshold)
    return {
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "avg_precision_score": average_precision_score(y_test, y_pred_proba),
    }


def convert_target_to_binary(y: Tensor, threshold: float) -> np.ndarray:
    """Apply the cutoff threshold, making predictions binary."""
    y = dtype_to_tensor(y).clone()
    return (y >= threshold).to(float32)


def plot_precision_recall_curve(
    y_test: Tensor, y_pred_proba: Tensor, threshold: float, ax
) -> None:
    """Plot a precision-recall curve with a marker pointing
    to the current cutoff threshold on axis."""
    PrecisionRecallDisplay.from_predictions(y_test, y_pred_proba, ax=ax)
    ax.set_title("Precision-recall curve")
    y_pred = convert_target_to_binary(y_pred_proba, threshold)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    ax.plot(
        recall,
        precision,
        marker="x",
        color="r",
        markersize=15,
        label="chosen threshold",
    )
    ax.legend()


def plot_confusion_matrix(y_test, y_pred, ax) -> None:
    """Plot a confusion matrix on axis."""
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap="Blues")
    ax.grid(False)
    ax.set_title("Confusion matrix")
