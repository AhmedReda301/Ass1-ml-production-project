"""Model evaluation: compute and log classification metrics."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from production_app.logging_config import get_logger

logger = get_logger(__name__)


def evaluate_model(
    pipeline: Pipeline,
    x_test: np.ndarray,
    y_test: np.ndarray,
    target_names: list[str],
) -> dict[str, float]:
    """Evaluate a fitted pipeline on the test set, log results, return metrics."""
    y_pred = pipeline.predict(x_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }

    logger.info(
        "Evaluation: accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f",
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
    )
    logger.info("Confusion matrix:\n%s", confusion_matrix(y_test, y_pred))
    logger.info(
        "Classification report:\n%s",
        classification_report(y_test, y_pred, target_names=target_names),
    )
    return metrics
