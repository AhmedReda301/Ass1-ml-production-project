"""Saving, loading, and using the fitted pipeline for predictions."""

from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

from production_app.config import settings
from production_app.logging_config import get_logger

logger = get_logger(__name__)

CLASS_NAMES = {0: "malignant", 1: "benign"}


def save_model(pipeline: Pipeline, path: Path = settings.model_path) -> Path:
    """Persist the fitted pipeline (preprocessing + model) to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    logger.info("Model saved to %s", path)
    return path


def load_model(path: Path = settings.model_path) -> Pipeline:
    """Load a previously saved pipeline; raise a clear error if it is missing."""
    if not path.exists():
        logger.error("Model file not found: %s", path)
        raise FileNotFoundError(
            f"No saved model at {path}. Run the training workflow first."
        )
    pipeline = joblib.load(path)
    logger.info("Model loaded from %s", path)
    return pipeline


def predict(pipeline: Pipeline, features: object) -> list[str]:
    """Validate the input and return a class name for each sample."""
    try:
        x = np.asarray(features, dtype=float)
    except (TypeError, ValueError) as exc:
        logger.error("Prediction input is not numeric")
        raise ValueError("Prediction input must contain only numbers") from exc

    if x.ndim == 1:
        x = x.reshape(1, -1)

    expected = pipeline.n_features_in_
    if x.ndim != 2 or x.shape[1] != expected:
        logger.error("Invalid input shape %s, expected %d features", x.shape, expected)
        raise ValueError(f"Expected {expected} features per sample, got shape {x.shape}")

    if np.isnan(x).any():
        logger.error("Prediction input contains NaN values")
        raise ValueError("Prediction input must not contain NaN values")

    names = [CLASS_NAMES[int(label)] for label in pipeline.predict(x)]
    logger.info("Predicted %d sample(s): %s", len(names), names)
    return names