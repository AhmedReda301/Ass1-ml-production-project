"""Dataset loading and train/test splitting."""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch

from production_app.config import settings
from production_app.logging_config import get_logger

logger = get_logger(__name__)


def load_dataset() -> Bunch:
    """Load the Breast Cancer Wisconsin dataset and log a quick summary."""
    data = load_breast_cancer()
    logger.info(
        "Dataset loaded: %d samples, %d features",
        data.data.shape[0],
        data.data.shape[1],
    )
    logger.info("Target names: %s", data.target_names.tolist())
    logger.info("Class distribution: %s", np.bincount(data.target).tolist())

    missing = int(np.isnan(data.data).sum())
    if missing:
        logger.warning("Dataset contains %d missing values", missing)
    return data


def get_features_and_labels(data: Bunch) -> tuple[np.ndarray, np.ndarray]:
    """Separate the feature matrix from the label vector."""
    return data.data, data.target


def split_data(
    features: np.ndarray,
    labels: np.ndarray,
    test_size: float = settings.test_size,
    random_seed: int = settings.random_seed,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/test split. No preprocessing happens here (avoids leakage)."""
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_seed,
        stratify=labels,
    )
    logger.info("Split done: train=%s, test=%s", x_train.shape, x_test.shape)
    return x_train, x_test, y_train, y_test
