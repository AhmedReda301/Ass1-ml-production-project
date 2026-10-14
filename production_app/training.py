"""Model construction and training."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from production_app.config import settings
from production_app.logging_config import get_logger
from production_app.preprocessing import build_preprocessor

logger = get_logger(__name__)


def build_pipeline() -> Pipeline:
    """Create an unfitted Pipeline: preprocessing + classifier."""
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            (
                "model",
                LogisticRegression(max_iter=1000, random_state=settings.random_seed),
            ),
        ]
    )


def train_model(x_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    """Fit the pipeline on training data and return it."""
    pipeline = build_pipeline()
    logger.info("Training started on %d samples", x_train.shape[0])
    try:
        pipeline.fit(x_train, y_train)
    except Exception:
        logger.exception("Training failed")
        raise
    logger.info("Training finished")
    return pipeline