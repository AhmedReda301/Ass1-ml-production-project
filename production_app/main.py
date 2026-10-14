"""Application orchestration for training and saved-model prediction."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from production_app.config import settings
from production_app.data import (
    get_features_and_labels,
    load_dataset,
    split_data,
)
from production_app.evaluation import evaluate_model
from production_app.logging_config import get_logger, setup_logging
from production_app.model_service import load_model, predict, save_model
from production_app.training import train_model

logger = get_logger(__name__)


def _parse_features(raw_features: str) -> object:
    """Parse a JSON list containing one or more feature rows."""
    try:
        return json.loads(raw_features)
    except json.JSONDecodeError as exc:
        raise ValueError("--features must be valid JSON") from exc


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for the application workflow."""
    parser = argparse.ArgumentParser(
        description="Breast cancer classification workflow"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("train", help="Train, evaluate, and save the pipeline")

    predict_parser = subparsers.add_parser(
        "predict", help="Load the saved pipeline and predict from JSON features"
    )
    predict_parser.add_argument(
        "--features",
        required=True,
        help="JSON list of 30 features, or a list of feature rows",
    )
    predict_parser.add_argument(
        "--model-path",
        type=Path,
        default=settings.model_path,
        help=f"Saved pipeline path (default: {settings.model_path})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested training or prediction command."""
    setup_logging()
    args = build_parser().parse_args(argv)

    try:
        if args.command == "train":
            dataset = load_dataset()
            features, labels = get_features_and_labels(dataset)
            x_train, x_test, y_train, y_test = split_data(features, labels)

            pipeline = train_model(x_train, y_train)
            target_names = [str(name) for name in dataset.target_names]
            metrics = evaluate_model(pipeline, x_test, y_test, target_names)
            model_path = save_model(pipeline, settings.model_path)
            logger.info(
                "Training workflow completed: model=%s metrics=%s",
                model_path,
                metrics,
            )
        else:
            pipeline = load_model(args.model_path)
            predictions = predict(pipeline, _parse_features(args.features))
            logger.info("Saved-model prediction completed: %s", predictions)
    except (FileNotFoundError, ValueError):
        logger.exception("Application workflow failed")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
