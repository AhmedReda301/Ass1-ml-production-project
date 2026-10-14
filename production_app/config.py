"""Application settings loaded from environment variables."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    environment: str
    log_level: str
    model_path: Path
    test_size: float
    random_seed: int


def _resolve_path(value: str) -> Path:
    """Make relative paths relative to the project root, not the current directory."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_settings() -> Settings:
    test_size = float(os.getenv("TEST_SIZE", "0.2"))
    if not 0 < test_size < 1:
        raise ValueError(f"TEST_SIZE must be between 0 and 1, got {test_size}")

    return Settings(
        environment=os.getenv("ENVIRONMENT", "development"),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        model_path=_resolve_path(
            os.getenv("MODEL_PATH", "models/breast_cancer_pipeline.joblib")
        ),
        test_size=test_size,
        random_seed=int(os.getenv("RANDOM_SEED", "42")),
    )


settings = load_settings()