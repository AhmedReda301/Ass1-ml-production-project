"""Reusable preprocessing steps (fitted only inside the Pipeline, on training data)."""

from sklearn.preprocessing import StandardScaler


def build_preprocessor() -> StandardScaler:
    """Return an unfitted scaler; the Pipeline fits it on training data only."""
    return StandardScaler()
