"""
Logging setup, configured once for the whole application.
"""

import logging

from production_app.config import settings

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

_configured = False


def setup_logging() -> None:
    """Configure the root logger once; repeated calls do nothing."""
    global _configured
    if _configured:
        return

    logging.basicConfig(
        level=settings.log_level,
        format=LOG_FORMAT,
    )
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger (use __name__ in each module)."""
    return logging.getLogger(name)
