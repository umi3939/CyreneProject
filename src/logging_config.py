"""
src/logging_config.py - Production logging configuration.

Log levels:
- DEBUG: Internal verification (default OFF, enable via CYRENE_DEBUG=1)
- INFO: Startup and fatal errors only
- WARNING/ERROR: System issues

Environment variables:
- CYRENE_DEBUG=1: Enable DEBUG level logging for internal inspection
"""

import logging
import os


def is_debug_enabled() -> bool:
    """Check if debug mode is enabled.

    Checks environment variable at call time (not import time)
    to support test fixtures that set CYRENE_DEBUG=1.
    """
    return os.environ.get("CYRENE_DEBUG", "0") == "1"


def configure_logging(level: int | None = None) -> None:
    """Configure logging for the application.

    Args:
        level: Override log level. If None, uses INFO (production) or DEBUG.
    """
    if level is not None:
        effective_level = level
    else:
        effective_level = logging.DEBUG if is_debug_enabled() else logging.INFO

    logging.basicConfig(
        level=effective_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Suppress noisy third-party loggers
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
