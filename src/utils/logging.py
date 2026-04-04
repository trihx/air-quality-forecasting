"""Logging setup — loguru configuration for the project."""

import sys
from pathlib import Path

from loguru import logger


def setup_logging(
    level: str = "INFO",
    log_dir: str | Path | None = None,
    serialize: bool = False,
) -> None:
    """Configure loguru for the project.

    Args:
        level: Minimum log level for console output.
        log_dir: Directory for log files. If None, only console logging.
        serialize: If True, also write JSON logs for structured analysis.
    """
    # Remove default handler
    logger.remove()

    # Console handler: colorful, concise
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level:<8}</level> | "
            "<cyan>{module}</cyan>:<cyan>{function}</cyan> | "
            "{message}"
        ),
        colorize=True,
    )

    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        # File handler: detailed, rotated
        logger.add(
            str(log_path / "{time:YYYYMMDD}.log"),
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {module}:{function}:{line} | {message}",
            rotation="10 MB",
            retention="30 days",
            compression="gz",
        )

        # JSON handler for structured analysis (optional)
        if serialize:
            logger.add(
                str(log_path / "structured.jsonl"),
                level="INFO",
                serialize=True,
                rotation="10 MB",
                retention="30 days",
            )

    logger.info(f"Logging configured: level={level}, log_dir={log_dir}")
