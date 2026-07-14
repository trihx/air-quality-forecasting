"""Path validation utilities — security rules from SKILL.md §1.7."""

from pathlib import Path

from loguru import logger

# Project root resolved once
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()


def validate_path(path: str | Path, must_exist: bool = True) -> Path:
    """Validate and resolve a file path within the project.

    Args:
        path: File path to validate.
        must_exist: If True, raise error when file doesn't exist.

    Returns:
        Resolved Path object.

    Raises:
        ValueError: If path is outside project directory.
        FileNotFoundError: If must_exist=True and file doesn't exist.
    """
    p = Path(path).resolve()

    if not str(p).startswith(str(PROJECT_ROOT)):
        raise ValueError(f"Path '{path}' is outside project directory: {PROJECT_ROOT}")

    if must_exist and not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    logger.debug(f"Path validated: {p}")
    return p


def validate_data_path(path: str | Path) -> Path:
    """Validate a data file path (must be in dataset/ directory).

    Args:
        path: Data file path.

    Returns:
        Resolved Path object.

    Raises:
        ValueError: If path is not in dataset/ directory.
    """
    p = validate_path(path, must_exist=True)
    dataset_dir = PROJECT_ROOT / "dataset"

    if not str(p).startswith(str(dataset_dir)):
        raise ValueError(f"Data path must be in {dataset_dir}, got: {p}")

    return p
