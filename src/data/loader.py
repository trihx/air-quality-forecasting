"""Data loader — load, validate, and provide basic info about PM2.5 dataset."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.utils.path_validator import validate_data_path

# Column definitions from SKILL.md §1.3
EXPECTED_COLUMNS = ["nhiet_do", "do_am", "diem_suong", "co2", "pm25", "ngay_tao"]
TARGET_COL = "pm25"
DATETIME_COL = "ngay_tao"
FEATURE_COLS = ["nhiet_do", "do_am", "diem_suong", "co2"]

# Data type mapping
DTYPES = {
    "nhiet_do": np.float64,
    "do_am": np.float64,
    "diem_suong": np.float64,
    "co2": np.float64,
    "pm25": np.float64,
}


def load_raw_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load raw PM2.5 dataset from CSV.

    Args:
        path: Path to CSV file. Defaults to dataset/raw/final_dataset.csv.

    Returns:
        DataFrame with parsed datetime index and correct dtypes.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If required columns are missing.
    """
    if path is None:
        project_root = Path(__file__).parent.parent.parent.resolve()
        path = project_root / "dataset" / "raw" / "final_dataset.csv"

    validated_path = validate_data_path(path)
    logger.info(f"Loading data from {validated_path.name}")

    df = pd.read_csv(
        validated_path,
        parse_dates=[DATETIME_COL],
        dtype=DTYPES,
    )

    # Validate columns
    _validate_columns(df)

    # Drop rows with missing datetime (NaT)
    n_nat = df[DATETIME_COL].isna().sum()
    if n_nat > 0:
        logger.warning(f"Dropped {n_nat} rows with missing datetime (NaT)")
        df = df.dropna(subset=[DATETIME_COL])

    # Sort by datetime
    df = df.sort_values(DATETIME_COL).reset_index(drop=True)

    # Basic stats
    logger.info(
        f"Loaded: {len(df):,} rows, {len(df.columns)} cols | "
        f"Date range: {df[DATETIME_COL].min()} → {df[DATETIME_COL].max()}"
    )
    logger.debug(f"Memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB | Dtypes: {df.dtypes.to_dict()}")

    return df


def get_data_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Generate summary statistics about the dataset.

    Args:
        df: Input DataFrame.

    Returns:
        Dictionary with summary info.
    """
    summary: dict[str, Any] = {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "columns": list(df.columns),
        "date_range": {
            "start": str(df[DATETIME_COL].min()),
            "end": str(df[DATETIME_COL].max()),
            "days": (df[DATETIME_COL].max() - df[DATETIME_COL].min()).days,
        },
        "missing": {},
        "stats": {},
    }

    # Missing values per column
    for col in FEATURE_COLS + [TARGET_COL]:
        n_missing = int(df[col].isna().sum())
        pct_missing = float(n_missing / len(df) * 100)
        summary["missing"][col] = {
            "count": n_missing,
            "pct": round(pct_missing, 2),
        }
        if pct_missing > 5:
            logger.warning(f"High null rate: {col} has {pct_missing:.1f}% missing")

    # Descriptive stats for numeric columns
    for col in FEATURE_COLS + [TARGET_COL]:
        col_data = df[col].dropna()
        summary["stats"][col] = {
            "min": float(round(float(col_data.min()), 2)),
            "max": float(round(float(col_data.max()), 2)),
            "mean": float(round(float(col_data.mean()), 2)),
            "median": float(round(float(col_data.median()), 2)),
            "std": float(round(float(col_data.std()), 2)),
            "skewness": float(round(float(col_data.skew()), 2)),
        }

    # Sampling frequency
    if len(df) > 1:
        time_diffs = df[DATETIME_COL].diff().dropna()
        median_interval = time_diffs.median()
        summary["sampling"] = {
            "median_interval_seconds": median_interval.total_seconds(),
            "median_interval_human": str(median_interval),
        }

    return summary


def validate_dataset(df: pd.DataFrame) -> list[str]:
    """Run basic validation checks on dataset.

    Args:
        df: Input DataFrame.

    Returns:
        List of warning/error messages. Empty = all checks passed.
    """
    issues: list[str] = []

    # 1. Check required columns
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")

    # 2. Check for duplicated timestamps
    n_dup = df[DATETIME_COL].duplicated().sum()
    if n_dup > 0:
        issues.append(f"Duplicated timestamps: {n_dup}")

    # 3. Check for negative values (should be >= 0 for all sensor readings)
    for col in FEATURE_COLS + [TARGET_COL]:
        if col in df.columns:
            n_negative = (df[col] < 0).sum()
            if n_negative > 0:
                issues.append(f"Negative values in {col}: {n_negative}")

    # 4. Check time ordering
    if not df[DATETIME_COL].is_monotonic_increasing:
        issues.append("Data is not sorted by datetime")

    # 5. Check for large gaps (> 1 hour)
    if len(df) > 1:
        time_diffs = df[DATETIME_COL].diff().dropna()
        large_gaps = time_diffs[time_diffs > pd.Timedelta(hours=1)]
        if len(large_gaps) > 0:
            issues.append(f"Large time gaps (>1h): {len(large_gaps)} gaps, max gap: {large_gaps.max()}")

    # 6. Check PM2.5 range (WHO: 0-500 AQI range)
    if TARGET_COL in df.columns:
        max_pm25 = df[TARGET_COL].max()
        if max_pm25 > 500:
            issues.append(f"PM2.5 exceeds 500 µg/m³: max={max_pm25}")

    # Report
    if issues:
        for issue in issues:
            logger.warning(f"Validation: {issue}")
    else:
        logger.info("Validation: All checks passed ✅")

    return issues


def _validate_columns(df: pd.DataFrame) -> None:
    """Check that all expected columns exist."""
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Expected: {EXPECTED_COLUMNS}, Got: {list(df.columns)}")
