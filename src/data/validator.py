"""Data validator — domain-specific quality checks for PM2.5 pipeline.

Validates data at each layer (Staging → Intermediate → Marts) per SKILL.md §3.6.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.data.loader import DATETIME_COL, EXPECTED_COLUMNS, FEATURE_COLS, TARGET_COL


class Severity(Enum):
    """Validation severity levels."""

    CRITICAL = "🔴 CRITICAL"
    WARNING = "🟡 WARNING"
    INFO = "🔵 INFO"


@dataclass
class ValidationResult:
    """Result of a single validation check."""

    check_name: str
    passed: bool
    severity: Severity
    message: str
    details: dict[str, Any] = field(default_factory=dict)


class DataValidator:
    """Validates data quality at each pipeline layer."""

    def __init__(self) -> None:
        self.results: list[ValidationResult] = []

    def validate_staging(self, df: pd.DataFrame) -> list[ValidationResult]:
        """Validate raw data (Staging layer).

        Checks per SKILL.md §3.6:
        - Columns exist + Schema match (CRITICAL)
        - PM2.5 range [0, 500] (WARNING)
        """
        self.results = []

        # CRITICAL: Columns exist
        self._check_columns_exist(df)

        # CRITICAL: Schema types
        self._check_schema_types(df)

        # CRITICAL: Non-empty
        self._check_non_empty(df)

        # WARNING: Value ranges
        self._check_pm25_range(df)
        self._check_feature_ranges(df)

        # WARNING: Missing value rates
        self._check_missing_rates(df)

        # WARNING: Temporal consistency
        self._check_temporal_consistency(df)

        # INFO: Data freshness
        self._check_data_freshness(df)

        self._log_summary("STAGING")
        return self.results

    def validate_intermediate(self, df: pd.DataFrame) -> list[ValidationResult]:
        """Validate cleaned data (Intermediate layer).

        Checks per SKILL.md §3.6:
        - No NaN after interpolation (CRITICAL)
        - Regular time frequency (WARNING)
        """
        self.results = []

        # CRITICAL: No NaN after cleaning
        self._check_no_nan(df)

        # CRITICAL: Monotonic index
        self._check_monotonic_index(df)

        # WARNING: Regular frequency
        self._check_regular_frequency(df)

        # WARNING: Sufficient data points
        self._check_sufficient_data(df, min_rows=1000)

        # INFO: Stats sanity
        self._check_stats_sanity(df)

        self._log_summary("INTERMEDIATE")
        return self.results

    def validate_marts(
        self,
        df: pd.DataFrame,
        target_col: str = TARGET_COL,
    ) -> list[ValidationResult]:
        """Validate feature-engineered data (Marts layer).

        Checks per SKILL.md §3.6:
        - No future data leakage (CRITICAL)
        """
        self.results = []

        # CRITICAL: No future data leakage in lag features
        self._check_no_leakage(df, target_col)

        # CRITICAL: No NaN in features
        self._check_no_nan(df)

        # WARNING: Feature variance
        self._check_feature_variance(df)

        self._log_summary("MARTS")
        return self.results

    def has_critical_failures(self) -> bool:
        """Check if any CRITICAL validation failed."""
        return any(not r.passed and r.severity == Severity.CRITICAL for r in self.results)

    # ================================================================
    # Staging checks
    # ================================================================

    def _check_columns_exist(self, df: pd.DataFrame) -> None:
        missing = set(EXPECTED_COLUMNS) - set(df.columns)
        self.results.append(
            ValidationResult(
                check_name="columns_exist",
                passed=len(missing) == 0,
                severity=Severity.CRITICAL,
                message=f"Missing columns: {missing}" if missing else "All expected columns present",
                details={"missing": list(missing)},
            )
        )

    def _check_schema_types(self, df: pd.DataFrame) -> None:
        type_errors = []
        for col in FEATURE_COLS + [TARGET_COL]:
            if col in df.columns and not np.issubdtype(df[col].dtype, np.number):
                type_errors.append(f"{col}: expected numeric, got {df[col].dtype}")

        if DATETIME_COL in df.columns and not pd.api.types.is_datetime64_any_dtype(df[DATETIME_COL]):
            type_errors.append(f"{DATETIME_COL}: expected datetime, got {df[DATETIME_COL].dtype}")

        self.results.append(
            ValidationResult(
                check_name="schema_types",
                passed=len(type_errors) == 0,
                severity=Severity.CRITICAL,
                message=f"Type errors: {type_errors}" if type_errors else "All dtypes correct",
                details={"errors": type_errors},
            )
        )

    def _check_non_empty(self, df: pd.DataFrame) -> None:
        self.results.append(
            ValidationResult(
                check_name="non_empty",
                passed=len(df) > 0,
                severity=Severity.CRITICAL,
                message=f"DataFrame has {len(df)} rows" if len(df) > 0 else "DataFrame is empty!",
            )
        )

    def _check_pm25_range(self, df: pd.DataFrame) -> None:
        if TARGET_COL not in df.columns:
            return
        col = df[TARGET_COL].dropna()
        n_out = ((col < 0) | (col > 500)).sum()
        self.results.append(
            ValidationResult(
                check_name="pm25_range",
                passed=n_out == 0,
                severity=Severity.WARNING,
                message=(
                    f"PM2.5 out of [0, 500]: {n_out} values" if n_out > 0 else "PM2.5 within valid range [0, 500]"
                ),
                details={"n_out_of_range": int(n_out), "min": float(col.min()), "max": float(col.max())},
            )
        )

    def _check_feature_ranges(self, df: pd.DataFrame) -> None:
        from src.data.cleaner import PHYSICAL_BOUNDS

        out_of_range = {}
        for col, (lower, upper) in PHYSICAL_BOUNDS.items():
            if col not in df.columns:
                continue
            series = df[col].dropna()
            n_out = ((series < lower) | (series > upper)).sum()
            if n_out > 0:
                out_of_range[col] = int(n_out)

        self.results.append(
            ValidationResult(
                check_name="feature_ranges",
                passed=len(out_of_range) == 0,
                severity=Severity.WARNING,
                message=(
                    f"Out-of-range values: {out_of_range}" if out_of_range else "All features within physical bounds"
                ),
                details={"out_of_range": out_of_range},
            )
        )

    def _check_missing_rates(self, df: pd.DataFrame, max_pct: float = 10.0) -> None:
        high_missing = {}
        for col in FEATURE_COLS + [TARGET_COL]:
            if col not in df.columns:
                continue
            pct = df[col].isna().mean() * 100
            if pct > max_pct:
                high_missing[col] = round(pct, 2)

        self.results.append(
            ValidationResult(
                check_name="missing_rates",
                passed=len(high_missing) == 0,
                severity=Severity.WARNING,
                message=(
                    f"Columns with >{max_pct}% missing: {high_missing}"
                    if high_missing
                    else f"All columns have <{max_pct}% missing values"
                ),
                details={"high_missing": high_missing},
            )
        )

    def _check_temporal_consistency(self, df: pd.DataFrame) -> None:
        if DATETIME_COL not in df.columns:
            return
        dates = df[DATETIME_COL].dropna()
        if len(dates) < 2:
            return

        time_diffs = dates.diff().dropna()
        n_negative = (time_diffs < pd.Timedelta(0)).sum()
        large_gaps = time_diffs[time_diffs > pd.Timedelta(hours=6)]

        self.results.append(
            ValidationResult(
                check_name="temporal_consistency",
                passed=n_negative == 0,
                severity=Severity.WARNING,
                message=(
                    f"Temporal issues: {n_negative} negative diffs, {len(large_gaps)} gaps >6h"
                    if n_negative > 0 or len(large_gaps) > 0
                    else "Temporal order consistent"
                ),
                details={
                    "n_negative_diffs": int(n_negative),
                    "n_large_gaps": len(large_gaps),
                    "max_gap": str(time_diffs.max()) if len(time_diffs) > 0 else "N/A",
                },
            )
        )

    def _check_data_freshness(self, df: pd.DataFrame) -> None:
        if DATETIME_COL not in df.columns:
            return
        dates = df[DATETIME_COL].dropna()
        if len(dates) == 0:
            return
        span_days = (dates.max() - dates.min()).days
        self.results.append(
            ValidationResult(
                check_name="data_freshness",
                passed=span_days > 365,
                severity=Severity.INFO,
                message=f"Data spans {span_days} days ({span_days / 365:.1f} years)",
                details={"span_days": span_days, "start": str(dates.min()), "end": str(dates.max())},
            )
        )

    # ================================================================
    # Intermediate checks
    # ================================================================

    def _check_no_nan(self, df: pd.DataFrame) -> None:
        nan_counts = df.isna().sum()
        total_nan = nan_counts.sum()
        cols_with_nan = {col: int(count) for col, count in nan_counts.items() if count > 0}
        self.results.append(
            ValidationResult(
                check_name="no_nan",
                passed=total_nan == 0,
                severity=Severity.CRITICAL,
                message=(
                    f"NaN values remaining: {cols_with_nan}" if total_nan > 0 else "No NaN values — data is complete"
                ),
                details={"total_nan": int(total_nan), "by_column": cols_with_nan},
            )
        )

    def _check_monotonic_index(self, df: pd.DataFrame) -> None:
        if isinstance(df.index, pd.DatetimeIndex):
            is_mono = df.index.is_monotonic_increasing
        elif DATETIME_COL in df.columns:
            is_mono = df[DATETIME_COL].is_monotonic_increasing
        else:
            is_mono = True  # Can't check

        self.results.append(
            ValidationResult(
                check_name="monotonic_index",
                passed=is_mono,
                severity=Severity.CRITICAL,
                message="Time index is monotonically increasing" if is_mono else "Time index NOT monotonic!",
            )
        )

    def _check_regular_frequency(self, df: pd.DataFrame) -> None:
        if not isinstance(df.index, pd.DatetimeIndex) or len(df) < 3:
            return

        inferred = pd.infer_freq(df.index)
        self.results.append(
            ValidationResult(
                check_name="regular_frequency",
                passed=inferred is not None,
                severity=Severity.WARNING,
                message=(
                    f"Regular frequency detected: {inferred}" if inferred else "Irregular frequency — check for gaps"
                ),
                details={"inferred_freq": inferred},
            )
        )

    def _check_sufficient_data(self, df: pd.DataFrame, min_rows: int = 1000) -> None:
        self.results.append(
            ValidationResult(
                check_name="sufficient_data",
                passed=len(df) >= min_rows,
                severity=Severity.WARNING,
                message=f"Data has {len(df):,} rows (min required: {min_rows:,})",
                details={"n_rows": len(df), "min_required": min_rows},
            )
        )

    def _check_stats_sanity(self, df: pd.DataFrame) -> None:
        issues = []
        for col in FEATURE_COLS + [TARGET_COL]:
            if col not in df.columns:
                continue
            data = df[col]
            if data.std() == 0:
                issues.append(f"{col}: zero variance (constant)")
            if data.nunique() < 3:
                issues.append(f"{col}: only {data.nunique()} unique values")

        self.results.append(
            ValidationResult(
                check_name="stats_sanity",
                passed=len(issues) == 0,
                severity=Severity.INFO,
                message=f"Stats issues: {issues}" if issues else "All features have reasonable variance",
                details={"issues": issues},
            )
        )

    # ================================================================
    # Marts checks
    # ================================================================

    def _check_no_leakage(self, df: pd.DataFrame, target_col: str) -> None:
        """Check for future data leakage in lag/rolling features."""
        leakage_cols = []
        for col in df.columns:
            if col == target_col:
                continue
            # Check if any feature has a perfect correlation with target (suspiciously close to 1.0)
            if target_col in df.columns and col in df.columns:
                try:
                    corr = df[col].corr(df[target_col])
                    if abs(corr) > 0.99:
                        leakage_cols.append(f"{col} (corr={corr:.4f})")
                except (ValueError, TypeError):
                    pass

        self.results.append(
            ValidationResult(
                check_name="no_leakage",
                passed=len(leakage_cols) == 0,
                severity=Severity.CRITICAL,
                message=(
                    f"Potential leakage — near-perfect correlation: {leakage_cols}"
                    if leakage_cols
                    else "No obvious data leakage detected"
                ),
                details={"leakage_columns": leakage_cols},
            )
        )

    def _check_feature_variance(self, df: pd.DataFrame) -> None:
        low_var = []
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].std() < 1e-10:
                low_var.append(col)

        self.results.append(
            ValidationResult(
                check_name="feature_variance",
                passed=len(low_var) == 0,
                severity=Severity.WARNING,
                message=(
                    f"Zero-variance features (remove!): {low_var}" if low_var else "All features have non-zero variance"
                ),
                details={"zero_variance_features": low_var},
            )
        )

    # ================================================================
    # Reporting
    # ================================================================

    def _log_summary(self, layer: str) -> None:
        n_pass = sum(1 for r in self.results if r.passed)
        n_fail = sum(1 for r in self.results if not r.passed)
        n_critical = sum(1 for r in self.results if not r.passed and r.severity == Severity.CRITICAL)

        logger.info(f"--- {layer} Validation: {n_pass} passed, {n_fail} failed ---")
        for r in self.results:
            icon = "✅" if r.passed else "❌"
            logger.info(f"  {icon} [{r.severity.value}] {r.check_name}: {r.message}")

        if n_critical > 0:
            logger.error(f"  ⛔ {n_critical} CRITICAL failures — pipeline should STOP")

    def get_report(self) -> dict[str, Any]:
        """Get structured validation report."""
        return {
            "total_checks": len(self.results),
            "passed": sum(1 for r in self.results if r.passed),
            "failed": sum(1 for r in self.results if not r.passed),
            "critical_failures": sum(1 for r in self.results if not r.passed and r.severity == Severity.CRITICAL),
            "checks": [
                {
                    "name": r.check_name,
                    "passed": r.passed,
                    "severity": r.severity.value,
                    "message": r.message,
                    "details": r.details,
                }
                for r in self.results
            ],
        }
