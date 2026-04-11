"""Data leakage detection tests for the forecasting pipeline.

These tests ensure no feature contains information from the future
or from the target variable at prediction time t.

Usage:
    uv run pytest tests/validation/test_leakage.py -v
"""

import numpy as np
import pandas as pd
import pytest
from loguru import logger
from src.data.loader import TARGET_COL


@pytest.fixture
def marts_df() -> pd.DataFrame:
    """Load marts feature data."""
    from pathlib import Path

    path = Path("dataset/processed/marts_features.csv")
    if not path.exists():
        pytest.skip("Marts data not available — run feature pipeline first")
    return pd.read_csv(path, index_col=0, parse_dates=True)


@pytest.fixture
def feature_cols(marts_df: pd.DataFrame) -> list[str]:
    """Get feature columns (everything except target)."""
    return [c for c in marts_df.columns if c != TARGET_COL]


class TestNoTargetLeakage:
    """Ensure no feature directly encodes the target at time t."""

    def test_no_perfect_correlation_with_target(self, marts_df: pd.DataFrame, feature_cols: list[str]) -> None:
        """No feature should have |correlation| > 0.99 with target.

        A near-perfect correlation strongly suggests the feature
        contains the target value (or a linear transform of it).
        """
        target = marts_df[TARGET_COL]
        high_corr_features: list[tuple[str, float]] = []

        for col in feature_cols:
            if marts_df[col].dtype in [np.float64, np.int64, float, int]:
                corr = float(marts_df[col].corr(target))
                if abs(corr) > 0.99:
                    high_corr_features.append((col, corr))

        if high_corr_features:
            msg_lines = ["Features with |corr| > 0.99 with target (probable leakage):"]
            for col, corr in sorted(high_corr_features, key=lambda x: abs(x[1]), reverse=True):
                msg_lines.append(f"  {col}: corr={corr:.6f}")
            pytest.fail("\n".join(msg_lines))

    def test_no_feature_equals_target(self, marts_df: pd.DataFrame, feature_cols: list[str]) -> None:
        """No feature column should be identical to the target column."""
        target = marts_df[TARGET_COL].values
        identical_features: list[str] = []

        for col in feature_cols:
            if np.allclose(marts_df[col].values, target, equal_nan=True):
                identical_features.append(col)

        assert not identical_features, (
            f"Features identical to target {TARGET_COL}: {identical_features}"
        )

    def test_diff_features_use_shifted_values(self, marts_df: pd.DataFrame) -> None:
        """Diff features should NOT contain y[t] in their calculation.

        diff(1) = y[t] - y[t-1] contains y[t] → LEAKAGE.
        Correct: shifted_diff = y[t-1] - y[t-2] (uses only past values).
        """
        diff_cols = [c for c in marts_df.columns if "_diff_" in c or "_pct_change_" in c]

        if not diff_cols:
            pytest.skip("No diff features found")

        target = marts_df[TARGET_COL]
        leaky_diffs: list[tuple[str, float]] = []

        for col in diff_cols:
            corr = abs(float(marts_df[col].corr(target)))
            if corr > 0.95:
                leaky_diffs.append((col, corr))

        assert not leaky_diffs, (
            "Diff features suspiciously correlated with target (likely contain y[t]):\n"
            + "\n".join(f"  {c}: corr={v:.4f}" for c, v in leaky_diffs)
        )

    def test_domain_features_no_current_target(self, marts_df: pd.DataFrame) -> None:
        """Domain features must not use target value at time t.

        co2_pm25_ratio should use lag target, not current target.
        pm25_aqi_cat should use lag target, not current target.
        """
        domain_features = ["co2_pm25_ratio", "pm25_aqi_cat"]
        target = marts_df[TARGET_COL]
        leaky: list[tuple[str, float]] = []

        for col in domain_features:
            if col in marts_df.columns:
                corr = abs(float(marts_df[col].corr(target)))
                if corr > 0.95:
                    leaky.append((col, corr))

        assert not leaky, (
            "Domain features likely use current target value:\n"
            + "\n".join(f"  {c}: corr={v:.4f}" for c, v in leaky)
        )


class TestTemporalIntegrity:
    """Ensure temporal ordering and no future information."""

    def test_lag_features_are_actually_lagged(self, marts_df: pd.DataFrame) -> None:
        """Verify lag features equal shifted target values."""
        if "pm25_lag_1h" not in marts_df.columns:
            pytest.skip("pm25_lag_1h not found")

        target = marts_df[TARGET_COL]
        lag_1h = marts_df["pm25_lag_1h"]

        # pm25_lag_1h should equal pm25 shifted by 1
        expected = target.shift(1)
        valid_idx = expected.notna()

        np.testing.assert_allclose(
            lag_1h[valid_idx].values,
            expected[valid_idx].values,
            rtol=1e-10,
            err_msg="pm25_lag_1h doesn't match shifted target — possible feature bug",
        )

    def test_rolling_features_use_past_only(self, marts_df: pd.DataFrame) -> None:
        """Rolling features should be computed on shifted (past) values only.

        A rolling feature that includes y[t] will have high correlation with target.
        """
        roll_cols = [c for c in marts_df.columns if "_roll_" in c and "pm25" in c]

        if not roll_cols:
            pytest.skip("No rolling features found")

        target = marts_df[TARGET_COL]
        suspicious: list[tuple[str, float]] = []

        for col in roll_cols:
            corr = abs(float(marts_df[col].corr(target)))
            # Rolling mean of short window with y[t] would be ~0.99+ correlated
            # Rolling mean of shifted past should be ~0.8-0.95 correlated
            if corr > 0.99:
                suspicious.append((col, corr))

        assert not suspicious, (
            "Rolling features may include current value (shift needed):\n"
            + "\n".join(f"  {c}: corr={v:.4f}" for c, v in suspicious)
        )


class TestShuffleTest:
    """Randomization test: model should fail with shuffled target."""

    def test_ridge_fails_with_shuffled_target(self, marts_df: pd.DataFrame, feature_cols: list[str]) -> None:
        """If we shuffle the target, R² should drop near 0.

        If R² stays high after shuffle → features encode target directly.
        """
        from sklearn.linear_model import Ridge
        from sklearn.metrics import r2_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        X = marts_df[feature_cols].select_dtypes(include=[np.number])
        y = marts_df[TARGET_COL].values.copy()

        # Use last 20% as test
        n = len(X)
        split = int(n * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y[:split], y[split:]

        # Shuffle target in training set
        rng = np.random.RandomState(42)
        y_train_shuffled = rng.permutation(y_train)

        pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
        pipe.fit(X_train, y_train_shuffled)
        y_pred = pipe.predict(X_test)
        r2 = r2_score(y_test, y_pred)

        logger.info(f"Shuffle test R²: {r2:.4f} (should be near 0 or negative)")

        # R² should be very low with shuffled target
        assert r2 < 0.5, (
            f"R²={r2:.4f} is too high with shuffled target — "
            f"features likely encode the target directly (data leakage)"
        )


class TestFeatureNamePatterns:
    """Check feature naming for suspicious patterns."""

    def test_no_raw_target_in_features(self, marts_df: pd.DataFrame, feature_cols: list[str]) -> None:
        """Feature list should not contain the raw target column name
        (it should be excluded during X/y split)."""
        assert TARGET_COL not in feature_cols, (
            f"Raw target column '{TARGET_COL}' found in features — must be excluded"
        )

    def test_suspicious_feature_patterns(self, marts_df: pd.DataFrame, feature_cols: list[str]) -> None:
        """Flag features whose names suggest they might use current target value."""
        suspicious_patterns = [
            ("_ratio", "ratio features may use current target"),
            ("_aqi_cat", "AQI category may use current target"),
        ]

        warnings: list[str] = []
        for pattern, reason in suspicious_patterns:
            matches = [c for c in feature_cols if pattern in c and TARGET_COL.split("_")[0] in c]
            if matches:
                warnings.append(f"  {matches} — {reason}")

        # This is a soft warning, not a hard fail
        if warnings:
            logger.warning("Suspicious feature patterns found:\n" + "\n".join(warnings))
