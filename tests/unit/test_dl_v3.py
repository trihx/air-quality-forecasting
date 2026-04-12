"""Tests for DL v3 — PCA feature selection + TFT validation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class TestPCAFeatureSelection:
    """Test PCA dimensionality reduction for DL pipeline."""

    def test_pca_preserves_variance(self):
        """PCA with 95% threshold should capture >= 95% variance."""
        np.random.seed(42)
        X = np.random.randn(500, 100)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=0.95, random_state=42)
        X_pca = pca.fit_transform(X_scaled)

        explained = sum(pca.explained_variance_ratio_)
        assert explained >= 0.95, f"PCA should explain >= 95%, got {explained:.3f}"
        assert X_pca.shape[1] < 100, "PCA should reduce dimensions"

    def test_pca_fit_on_train_only(self):
        """PCA must be fit on training data only to prevent leakage."""
        np.random.seed(42)
        # Simulate train/test with different distributions
        X_train = np.random.randn(300, 50)
        X_test = np.random.randn(100, 50) + 3  # shifted distribution

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        pca = PCA(n_components=0.95, random_state=42)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)

        assert X_train_pca.shape[1] == X_test_pca.shape[1], "Same dims after transform"
        assert X_test_pca.shape[0] == 100, "Test samples preserved"

    def test_pca_reduces_from_117_features(self):
        """Simulated 117-feature data should reduce significantly."""
        np.random.seed(42)
        # Create correlated features (like PM2.5 lag/rolling/ewm)
        base = np.random.randn(500, 10)
        # Add correlated copies with noise
        X = np.hstack([
            base,
            base + np.random.randn(500, 10) * 0.1,  # highly correlated
            base[:, :5] + np.random.randn(500, 5) * 0.5,  # medium correlated
            np.random.randn(500, 92),  # random features
        ])
        assert X.shape[1] == 117, "Should have 117 features"

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=0.95, random_state=42)
        X_pca = pca.fit_transform(X_scaled)

        # Should reduce significantly due to correlated features
        assert X_pca.shape[1] < 117, f"PCA should reduce, got {X_pca.shape[1]}"
        print(f"  117 → {X_pca.shape[1]} components (95% variance)")

    def test_pca_no_nan_output(self):
        """PCA output should never contain NaN."""
        np.random.seed(42)
        X = np.random.randn(200, 50)
        X = np.nan_to_num(X, nan=0.0)  # same safeguard as in script

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=0.95, random_state=42)
        X_pca = pca.fit_transform(X_scaled)

        assert not np.isnan(X_pca).any(), "PCA output should have no NaN"
        assert not np.isinf(X_pca).any(), "PCA output should have no inf"


class TestTopNFeatureSelection:
    """Test top-N feature importance selection."""

    def test_top_n_correct_count(self):
        """Top-N selection should return exactly N features."""
        importances = pd.Series(np.random.rand(117), index=[f"feat_{i}" for i in range(117)])
        for n in [10, 20, 40]:
            top = list(importances.sort_values(ascending=False).head(n).index)
            assert len(top) == n, f"Top-{n} should have {n} features, got {len(top)}"

    def test_top_n_ordered_by_importance(self):
        """Selected features should be ordered by importance."""
        importances = pd.Series({
            "pm25_lag_1h": 500, "co2": 200, "nhiet_do": 100,
            "noise_1": 1, "noise_2": 0,
        })
        top3 = list(importances.sort_values(ascending=False).head(3).index)
        assert top3 == ["pm25_lag_1h", "co2", "nhiet_do"]


class TestTFTDataPrep:
    """Test TFT data preparation with v2 features."""

    def test_static_cols_are_cyclical(self):
        """Static columns should be calendar cyclical features."""
        feature_cols = [
            "pm25_lag_1h", "pm25_lag_2h", "nhiet_do",
            "hour_sin", "hour_cos", "day_sin", "day_cos",
            "month_sin", "month_cos", "fourier_daily_sin_1",
        ]
        static = [c for c in feature_cols if c.startswith(("hour_", "day_", "month_"))]
        temporal = [c for c in feature_cols if c not in static]

        assert "hour_sin" in static
        assert "hour_cos" in static
        assert "pm25_lag_1h" in temporal
        assert "fourier_daily_sin_1" in temporal, "Fourier features should be temporal"

    def test_temporal_excludes_target(self):
        """Temporal features must NOT include target column."""
        all_cols = ["pm25", "pm25_lag_1h", "nhiet_do", "is_imputed", "target_1h"]
        exclude = {"is_imputed", "pm25"}
        exclude.update(c for c in all_cols if c.startswith("target_"))
        features = [c for c in all_cols if c not in exclude]

        assert "pm25" not in features
        assert "is_imputed" not in features
        assert "target_1h" not in features
        assert "pm25_lag_1h" in features


class TestLogInverseTransform:
    """Verify log1p/expm1 and scaler inverse produce correct results."""

    def test_scaler_inverse_preserves_values(self):
        """StandardScaler inverse should recover original values."""
        np.random.seed(42)
        values = np.random.normal(15, 5, 100)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(values.reshape(-1, 1)).flatten()
        recovered = scaler.inverse_transform(scaled.reshape(-1, 1)).flatten()
        np.testing.assert_allclose(recovered, values, rtol=1e-10)

    def test_clip_prevents_negative_predictions(self):
        """Clipping should prevent negative PM2.5 predictions."""
        preds = np.array([-5.0, -1.0, 0.0, 5.0, 100.0])
        clipped = np.clip(preds, 0, None)
        assert (clipped >= 0).all()
        assert clipped[3] == 5.0  # unchanged
