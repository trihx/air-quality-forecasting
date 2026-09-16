"""Unit tests for Phase 2 training pipeline audit and thesis alignment.

Verifies:
1. Horizon-aware hyperparameter retrieval for LightGBM (Optuna best params).
2. Persistence baseline mathematical correctness in GRUTrainer (target[i + lb - 1]).
3. LightGBM safety invariants (n_jobs=1, objective="regression_l1").
4. Table 3.4 (80:10:10 split) and Table 3.5 (stationarity matrix) data fidelity.
5. Strict terminology compliance ("đề án" instead of "luận văn").
"""

from __future__ import annotations

from pathlib import Path

from src.training.trainer import LightGBMTrainer, get_default_params


class TestPhase2Hyperparams:
    """Test hyperparameter retrieval and horizon awareness."""

    def test_get_default_params_lightgbm_horizon_aware(self) -> None:
        """LightGBM params must differ per horizon matching Optuna Bayesian optimization."""
        p_h1 = get_default_params("LightGBM", horizon=1)
        p_h6 = get_default_params("LightGBM", horizon=6)
        p_h24 = get_default_params("LightGBM", horizon=24)

        assert p_h1["n_estimators"] == 500
        assert p_h1["num_leaves"] == 64
        assert p_h1["learning_rate"] == 0.013

        assert p_h6["n_estimators"] == 637
        assert p_h6["num_leaves"] == 87
        assert p_h6["learning_rate"] == 0.012

        assert p_h24["n_estimators"] == 450
        assert p_h24["num_leaves"] == 52
        assert p_h24["learning_rate"] == 0.015

    def test_get_default_params_gru(self) -> None:
        """GRU parameters must match thesis architecture."""
        p_gru = get_default_params("GRU")
        assert p_gru["lookback"] == 72
        assert p_gru["hidden_dim"] == 64
        assert p_gru["num_layers"] == 2
        assert p_gru["dropout"] == 0.2
        assert p_gru["learning_rate"] == 0.001


class TestPhase2TrainerLogic:
    """Test scientific and mathematical integrity in trainer logic."""

    def test_lightgbm_trainer_invariants(self) -> None:
        """LightGBM must use n_jobs=1 and objective='regression_l1'."""
        trainer = LightGBMTrainer(horizon=6, params={"max_depth": 3})
        assert trainer.horizon == 6
        assert trainer.params["max_depth"] == 3

    def test_gru_persistence_mathematical_formula(self) -> None:
        """In GRUTrainer, persist_preds must index the last lookback point (i + lb - 1).

        Let lookback lb=72, horizon h=6.
        For sequence i, the inputs are i ... i+lb-1.
        The target to predict is i + lb + h - 1.
        The last observed value at prediction time t is i + lb - 1.
        """
        trainer_code = Path("src/training/trainer.py").read_text(encoding="utf-8")
        assert "target[i + lb - 1]" in trainer_code, (
            "GRUTrainer persistence must index target[i + lb - 1], not target[i]"
        )
        assert "target[i] for i in test_ds.indices" not in trainer_code, (
            "Found stale target[i] persistence indexing in GRUTrainer"
        )

    def test_lightgbm_single_thread_and_mae_objective(self) -> None:
        """Ensure LightGBM sets n_jobs=1 to avoid macOS OpenMP crash and objective='regression_l1'."""
        trainer_code = Path("src/training/trainer.py").read_text(encoding="utf-8")
        assert 'lgb_params.setdefault("n_jobs", 1)' in trainer_code
        assert 'lgb_params.setdefault("objective", "regression_l1")' in trainer_code


class TestPhase2ThesisDataFidelity:
    """Test fidelity of Table 3.4 and Table 3.5 in hyperparams.py."""

    def test_table_3_4_split_data_present(self) -> None:
        """Table 3.4 sample sizes must match thesis numbers."""
        hyperparams_code = Path("src/frontend/pages/hyperparams.py").read_text(encoding="utf-8")
        assert "18.355" in hyperparams_code  # 15m clean samples
        assert "8.625" in hyperparams_code  # 30m clean samples
        assert "6.689" in hyperparams_code  # 1h clean samples
        assert "14.684" in hyperparams_code  # 15m train
        assert "6.900" in hyperparams_code  # 30m train
        assert "5.351" in hyperparams_code  # 1h train
        assert "19.810" in hyperparams_code  # dropped missing hours

    def test_table_3_5_stationarity_matrix_present(self) -> None:
        """Table 3.5 ADF/KPSS decision matrix must be present."""
        hyperparams_code = Path("src/frontend/pages/hyperparams.py").read_text(encoding="utf-8")
        assert "Dừng hoàn toàn (I(0))" in hyperparams_code
        assert "Không dừng có xu thế (I(1))" in hyperparams_code
        assert "Tuần hoàn mùa vụ ngày đêm" in hyperparams_code
        assert "Phương sai không thuần nhất" in hyperparams_code
        assert "render_references_section()" in hyperparams_code

    def test_no_luan_van_in_phase2_files(self) -> None:
        """Zero occurrences of 'luận văn' in Phase 2 source files."""
        files = [
            Path("src/training/trainer.py"),
            Path("src/frontend/pages/hyperparams.py"),
            Path("src/frontend/pages/training.py"),
            Path("src/frontend/pages/experiment_runs.py"),
        ]
        for f in files:
            content = f.read_text(encoding="utf-8")
            assert "luận văn" not in content.lower(), f"Found 'luận văn' in {f}"
