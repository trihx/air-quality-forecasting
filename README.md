# 🌫️ Time Series Forecasting — PM2.5 Air Quality Prediction

> Dự báo nồng độ PM2.5 bằng ML/DL từ dữ liệu IoT sensor (~2 phút/mẫu, 3+ năm).

## Quick Start

```bash
# Cài đặt dependencies
uv sync

# Chạy lint check
uv run ruff check src/ --fix && uv run ruff format src/

# Chạy tests
uv run pytest tests/ -v --cov=src

# Chạy experiment
uv run python scripts/run_experiment.py --model xgboost --config configs/model_configs/xgboost.yaml
```

## Project Structure

```
├── .agent/          # 🤖 AI Agent Skill System (SKILL.md, guides, workflows, memory)
├── configs/         # ⚙️ YAML configs (base + model-specific)
├── dataset/         # 📊 raw/ → interim/ → processed/
├── docs/            # 📚 31 reference books & papers
├── src/             # 🔧 Source code (data, features, models, evaluation, pipelines)
├── tests/           # 🧪 unit/ + integration/ + validation/
├── models/          # 💾 Trained model artifacts
├── research/        # 🔬 Experiment runs + Optuna studies
├── notebooks/       # 📓 EDA & prototyping
└── pyproject.toml   # Dependencies (managed by uv)
```

## Dataset

| Property | Value |
|----------|-------|
| Records | 209,594 |
| Time Span | 2022-03-25 → 2025-05-11 (~3.1 years) |
| Sampling | ~2 min/record |
| Features | nhiet_do (°C), do_am (%), diem_suong (°C), co2 (ppm) |
| Target | pm25 (µg/m³) |

## Model Progression

```
Level 0: Naive Baselines  → Persistence, Seasonal Naive, Mean
Level 1: Statistical       → ARIMA, SARIMA, SARIMAX
Level 2: Linear ML         → Ridge, Lasso, ElasticNet
Level 3: Tree-based ML     → Random Forest, XGBoost, LightGBM
Level 4: Deep Learning     → LSTM, GRU, Transformer
Level 5: Ensemble          → Stacking, Blending
Level 6: AutoML            → Optuna + Best model
```

## Evaluation Metrics

| Metric | Role |
|--------|------|
| **MAE** | Primary metric |
| **MASE** | Mandatory benchmark (< 1.0 = better than naive) |
| RMSE | Penalizes large errors |
| R² | Overall fit |

## Tech Stack

- **Language**: Python 3.11
- **Package Manager**: uv
- **ML**: scikit-learn, XGBoost, LightGBM
- **DL**: PyTorch
- **Visualization**: matplotlib, seaborn, plotly
- **Quality**: ruff, bandit, mypy, pytest
- **Optimization**: Optuna
- **Explainability**: SHAP

## Development

Xem `.agent/SKILL.md` cho quy trình chi tiết. Mọi thay đổi code phải tuân thủ Quality Loop:

```
Ruff Check → Ruff Format → Bandit → MyPy → Pytest → ✅ Commit
```
