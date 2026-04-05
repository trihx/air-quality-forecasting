# PROJECT CONTEXT

## Trạng thái hiện tại (2026-04-05 10:48)
- **Phase**: ✅ ALL 5 PHASES COMPLETE — Pipeline + Experiments + Dashboard + Export + TFT
- **Best model (1h)**: TFT (MAE=2.573, **MASE=1.029**) — tiệm cận Persistence, tốt nhất trong ML/DL
- **Best model (6h)**: ⭐ GRU_ens (MAE=4.729, **MASE=0.698**) — giảm 30% vs Persistence!
- **Best model (24h)**: ⭐⭐ GRU (MAE=4.562, **MASE=0.727**) — giảm 27.3%!
- **TFT Results**: 1h=1.029 | 6h=0.822 | 24h=0.812 (25,089 params, simplified architecture)
- **Strategy tốt nhất**: Hybrid (Spline≤6h + KNN 6-24h)
- **Dữ liệu**: `dataset/raw/final_dataset.csv` - 209,594 bản ghi, 6 cột
- **Hybrid**: 7,742 rows → Features: 7,574 rows × 95 cols
- **Tests**: 133/133 passed ✅ (including leakage tests + validator tests)
- **Hardware**: Apple Silicon M1 Pro — dùng `torch.device("mps")` cho GPU. Lazy import torch.

## ✅ Completed Milestones

### Phase 1: Pipeline & Experiments (2026-04-04)
- ✅ Pipeline Audit v2 — Fixed 2 critical bugs (h=1 target, Persistence formula)
- ✅ Multi-horizon LightGBM (Optuna) — h=6: 0.745, h=24: 0.842
- ✅ ARIMA/SARIMA — SARIMA 24h MASE=0.813
- ✅ DL (GRU/LSTM) — GRU 24h **MASE=0.727** ⭐⭐ BEST
- ✅ Ensemble (LightGBM+GRU stacking) — GRU_ens 6h MASE=0.698 ⭐
- ✅ SHAP Explainability + DM test + Residual diagnostics

### Phase 2: Prediction Intervals (2026-04-05)
- ✅ Conformal Prediction (LightGBM) — coverage ~80%
- ✅ Quantile Regression (LightGBM) — coverage ~86% (best)
- ✅ MC Dropout (GRU) — coverage ~37% (needs calibration)
- ✅ Script: `scripts/prediction_intervals.py`

### Phase 3: Streamlit Dashboard (2026-04-05)
- ✅ "Scientific Observatory" dark theme + teal accent (#00D4AA)
- ✅ 6 pages: Overview, Multi-Horizon, SHAP, PI, EDA, Hyperparameters
- ✅ TFT integrated into all charts + ranking table
- ✅ `app.py` — HTTP 200 verified

### Phase 4: Model Export (2026-04-05)
- ✅ GRU → TorchScript (.pt) for 3 horizons — `models/exported/`
- ✅ LightGBM → Native .txt for 3 horizons
- ✅ Scaler metadata + manifest.json

### Phase 5: TFT — Temporal Fusion Transformer (2026-04-05)
- ✅ Simplified TFT in pure PyTorch (no pytorch-forecasting dep)
- ✅ Components: GRN, GLU, Multi-head Attention (4 heads), Static Encoder
- ✅ Script: `scripts/tft_multi_horizon.py`
- ✅ Results: 1h=1.029 | 6h=0.822 | 24h=0.812

### Phase 6: Documentation (2026-04-05)
- ✅ Thesis Draft Ch1-5 + TFT results integrated + [13] Lim et al. citation
- ✅ Dashboard updated with TFT in all charts/tables
- ✅ Lint fixed: scripts clean, only HTML template E501 remaining

### Phase 7: AI Chatbot — Trợ Lý Dự Án (2026-04-05)
- ✅ RAG pipeline: ChromaDB + sentence-transformers (all-MiniLM-L6-v2)
- ✅ LLM client: OpenAI SDK → LM Studio (port 8888)
- ✅ Knowledge Base: auto-index docs + experiment JSONs + thesis
- ✅ System prompt: ưu tiên phương pháp luận + chuẩn bị phản biện
- ✅ Chat UI: Streamlit chat_message + streaming + preset defense questions
- ✅ Files: `src/chatbot/{__init__,knowledge_base,llm_client,chat_page}.py`
- Model: Gemma 4 E4B (Q4, ~4-5GB) hoặc Qwen3-8B

## Final Rankings (ALL 9 MODELS)

### h=1: TFT closest to Persistence
| Model | MASE | Note |
|-------|------|------|
| Persistence | 1.000 | Baseline (autocorr=0.97) |
| **TFT** | **1.029** | ⭐ Best ML/DL at h=1 |
| GRU | 1.173 | |
| ARIMA | 1.023 | |
| SARIMA | 1.283 | |
| LightGBM | 1.492 | |
| LSTM | 1.560 | |

### h=6: ML/DL đều thắng
| Model | MASE | Source |
|-------|------|--------|
| GRU_ens | **0.698** | ensemble |
| LightGBM | 0.745 | multi_horizon_v2 |
| SARIMA | 0.762 | arima |
| GRU | 0.812 | dl |
| TFT | 0.822 | tft |
| ARIMA | 0.856 | arima |
| LSTM | 0.914 | dl |

### h=24: GRU dominates
| Model | MASE | Source |
|-------|------|--------|
| **GRU** | **0.727** | dl |
| GRU_ens | 0.730 | ensemble |
| Stack(Ridge) | 0.784 | ensemble |
| TFT | 0.812 | tft |
| SARIMA | 0.813 | arima |
| LSTM | 0.830 | dl |
| LightGBM | 0.842 | multi_horizon_v2 |
| ARIMA | 0.913 | arima |

## Pipeline Architecture
```
Raw (209K rows)
  → Clean (dedup, bounds, outliers IQR 3.0, resample 1h)
  → Impute (Hybrid: Spline≤6h + KNN 6-24h) — is_imputed tracking
  → Features (95 cols, anti-leakage ✅)
  → Split (80/10/10 temporal, TEST = REAL DATA ONLY)
  → Target: shift(-horizon) for ALL h (including h=1)
  → Persist: y[t] directly from TARGET_COL
  → Model → Evaluate (MAE primary, MASE mandatory)
```

## Project Inventory
| Category | Count |
|----------|-------|
| Scripts | 19 |
| Test files | 9 (133 tests) |
| Experiments (.json) | 11 |
| Exported models | 13 files |
| Figures (.png) | 21 |

## Tài liệu quan trọng
- `.agent/memory/CONTEXT.md` — File này (HOT memory)
- `.agent/memory/RUNS_LOG.md` — Lịch sử experiments
- `.agent/memory/LESSONS_LEARNED.md` — Bài học
- `docs/THESIS_DRAFT_CTU_1799.md` — Thesis chính thức (có TFT + [13] Lim et al.)
- `app.py` — Streamlit Dashboard (Scientific Observatory, có TFT)
- `models/exported/` — GRU TorchScript + LightGBM Native
- `research/experiments/tft/` — TFT experiment results
- `research/experiments/prediction_intervals/` — PI experiment results
- `research/best_models_configs.json` — Hyperparams tổng hợp

## Execution Commands
```bash
uv run streamlit run app.py                           # Dashboard
uv run python scripts/prediction_intervals.py         # PI experiment
uv run python scripts/export_models.py                # Model export
uv run python scripts/tft_multi_horizon.py            # TFT experiment
uv run pytest tests/ -v                               # Full test suite
```

## Hyperparameter Tuning Strategy
- **ML (LightGBM)**: Optuna Bayesian (TPE) — 100 trials, TimeSeriesSplit(5), minimize MAE
- **DL (GRU, LSTM)**: Manual + Early Stopping (patience=10)
- **TFT**: Manual — hidden=32, heads=4, patience=15 (simplified for small dataset)
- **Best params lưu tại**: `research/best_models_configs.json`
