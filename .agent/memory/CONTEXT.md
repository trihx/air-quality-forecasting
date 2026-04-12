# PROJECT CONTEXT

## Trạng thái hiện tại (2026-04-12 09:37)
- **Phase**: ✅ ALL 8 PHASES + RC Integration (v6) + Performance Opt + Docs
- **Best model**: GRU v2+log — 6h MASE=0.692 (giảm 31.0% vs Persistence) ⭐⭐⭐
- **Dữ liệu**: 209K records → Hybrid 7,742 rows → 119 features (v2 enhanced, anti-leakage ✅)
- **Models**: 28 models across 5 families (Linear, Tree, DL, Statistical, Ensemble) + PCA/TopN variants
- **Tests**: 154/154 passed ✅
- **Snapshots**: v1→v6 trong `dashboard_runs/` (có chú thích what/why/result/conclusion)
- **Hardware**: Apple M1 Pro 16GB — `torch.device("mps")`, lazy import torch

## Top Model Rankings (MASE) — Updated v6
| Horizon | Best | MASE | Runner-up | MASE | 3rd |
|---------|------|------|-----------|------|-----|
| **1h** | TFT v1 | 1.029 | GRU v1 | 1.173 | Ens_Weighted 1.249 |
| **6h** | **GRU v2+log** | **0.692** ⭐⭐⭐ | GRU_ens | 0.698 | Ens_Weighted 0.705 |
| **24h** | GRU v1 | **0.727** ⭐⭐ | GRU_ens | 0.730 | LSTM v2 0.734 |

> Persistence rất mạnh ở 1h (autocorr=0.99). Feature engineering giúp 6h/24h nhưng HẠI 1h.
> v5: GRU_log 6h = NEW BEST entire pipeline. v6: PCA/TopN không cải thiện 1h. TFT v2 cần hidden lớn hơn.

## Pipeline Architecture
```
Raw (209K) → Clean (IQR 3.0, dedup) → Resample 1h → Impute (Spline≤6h + KNN 6-24h)
→ Features (119 cols: lag+rolling+ewm+diff+calendar+domain+fourier+interaction)
→ Split 80/10/10 temporal → Target: shift(-h) for ALL h
→ Model → Evaluate (MAE+MASE+Classification: Brier/F1 @ WHO thresholds)
```

## Completed Phases (tóm tắt)
1. ✅ **Pipeline & ML** — LightGBM (Optuna 100 trials), ARIMA/SARIMA, leakage audit v2
2. ✅ **Deep Learning** — GRU/LSTM (MPS GPU, batch=256), Ensemble stacking
3. ✅ **Prediction Intervals** — Conformal, Quantile (best 86%), MC Dropout
4. ✅ **TFT** — Pure PyTorch, GRN+GLU+Attention, 25K params, best h=1
5. ✅ **Dashboard** — Streamlit "Scientific Observatory", 12 pages, dark theme
6. ✅ **Export** — GRU TorchScript + LightGBM native, manifest.json
7. ✅ **Docs** — Thesis CTU QĐ 1799, IEEE citations, 5 chapters
8. ✅ **AI Chatbot** — RAG (ChromaDB + multilingual embeddings) + LM Studio
9. ✅ **RC Integration** — v2: Fourier+interactions+log1p. v3: RF+GB+Stacking+Ensemble

## AI Chatbot (Phase 8) — ✅ Done
- **Files**: `src/chatbot/{__init__,knowledge_base,llm_client,chat_page}.py`
- **RAG**: ChromaDB + `paraphrase-multilingual-MiniLM-L12-v2` (50+ languages)
- **LLM**: LM Studio port 8888 — Gemma 4 E4B hoặc Qwen3-8B
- **Features**: Model selector, Re-index button, 20 preset defense questions
- **Status**: ✅ Tested: 241 docs indexed, câu hỏi tiếng Việt match tốt

## Dashboard Performance (2026-04-11) — ✅ Done
- **Issue**: Streamlit file watcher scan 200+ transformers modules → startup chậm 50%
- **Fix**: `fileWatcherType = "none"` + lazy imports trong `main()` → ~1.5-2s startup

## Documentation Sync (2026-04-11) — ✅ Done
- **Walkthrough**: 878 → 1,093 dòng (+§8-11: DL/TFT/Ensemble/PI)
- **Thesis**: §4.7 ordering fix, §5.2/5.3 PI results updated

## Snapshot Versioning (dashboard_runs/)
| Version | Date | What | Best Result |
|---------|------|------|-----------|
| v1_baseline | 2026-04-11 | 13 models, 95 features | GRU 24h MASE=0.727 |
| v2_enhanced | 2026-04-11 | +Fourier+interactions+linear, 119 features | LightGBM 1h MAE↓14.2% |
| v3_sklearn_ensemble | 2026-04-12 | +RF+GB+Stacking+Ensemble, 18 models | Ens_Weighted 6h MASE=0.705 |
| v4_roc_auc | 2026-04-12 | +ROC-AUC metric, Dashboard info cards | Full eval parity with RC |
| v5_dl_retrain | 2026-04-12 | GRU/LSTM retrain+CV+log comparison, 22 models | GRU_log 6h MASE=0.692 (NEW BEST) |
| v6_pca_tft | 2026-04-12 | PCA+TopN feature sel (1h) + TFT v2 retrain, 28 models | 1h fundamentally limited by autocorrelation |

## Tài liệu quan trọng
| File | Mô tả |
|------|--------|
| `app.py` | Dashboard chính (12 pages + version comparison) |
| `src/chatbot/` | AI Assistant module |
| `docs/THESIS_DRAFT_CTU_1799.md` | Thesis chính thức |
| `.agent/memory/CONTEXT.md` | HOT memory (file này) |
| `.agent/memory/DECISIONS.md` | Quyết định kỹ thuật |
| `.agent/memory/LESSONS_LEARNED.md` | Bài học kinh nghiệm |
| `research/experiments/dashboard_runs/` | Snapshot v1→v6 (JSON) |
| `research/experiments/dl_v2/` | DL retrain v2 results |
| `research/experiments/dl_v3/` | PCA + TFT v2 results |
| `models/exported/` | GRU (.pt) + LightGBM (.txt) |

## Execution Commands
```bash
uv run streamlit run app.py                    # Dashboard
uv run pytest tests/ -v                        # Tests
lsof -ti :8501 | xargs kill -9                 # Kill port 8501
```
