# PROJECT CONTEXT

## Trạng thái hiện tại (2026-04-11 10:08)
- **Phase**: ✅ ALL 8 PHASES COMPLETE + Performance Optimization + Docs Sync
- **Best model**: GRU — 24h MASE=0.727 (giảm 27.3% vs Persistence) ⭐⭐
- **Dữ liệu**: 209K records → Hybrid 7,742 rows → 95 features (anti-leakage ✅)
- **Tests**: 133/133 passed ✅
- **Hardware**: Apple M1 Pro 16GB — `torch.device("mps")`, lazy import torch

## Top Model Rankings (MASE)
| Horizon | Best | MASE | Runner-up | MASE |
|---------|------|------|-----------|------|
| **1h** | TFT | 1.029 | ARIMA | 1.023 |
| **6h** | GRU_ens | **0.698** ⭐ | LightGBM | 0.745 |
| **24h** | GRU | **0.727** ⭐⭐ | GRU_ens | 0.730 |

> Persistence rất mạnh ở 1h (autocorr=0.97). ML/DL thắng từ 6h trở lên.

## Pipeline Architecture
```
Raw (209K) → Clean (IQR 3.0, dedup) → Resample 1h → Impute (Spline≤6h + KNN 6-24h)
→ Features (95 cols, shift(1) anti-leakage) → Split 80/10/10 temporal
→ Target: shift(-h) for ALL h → Model → Evaluate (MAE + MASE mandatory)
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

## Tài liệu quan trọng
| File | Mô tả |
|------|--------|
| `app.py` | Dashboard chính (12 pages) |
| `src/chatbot/` | AI Assistant module |
| `docs/THESIS_DRAFT_CTU_1799.md` | Thesis chính thức |
| `.agent/memory/CONTEXT.md` | HOT memory (file này) |
| `.agent/memory/DECISIONS.md` | Quyết định kỹ thuật |
| `.agent/memory/LESSONS_LEARNED.md` | Bài học kinh nghiệm |
| `research/experiments/` | JSON experiment results |
| `models/exported/` | GRU (.pt) + LightGBM (.txt) |

## Execution Commands
```bash
uv run streamlit run app.py                    # Dashboard
uv run pytest tests/ -v                        # Tests
lsof -ti :8501 | xargs kill -9                 # Kill port 8501
```
