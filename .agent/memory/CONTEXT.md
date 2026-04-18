# PROJECT CONTEXT

## Trạng thái hiện tại (2026-04-12 20:15)
- **Status**: 💯 Hoàn tất mọi mục tiêu trong tuần (Đã chèn nội dung khoa học vào bản nháp luận văn $\rightarrow$ đợi tuần sau tiếp tục triển khai tiếp theo mong muốn của User).
- **Phase**: ✅ 100% hoàn thành (Phase 1-6 + v7-retrain + Thesis integration).
- **Knowledge Base**: `docs/KNOWLEDGE_BASE.md` — 30 cuốn sách → kiến thức pipeline (LUÔN đọc trước implement)
- **Best model**: GRU v2+log — 6h MASE=0.692 (giảm 31.0% vs Persistence) ⭐⭐⭐
- **Dữ liệu**: 209K records → Hybrid 7,742 rows → 119 features (v2 enhanced, anti-leakage ✅, Purging Gap ✅)
- **Models**: 28 models across 5 families + PCA/TopN variants
- **Tests**: 167/167 passed ✅ (v7: +13 audit tests)
- **Snapshots**: v1→v7 (Audit & S-ESD) trong `dashboard_runs/` + Tích hợp thành công `THESIS_DRAFT_CTU_1799.md`.
- **Hardware**: Apple M1 Pro 16GB — `torch.device("mps")`, lazy import torch

## 🚨 DATA INTEGRITY RULES (TUYỆT ĐỐI)
1. **KHÔNG tưởng tượng số liệu** — mọi metric PHẢI reference từ JSON/log output thực tế
2. **Mọi transform FIT trên TRAIN ONLY** — STL, PCA, BoxCox, Scaler. KHÔNG fit full data
3. **Leakage audit sau MỖI experiment** — check MASE < 0.1 hoặc R² > 0.99 = red flag
4. **Test set = REAL data only** — `is_imputed==0` filter BẮT BUỘC
5. **Cross-reference trước khi ghi Dashboard** — đọc lại JSON output, KHÔNG copy từ memory
6. **Shift(1) cho mọi feature dùng target** — diff, pct_change, ratio, rolling trên pm25
7. **Cập nhật `docs/PIPELINE_REFERENCES.md`** sau MỖI task — ghi nguồn IEEE (Book Ch.X, pp.Y) cho quyết định mới

## Top Model Rankings (MASE) — v7-retrain (standardized)
| Horizon | Best | MASE | Runner-up | MASE | 3rd |
|---------|------|------|-----------|------|-----|
| **1h** | Persistence | 1.000 | Ens_Weighted | 1.239 | LSTM_v2_log 1.300 |
| **6h** | **Ens_Weighted** | **0.703** ⭐⭐⭐ | RF | 0.705 | LightGBM 0.733 |
| **24h** | **LSTM_v2_log** | **0.691** ⭐⭐⭐ | ARIMA | 0.764 | Ens_Weighted 0.788 |

> Persistence vẫn mạnh nhất ở 1h (autocorr~0.99). ML > DL ở 6h, DL > ML ở 24h.
> DL MASE standardized: Unified Persistence MAE (2.49, 6.42, 6.45) across all families.
> v7-retrain: Outlier fix PM2.5 domain [0,500]. All in `research/experiments/v7_retrain/`.

## Pipeline Architecture
```
Raw (209K) → Clean (PM2.5: domain [0,500], others: IQR 3.0) → Resample 1h → Impute (Spline≤6h + KNN 6-24h)
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
| v7_audit | 2026-04-12 | Pipeline Audit: +5 EDA charts, +4 metrics, deseasonalizing exp | Fourier makes deseasonalizing redundant |

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
