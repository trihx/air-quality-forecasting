# PM2.5 Forecasting — Project AGENTS.md (L0)

> **Dự án**: Dự báo nồng độ PM2.5 — Đề án Thạc sĩ, ĐH Cần Thơ
> **Cập nhật**: 2026-04-11
> **Global rules**: `~/.gemini/AGENTS.md` — ĐỌC TRƯỚC, file này chỉ bổ sung project-specific.

---

## 🔒 Quy Tắc Dự Án

1. **LUÔN** đọc: Global AGENTS → file này → `.agent/memory/CONTEXT.md` → `LESSONS_LEARNED.md`
2. **KHÔNG** import torch ở top-level → segfault MPS + LightGBM. Dùng lazy import.
3. **Anti-leakage bắt buộc**: Mọi feature PHẢI dùng `shift(1)`. R²>0.99 = 🚨 audit ngay.
4. **Test set = real data ONLY**: Không bao giờ dùng imputed data trong test/eval.
5. **MASE mandatory**: Mọi model PHẢI report MASE bên cạnh MAE. MASE<0.1 = leakage flag.

---

## 🏗️ Kiến Trúc

```
dataset/raw/           → Dữ liệu gốc IoT (209K records, 6 cols)
src/
├── pipeline/          → Clean, impute, feature engineering
├── evaluation/        → Metrics (MAE, RMSE, MASE, R²)
├── chatbot/           → AI Assistant (RAG + LLM)
scripts/               → Experiment scripts (ML, DL, TFT, PI)
research/experiments/  → JSON experiment results
models/exported/       → GRU (.pt) + LightGBM (.txt)
app.py                 → Streamlit Dashboard (12 pages)
docs/THESIS_DRAFT_CTU_1799.md → Thesis chính thức (IEEE, QĐ 1799)
```

### Pipeline
```
Raw (209K) → Clean (IQR 3.0, dedup) → Resample 1h
→ Impute (Spline≤6h + KNN 6-24h) → Features (95 cols, shift(1))
→ Split 80/10/10 temporal → Target: shift(-h) → Model → Evaluate
```

---

## ⚡ Commands

```bash
uv run streamlit run app.py                    # Dashboard
uv run pytest tests/ -v                        # Tests (133 tests)
uv run python scripts/tft_multi_horizon.py     # TFT experiment
uv run python scripts/dl_multi_horizon.py      # GRU/LSTM
uv run python scripts/multi_horizon_eval.py    # LightGBM (Optuna)
lsof -ti :8501 | xargs kill -9                 # Kill Streamlit port
```

---

## ⚠️ Gotchas (Project-Specific)

| Trap | Fix |
|------|-----|
| PyTorch MPS + LightGBM segfault | Lazy import torch SAU KHI LightGBM xong |
| h=1 target dùng y[t] | PHẢI `shift(-h)` cho MỌI horizon |
| Persistence dùng lag features | PHẢI dùng `df[TARGET_COL]` trực tiếp |
| TFT overfit (7.5K rows) | `hidden_dim=32`, `heads=4`, patience=15 |
| GRU thiếu pm25 input | PHẢI thêm `pm25` vào feature list |
| Embedding English-only | Dùng `paraphrase-multilingual-MiniLM-L12-v2` |
| ChromaDB re-index | Xóa `.chroma_db/` trước khi đổi embedding model |

---

## 🧠 Bộ Nhớ Dự Án

| Tầng | File | Mô tả |
|------|------|-------|
| L0 | `AGENTS.md` (file này) | Rules + Architecture (≤100 dòng) |
| L1 | `.agent/memory/CONTEXT.md` | Trạng thái + Rankings + TODO |
| L2 | `.agent/memory/LESSONS_LEARNED.md` | Bug patterns (89 dòng) |
| L3 | `.agent/memory/DECISIONS.md` | Quyết định kỹ thuật (84 dòng) |
| Log | `.agent/memory/RUNS_LOG.md` | Experiment history |

---

## 📊 Models & Results (tóm tắt)

| Horizon | Best | MASE | Method |
|---------|------|------|--------|
| 1h | TFT | 1.029 | Attention (autocorr=0.97 → Persistence mạnh) |
| 6h | GRU_ens | **0.698** ⭐ | Stacking ensemble |
| 24h | GRU | **0.727** ⭐⭐ | Recurrent + temporal features |

**9 models**: Persistence, ARIMA, SARIMA, LightGBM, LSTM, GRU, GRU_ens, Stack(Ridge), TFT

---

## 🤖 AI Chatbot

- **RAG**: ChromaDB + `paraphrase-multilingual-MiniLM-L12-v2`
- **LLM**: LM Studio port `8888` (Gemma 4 E4B / Qwen3-8B)
- **Files**: `src/chatbot/{knowledge_base,llm_client,chat_page}.py`
- **System prompt**: Ưu tiên phương pháp luận > quy trình > kết quả
