# 🧬 PM2.5 Forecasting — L0 Rules & Pointers (Project AGENTS)

> **Tác giả:** trihx
> **Dự án:** Time Series Forecasting (PM2.5) — Luận văn Thạc sĩ CTU QĐ 1799.
> **Trạng thái:** Sắp nghiệm thu / Triển khai Docker.
> **Global rules:** `~/.gemini/AGENTS.md` — ĐỌC TRƯỚC, file này chỉ bổ sung project-specific.

---

## 1. Mục Tiêu Dự Án (Core Objectives)
- Xây dựng pipeline chuẩn học thuật về Data Cleaning, MLOps, Time Series.
- Phát hiện rò rỉ dữ liệu (leakage audit), tập trung vào Multi-Horizon (1h, 6h, 24h).
- Xây dựng Dashboard Scientific Observability đẹp, minh bạch (Streamlit).

## 2. Kiến Trúc (Architecture)
- **Tech Stack:** Python, Streamlit, PyTorch (MPS GPU), LightGBM, ChromaDB.
- **Data Pipeline:** `Raw → Impute → Features (v2) → TSF Split → Train`.
- **Model Baseline:** Persistence (Autocorrelation ~0.99 ở h=1).
- **Evaluating:** MLOps Dashboard với 181/181 tests coverage. So sánh MAE, MASE.

```text
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

## 3. Quy Tắc Sinh Tử (Gotchas & Deadlines)

### 🚨 Data & Leakage Rules
- Mọi thao tác Transform (PCA, Scaler, Imputer) chỉ được phép `fit` trên tập **TRAIN ONLY** để tránh leakage bias. CẤM `fit` full data.
- Leakage Audit R² > 0.99 thì KHÔNG BAO GIỜ được deploy, phải quay lại sửa feature. Mọi feature PHẢI dùng `shift(1)`.
- **Test set = real data ONLY**: Không bao giờ dùng imputed data trong test/eval.
- **MASE mandatory**: Mọi model PHẢI report MASE bên cạnh MAE. MASE<0.1 = leakage flag.
- Mọi đánh giá/chỉ số viết trong luận văn và Dashboard PHẢI lấy từ kết quả chạy thực (trong `dashboard_runs/`). CẤM tự bịa/halucinate MAE/MASE.

### 🎨 Dashboard & UI Rules
- **Zero-Hardcode Policy**: Mọi đoạn text dài (Insights, Lessons, Literature Comparison) trên UI đều PHẢI lưu trong `research/experiments/dashboard_content.json` và load qua `ContentManager`. UI file (`app.py`, `info_cards.py`) CHỈ dành cho layout và data visualization.
- **Visual Theme Framework (VTF)**: Toàn bộ biểu đồ Plotly PHẢI cấu hình thông qua `src/viz/theme.py`. KHÔNG hardcode màu sắc linh tinh gây lỗi tương phản Light/Dark mode. Sử dụng màu Kẽm 500 (`#71717A`) cho text để tương thích cả 2 nền. Cấm để legend/tooltip overlay đè lên dữ liệu chart.
- File CSS của Streamlit Dashboard đã được override: `<div class="kpi-card">`, màu `var(--secondary)`. TUYỆT ĐỐI nhớ xử lý tương phản nền Light/Dark khi tạo card màu tối.

### 💻 Code & Tech Rules
- **KHÔNG** import torch ở top-level → segfault MPS + LightGBM. Dùng lazy import.

### 🔧 Specific Gotchas
| Trap | Fix |
|------|-----|
| PyTorch MPS + LightGBM segfault | Lazy import torch SAU KHI LightGBM xong |
| h=1 target dùng y[t] | PHẢI `shift(-h)` cho MỌI horizon |
| Persistence dùng lag features | PHẢI dùng `df[TARGET_COL]` trực tiếp |
| TFT overfit (7.5K rows) | `hidden_dim=32`, `heads=4`, patience=15 |
| GRU thiếu pm25 input | PHẢI thêm `pm25` vào feature list |
| Embedding English-only | Dùng `paraphrase-multilingual-MiniLM-L12-v2` |
| ChromaDB re-index | Xóa `.chroma_db/` trước khi đổi embedding model |

## 4. Workflows Lệnh
```bash
uv run streamlit run app.py                    # Chạy Dashboard
uv run pytest tests/ -v                        # Chạy Test Suite
uv run python scripts/tft_multi_horizon.py     # TFT experiment
uv run python scripts/dl_multi_horizon.py      # GRU/LSTM
uv run python scripts/multi_horizon_eval.py    # LightGBM (Optuna)
lsof -ti :8501 | xargs kill -9                 # Kill Streamlit port
```

## 5. Cấu trúc Tri Thức (Tiered Memory)
1. **L0:** `.agent/guides/AGENTS.md` (Bạn đang đọc)
2. **L1:** `docs/MEMORY_HOT.md` (Trạng thái hot session - Đọc mỗi phiên)
3. **L2:** `docs/LESSONS_LEARNED.md` (Bài học + fix - Đọc trước khi edit code)
4. **L3:** `docs/DECISIONS_LOG.md` (Ghi log siêu chi tiết để lưu trữ)

## 6. Models & Results (tóm tắt)
| Horizon | Best | MASE | Method |
|---------|------|------|--------|
| 1h | TFT | 1.029 | Attention (autocorr=0.97 → Persistence mạnh) |
| 6h | GRU_ens | **0.698** ⭐ | Stacking ensemble |
| 24h | GRU | **0.727** ⭐⭐ | Recurrent + temporal features |

## 7. AI Chatbot
- **RAG**: ChromaDB + `paraphrase-multilingual-MiniLM-L12-v2`
- **LLM**: LM Studio port `8888` (Gemma 4 E4B / Qwen3-8B)
- **Files**: `src/chatbot/{knowledge_base,llm_client,chat_page}.py`
- **System prompt**: Ưu tiên phương pháp luận > quy trình > kết quả
