---
name: Time Series Forecasting & Prediction
description: Kim chỉ nam (L0) cho toàn bộ dự án dự báo PM2.5. Định tuyến đến các file chuyên sâu trong .agents/skills/time-series-forecasting/guides/.
---

# 🧠 Time Series Forecasting & Prediction Skill (L0 Index)

> **Dự án:** Time Series Forecasting (PM2.5) — Luận văn Thạc sĩ CTU QĐ 1799.
> **Mục đích**: Đây là bảng điều khiển trung tâm (Router) cho toàn bộ dự án.
> Thay vì chứa chi tiết, file này định tuyến đến đúng file hướng dẫn chuyên sâu (`guides/`) khi thực hiện task cụ thể.
> **Global rules:** `~/.gemini/AGENTS.md` — ĐỌC TRƯỚC, file này chỉ bổ sung project-specific.

---

## 1. QUY TẮC VÀNG (GOLDEN RULES)

> [!CAUTION]
> Các quy tắc này là **BẮT BUỘC**. Vi phạm bất kỳ quy tắc nào sẽ dẫn đến kết quả không đáng tin cậy.

### 1.1 Gate-keeping & Approval Process

**TRƯỚC KHI thực hiện bất kỳ thay đổi nào**, agent PHẢI:
1. Đọc file này (SKILL.md)
2. Đọc `docs/MEMORY_HOT.md` khi task liên quan trạng thái dự án
3. Đọc `docs/LESSONS_LEARNED.md` khi viết code logic phức tạp hoặc fix bug
4. Trình bày kế hoạch cho user duyệt. Chỉ implement SAU KHI user approve.

**SAU KHI implement**, agent PHẢI:
1. Chạy `make check` (lint + test)
2. Ghi kết quả vào `task.md` hoặc báo cáo user.
3. Nếu có lỗi mới → Cập nhật `docs/LESSONS_LEARNED.md`

### 1.2 Nguyên tắc Trung thực Dữ liệu
- **KHÔNG BAO GIỜ** tưởng tượng hoặc bịa đặt kết quả. Mọi số liệu phải đến từ code chạy thực tế.
- Nếu kết quả không như mong đợi → ghi nhận thực tế, phân tích nguyên nhân, KHÔNG chỉnh sửa số liệu.

### 1.3 Nguyên tắc Bảo mật
- **Path Validation**: Dùng `pathlib.Path.resolve()` để chống path traversal.
- **Model Load**: Chỉ load model từ thư mục `models/` của dự án. Không dùng `joblib.load()` trên file lạ.
- **Secrets**: KHÔNG commit API keys hay credentials. Dùng `python-dotenv`.

---

## 2. BẢNG ĐIỀU HƯỚNG TÀI LIỆU (GUIDES ROUTER)

> [!IMPORTANT]
> Khi nhận một task cụ thể, **HÃY ĐỌC** file guide tương ứng trước khi viết code.

| Bạn đang làm việc về... | Hãy đọc file này |
|-------------------------|------------------|
| Lập kế hoạch, tạo task | [concise-planning.md](guides/concise-planning.md) |
| Cải tiến code, refactor | [kaizen.md](guides/kaizen.md) |
| Code quality, Unit tests | [lint-and-validate.md](guides/lint-and-validate.md) |
| Fix bug, gặp lỗi pipeline | [systematic-debugging.md](guides/systematic-debugging.md) |
| Logging, lưu experiment | [logging.md](guides/logging.md) |
| Tải data, Clean, Features | [data-engineering.md](guides/data-engineering.md) |
| Vẽ biểu đồ, EDA, Phân tích | [visualization-storytelling.md](guides/visualization-storytelling.md) |
| Thiết kế Experiment, Baseline | [analytics-experiment-design.md](guides/analytics-experiment-design.md) |
| Train Models (ML, DL, Optuna) | [model-training.md](guides/model-training.md) |
| Tính MAE, MASE, SHAP | [evaluation-metrics.md](guides/evaluation-metrics.md) |
| Dashboard UI, ClickHouse Style | [DESIGN.md](guides/DESIGN.md) |
| Viết luận văn, IEEE format | [academic-writing.md](guides/academic-writing.md) |

---

## 3. KIẾN TRÚC DỰ ÁN (PROJECT ARCHITECTURE)

```
time-series-forecasting/
├── .agents/skills/.../guides/   # 🤖 Hướng dẫn chuyên sâu (Lazy-Load)
├── configs/                     # ⚙️ Cấu hình tập trung (.yaml)
├── dataset/                     # 📊 Dữ liệu (raw, interim, processed)
├── docs/                        # 📚 MEMORY_HOT, LESSONS_LEARNED, THESIS
├── src/                         # 🔧 Source code chính
│   ├── api/                     # FastAPI Backend
│   ├── data/                    # Load, clean, split
│   ├── features/                # Feature engineering
│   ├── models/                  # ML/DL models
│   ├── evaluation/              # Metrics
│   ├── pipelines/               # End-to-end (train, predict)
│   ├── frontend/                # Streamlit API Client
│   ├── viz/                     # Visual Theme Framework (VTF)
│   └── utils/                   # Utilities (logger, etc.)
├── tests/                       # 🧪 Unit & Integration Tests
├── app.py                       # 🌐 Streamlit Dashboard UI
├── main.py                      # FastAPI Entrypoint
├── Makefile                     # 🔧 Automation hub
└── pyproject.toml               # Dependencies (uv)
```

---

## 4. PROJECT-SPECIFIC GOTCHAS

| Trap | Fix |
|------|-----|
| PyTorch MPS + LightGBM segfault | Lazy import torch SAU KHI LightGBM xong |
| h=1 target dùng y[t] | PHẢI `shift(-h)` cho MỌI horizon |
| Persistence dùng lag features | PHẢI dùng `df[TARGET_COL]` trực tiếp |
| TFT overfit (7.5K rows) | `hidden_dim=32`, `heads=4`, patience=15 |
| GRU thiếu pm25 input | PHẢI thêm `pm25` vào feature list |
| Embedding English-only | Dùng `paraphrase-multilingual-MiniLM-L12-v2` |
| ChromaDB re-index | Xóa `.chroma_db/` trước khi đổi embedding model |
| Anti-Leakage | KHÔNG dùng target tại t trong features. `diff(y)` chứa y[t] → dùng `shift(1).diff()` |
| Tiered Imputation | Spline (gap ≤6h) + KNN (6-24h) + Drop (>24h). KHÔNG univariate cho gap >6h |
| Test-on-Real-Only | Imputed data chỉ dùng train. Test BẮT BUỘC `is_imputed == 0` |
| statsmodels + gaps | Dùng `.values` (numpy) khi truyền ARIMA/SARIMAX. `forecast()` trả numpy → `[-1]` |
| Autocorrelation trap | autocorr >0.95 → Persistence rất mạnh ở h=1 → focus ML thắng ở 6h, 24h |

---

## 5. MODELS & RESULTS

| Horizon | Best | MASE | Method |
|---------|------|------|--------|
| 1h | TFT | 1.029 | Attention (autocorr=0.97 → Persistence mạnh) |
| 6h | GRU_ens | **0.698** ⭐ | Stacking ensemble |
| 24h | GRU | **0.727** ⭐⭐ | Recurrent + temporal features |

## 6. AI CHATBOT

- **RAG**: ChromaDB + `paraphrase-multilingual-MiniLM-L12-v2`
- **LLM**: LM Studio port `8888` (Gemma 4 E4B / Qwen3-8B)
- **Files**: `src/chatbot/{knowledge_base,llm_client,chat_page}.py`
- **System prompt**: Ưu tiên phương pháp luận > quy trình > kết quả
