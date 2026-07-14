---
name: Time Series Forecasting & Prediction
description: Kim chỉ nam (L0) cho toàn bộ dự án dự báo PM2.5. Định tuyến đến các file chuyên sâu trong .agents/skills/time-series-forecasting/guides/.
---

# 🧠 Time Series Forecasting & Prediction Skill (L0 Index)

> **Mục đích**: Đây là bảng điều khiển trung tâm (Router) cho toàn bộ dự án. 
> Thay vì chứa chi tiết, file này định tuyến bạn đến đúng file hướng dẫn chuyên sâu (`guides/`) khi bạn thực hiện một task cụ thể.
> AI agent **PHẢI** đọc file này và các file trong `docs/MEMORY_HOT.md` **TRƯỚC** khi thực hiện bất kỳ thay đổi nào.

---

## 1. QUY TẮC VÀNG (GOLDEN RULES)

> [!CAUTION]
> Các quy tắc này là **BẮT BUỘC**. Vi phạm bất kỳ quy tắc nào sẽ dẫn đến kết quả không đáng tin cậy.

### 1.1 Gate-keeping & Approval Process

**TRƯỚC KHI thực hiện bất kỳ thay đổi nào**, agent PHẢI:
1. Đọc file này (`.agents/skills/time-series-forecasting/SKILL.md`)
2. Đọc `docs/MEMORY_HOT.md` (Trạng thái hiện tại)
3. Đọc `docs/LESSONS_LEARNED.md` (Các lỗi cần tránh)
4. Trình bày kế hoạch cho user duyệt. Chỉ implement SAU KHI user approve.

**SAU KHI implement**, agent PHẢI:
1. Chạy lint/validate (`uv run ruff check src/` và `uv run pytest`)
2. Ghi kết quả vào `task.md` hoặc báo cáo user.
3. Nếu có lỗi mới → Cập nhật `docs/LESSONS_LEARNED.md`

### 1.2 Nguyên tắc Trung thực Dữ liệu
- **KHÔNG BAO GIỜ** tưởng tượng hoặc bịa đặt kết quả. Mọi số liệu phải đến từ code chạy thực tế.
- Nếu kết quả không như mong đợi → ghi nhận thực tế, phân tích nguyên nhân, KHÔNG chỉnh sửa số liệu.

### 1.3 Nguyên tắc Bảo mật (Security Rules)
- **Path Validation**: Dùng `pathlib.Path.resolve()` để chống path traversal.
- **Model Load**: Chỉ load model từ thư mục `models/` của dự án. Không dùng `joblib.load()` trên file lạ.
- **Exception Handling**: Không dùng bare `except:`. Log toàn bộ traceback.
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

---

## 3. KIẾN TRÚC DỰ ÁN (PROJECT ARCHITECTURE)

```
time-series-forecasting/
├── .agents/skills/time-series-forecasting/guides/          # 🤖 Các hướng dẫn chuyên sâu (Router L1/L2)
├── configs/                # ⚙️ Cấu hình tập trung (.yaml)
├── dataset/                # 📊 Dữ liệu (raw, interim, processed)
├── docs/                   # 📚 Tài liệu tham khảo, MEMORY_HOT.md, THESIS
├── src/                    # 🔧 Source code chính
│   ├── api/                # FastAPI Backend
│   ├── data/               # Load, clean, split
│   ├── features/           # Feature engineering
│   ├── models/             # Định nghĩa Base Model và ML/DL models
│   ├── evaluation/         # Metrics
│   ├── pipelines/          # End-to-end (train, predict)
│   ├── frontend/           # Streamlit API Client
│   ├── viz/                # Visual Theme Framework (VTF)
│   └── utils/              # Utilities (logger, etc.)
├── tests/                  # 🧪 Unit & Integration Tests
├── app.py                  # 🌐 Streamlit Dashboard UI
├── main.py                 # FastAPI Entrypoint
└── pyproject.toml          # Quản lý dependencies bằng uv
```

---

## 4. QUICK COMMANDS (UV)

```bash
# === Môi trường & Dependencies ===
uv sync                                    # Cài đặt dependencies
uv add <package>                           # Thêm dependency

# === Quality & Testing (BẮT BUỘC) ===
uv run ruff check src/ --fix               # Lint & auto-fix
uv run ruff format src/                    # Format code
uv run mypy src/                           # Type check
uv run pytest tests/ -v                    # Chạy toàn bộ tests

# === Chạy Ứng dụng ===
uv run uvicorn src.api.main:app --reload   # Chạy Backend (FastAPI)
uv run streamlit run app.py                # Chạy Frontend (Streamlit)
```
