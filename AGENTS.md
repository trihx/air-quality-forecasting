# 🧬 PM2.5 Forecasting — L0 Rules & Pointers

> **Tác giả:** trihx
> **Dự án:** Time Series Forecasting (PM2.5) — Luận văn Thạc sĩ CTU QĐ 1799.
> **Trạng thái:** Sắp nghiệm thu / Triển khai Docker.

## 1. Mục Tiêu Dự Án (Core Objectives)
- Xây dựng pipeline chuẩn học thuật về Data Cleaning, MLOps, Time Series.
- Phát hiện rò rỉ dữ liệu (leakage audit), tập trung vào Multi-Horizon (1h, 6h, 24h).
- Xây dựng Dashboard Scientific Observability đẹp, minh bạch (Streamlit).

## 2. Kiến Trúc (Architecture)
- **Tech Stack:** Python, Streamlit, PyTorch (MPS GPU), LightGBM, ChromaDB.
- **Data Pipeline:** `Raw → Impute → Features (v2) → TSF Split → Train`.
- **Model Baseline:** Persistence (Autocorrelation ~0.99 ở h=1).
- **Evaluating:** MLOps Dashboard với 167/167 tests coverage. So sánh MAE, MASE.

## 3. Quy Tắc Sinh Tử (Gotchas & Deadlines)
- Mọi đánh giá/chỉ số viết trong luận văn và Dashboard PHẢI lấy từ kết quả chạy thực (trong `dashboard_runs/`). CẤM tự bịa/halucinate MAE/MASE.
- Mọi thao tác Transform (PCA, Scaler, Imputer) chỉ được phép `fit` trên tập **TRAIN ONLY** để tránh leakage bias. CẤM `fit` full data.
- Leakage Audit R² > 0.99 thì KHÔNG BAO GIỜ được deploy, phải quay lại sửa feature.
- File CSS của Streamlit Dashboard đã được override: `<div class="kpi-card">`, màu `var(--secondary)`. TUYỆT ĐỐI nhớ xử lý tương phản nền Light/Dark khi tạo card màu tối.
- **Zero-Hardcode Policy**: Mọi đoạn text dài (Insights, Lessons, Literature Comparison) trên UI đều PHẢI lưu trong `research/experiments/dashboard_content.json` và load qua `ContentManager`. UI file (`app.py`, `info_cards.py`) CHỈ dành cho layout và data visualization.
- **Visual Theme Framework (VTF)**: Toàn bộ biểu đồ Plotly PHẢI cấu hình thông qua `src/viz/theme.py`. KHÔNG hardcode màu sắc linh tinh gây lỗi tương phản Light/Dark mode. Sử dụng màu Kẽm 500 (`#71717A`) cho text để tương thích cả 2 nền. Cấm để legend/tooltip overlay đè lên dữ liệu chart.

## 4. Workflows Lệnh
- Chạy Dashboard: `uv run streamlit run app.py`
- Chạy Test Suite: `uv run pytest tests/ -v`

## 5. Cấu trúc Tri Thức (Tiered Memory)
1. **L0:** `AGENTS.md` (Bạn đang đọc)
2. **L1:** `docs/MEMORY_HOT.md` (Trạng thái hot session - Đọc mỗi phiên)
3. **L2:** `docs/LESSONS_LEARNED.md` (Bài học + fix - Đọc trước khi edit code)
4. **L3:** `docs/DECISIONS_LOG.md` (Ghi log siêu chi tiết để lưu trữ)
