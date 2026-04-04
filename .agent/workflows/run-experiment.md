---
description: Quy trình chạy thí nghiệm (experiment) end-to-end cho dự án Time Series Forecasting
---

# Workflow: Run Experiment

Quy trình chuẩn để chạy một thí nghiệm mới, đảm bảo tất cả kết quả được lưu trữ và truy vết.

## Pre-flight Checklist

1. Đọc `.agent/memory/CONTEXT.md` để nắm trạng thái hiện tại
2. Đọc `.agent/memory/RUNS_LOG.md` để biết kết quả các lần chạy trước
3. Đọc `.agent/memory/LESSONS_LEARNED.md` để tránh lỗi đã biết
4. Xác nhận config trong `configs/` đã đúng tham số mong muốn

## Execution Steps

// turbo
5. Sync dependencies: `uv sync`

// turbo
6. Lint check trước khi chạy: `uv run ruff check src/ --fix && uv run ruff format src/`

7. Chạy experiment:
```bash
uv run python scripts/run_experiment.py --model <model_name> --config configs/model_configs/<model>.yaml
```

> **Tham khảo**: `.agent/guides/analytics-experiment-design.md` cho Hypothesis Lock và Experiment Checklist

8. Kiểm tra kết quả trong `research/runs/YYYYMMDD_HHMMSS/`:
   - `config.json` — tham số đã dùng
   - `metrics.csv` — MAE, RMSE, R², MAPE, MASE
   - `predictions.csv` — y_true vs y_pred
   - `plots/` — biểu đồ kết quả

// turbo
9. Chạy tests để đảm bảo không có regression: `uv run pytest tests/ -v`

## Post-flight Checklist

10. Cập nhật `.agent/memory/RUNS_LOG.md` với kết quả mới
11. Cập nhật `.agent/memory/CONTEXT.md` nếu có model tốt hơn
12. Nếu có lỗi → ghi vào `.agent/memory/LESSONS_LEARNED.md`
13. Cập nhật `.agent/memory/TODO.md` với bước tiếp theo
