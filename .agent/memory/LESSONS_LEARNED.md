# LESSONS LEARNED

> Ghi lại **MỌI** lỗi gặp phải và cách khắc phục. Đọc file này TRƯỚC KHI implement để tránh lặp lại lỗi cũ.

---

## Archived (2026-03-29) — Tóm tắt 1 dòng
<!-- Chi tiết: xem git log hoặc conversation logs -->
- **Data Leakage**: `diff/pct_change/ratio/aqi_cat` chứa y[t] → fix: `shift(1).diff()`, dùng `pm25_lag_1h`. R²>0.99 = red flag.
- **CSV Loading chậm**: Dùng `usecols`, cân nhắc parquet. Luôn `flush=True`.
- **Stationarity**: PM2.5 = trend-stationary (ADF ✅, KPSS ❌). Luôn chạy CẢ hai.
- **Unit test phải update**: Thay đổi logic → update test assertions. `shift(1).diff()` NaN tại iloc[1].
- **Kill processes**: `pkill -f "python.*scripts/"` trước pipeline run.

---

## Archived (2026-04-04 → 2026-04-05) — Tóm tắt 1 dòng
<!-- Chi tiết: xem DECISIONS.md hoặc conversation logs -->
- **Leakage Impact**: Ridge MAE 0.004→2.824. MASE<0.1 = 🚨 leakage.
- **Result Validation**: MAE PM2.5 phải 1.5-15µg/m³. Shuffle test: MASE>>1.0.
- **PyTorch MPS + LightGBM**: Lazy import torch SAU LightGBM. Không import top-level.
- **Multi-Horizon Target Bug**: `target=shift(-h)`, `persist=df[TARGET_COL]`. KHÔNG dùng lag features cho Persist.
- **MPS GPU Training**: `torch.device("mps")` ~5x CPU. batch=256. GRU cần pm25 trong input features.
- **Thesis Metric Consistency**: Bảng thesis PHẢI cross-ref JSON files. Ensemble attribution rõ nguồn.
- **TFT Sizing**: Dataset <10K → hidden≤32, heads≤4, patience 15. Attention best short-horizon.
- **Plotly Colors**: CHART_COLORS phải ≥ N models. Dùng `i % len()` an toàn.
- **RAG Multilingual**: Dự án Vietnamese → `paraphrase-multilingual-MiniLM-L12-v2`. KHÔNG dùng English-only.
- **Path.parent**: `src/chatbot/file.py` cần `.parent.parent.parent` → root. Luôn test path trước.

## [2026-04-05] Streamlit Port Management
- **Lỗi**: `Ctrl+C` Streamlit đôi khi không kill process → port bị chiếm → khởi động lại fail.
- **Fix**: `lsof -ti :8501 | xargs kill -9` rồi restart.
- **Rule**: Khi Streamlit refuse start vì port → LUÔN kill process cũ trước.

## [2026-04-11] Streamlit File Watcher + transformers = Performance Kill
- **Lỗi**: `local_sources_watcher` scan 200+ `transformers.models.*` modules → hàng trăm `ModuleNotFoundError: torchvision` tracebacks → startup chậm ~50%.
- **Fix**: `.streamlit/config.toml` → `fileWatcherType = "none"`. Lazy import chatbot modules trong `main()`.
- **Rule**: Khi dùng `sentence-transformers`/`transformers` trong Streamlit → LUÔN tắt file watcher hoặc dùng `"poll"`.
