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

## [2026-04-11] Fourier Features = High-Value Low-Cost
- **Bài học**: `fourier_daily_sin_2` xếp #2 feature importance trong LightGBM (sau pm25_lag_1h). 12 Fourier features chỉ dùng timestamp → zero leakage risk, zero compute cost.
- **Rule**: Fourier nên là DEFAULT feature cho mọi time series project có seasonality.

## [2026-04-12] Stacking Ensemble ≠ Always Better
- **Bài học**: Stacking (ElasticNet+RF+GB→Ridge) tệ hơn RF đơn lẻ ở cả 6h (0.735 vs 0.706) và 24h (0.857 vs 0.798). Meta learner (Ridge) không khai thác được diversity.
- **Rule**: Weighted Ensemble (grid search) > Stacking khi base models tương tự nhau. Luôn benchmark individual models trước.

## [2026-04-12] RC vs TSF Metrics — Evaluation Policy Matters
- **Bài học**: RC MAE thấp hơn (1.845 vs 2.460 ở 1h) vì test set bao gồm imputed data (smooth → dễ predict). TSF test-on-real-only → metrics phản ánh thực tế hơn.
- **Rule**: LUÔN ghi rõ evaluation policy (test-on-real, bao-gồm-imputed). Không so sánh trực tiếp MAE giữa 2 policy khác nhau.

## [2026-04-12] Snapshot Versioning cho Experiment Tracking
- **Bài học**: Mỗi lần bổ sung model/feature → lưu snapshot riêng (v1→v2→v3) với trường `changes: {what, why, result}`. Dashboard tự đọc và hiển thị diff.
- **Rule**: KHÔNG ghi đè snapshot cũ. `parent_version` field để trace lineage.

## [2026-04-12] DL + High-Dim Features = Curse of Dimensionality ở Short Horizon
- **Bài học**: GRU/LSTM v1 (5 features) → MASE=1.173 ở 1h. V2 (117 features) → MASE=1.531-1.888. Features nhiều hơn ≠ tốt hơn cho DL ở 1h do autocorrelation ≈1.0.
- **Rule**: DL với >100 features cần PCA/feature selection ở 1h horizon. 6h/24h thì OK.

## [2026-04-12] Log Transform Effect Phụ Thuộc Kiến Trúc
- **Bài học**: log1p giúp GRU rõ rệt (6h: raw=0.783 vs log=0.692 ↓11.6%). Nhưng LSTM ưa raw ở 6h (0.719 vs 0.753). Ở 24h: cả 2 đều hưởng lợi nhẹ từ log.
- **Rule**: KHÔNG áp dụng log toàn bộ. Test raw+log cho TỪNG model family, TỪNG horizon.

## [2026-04-12] CV Features (std/mean) Cần Safeguard
- **Bài học**: Coefficient of Variation = std/mean. Khi mean≈0 (PM2.5 rất thấp), CV explodes → inf. Safeguard: clamp mean >= 1.0 + clip CV <= 5.0.
- **Rule**: NaN/inf audit cho mọi engineered feature. Test edge case (all zeros, near-zero mean).

## [2026-04-12] Feature Engineering = Con Dao Hai Lưỡi cho DL
- **Bài học**: 117 features giúp GRU 6h (↓14.8%) nhưng HẠI 1h (+30.5%). PCA (37), TopN (10/20/40) đều KHÔNG cứu được 1h. V1 (5 raw) vẫn best ở 1h. Autocorrelation ≈1.0 → chỉ cần lag_1h.
- **Rule**: Ở horizon ≤2h, ƯU TIÊN simple model (ít features). Feature engineering chỉ giúp ở h≥6.

## [2026-04-12] TFT Cần Model Capacity Phù Hợp Với Feature Dim
- **Bài học**: TFT v1 (5 temporal, hidden=32, 25K params) → MASE=1.029 ở 1h. TFT v2 (113 temporal, hidden=32, 28K params) → MASE=1.976 (+92%). Features tăng 22x nhưng capacity gần như không đổi.
- **Rule**: Khi tăng input dim, PHẢI tăng hidden_dim tương ứng hoặc giảm feature dim trước bằng PCA/selection.

## [2026-04-12] OMP Threading Crash trên Apple Silicon + LightGBM
- **Bài học**: `n_jobs=-1` gây `OMP: Error #179: Function pthread_mutex_init failed` khi LightGBM chạy cùng PyTorch trên M1 Pro.
- **Rule**: LUÔN dùng `n_jobs=1` + `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` khi mix LightGBM + PyTorch.
