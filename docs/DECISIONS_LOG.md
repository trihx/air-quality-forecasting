# TECHNICAL DECISIONS LOG

> Ghi lại các quyết định kỹ thuật quan trọng để team và AI assistant tham chiếu.

---

## Archived (2026-03-28 → 2026-03-29) — Tóm tắt
<!-- Chi tiết: xem git log hoặc conversation 30cab957 -->
- **Cấu trúc thư mục**: Custom based on Cookiecutter DS + `memory/` cho context engineering
- **SKILL.md**: Hybrid approach (inline rules + separate guides). 777+ dòng → tách 7 guide files
- **Logging**: loguru (research DX > enterprise). Migrate structlog khi production.
- **Visualization**: Data Storytelling approach. 12 chart types, WHO reference line bắt buộc.
- **Feature Engineering**: 95 features, anti-leakage (shift ≥ 1). ⚠️ diff/domain CÓ leakage → ĐÃ FIX.
- **Leakage Audit**: 4 nguồn fix. MASE>1 cho tất cả ML. ✅ HOÀN THÀNH 2026-04-04.
- **Stationarity**: PM2.5 = trend-stationary. ML dùng raw, ARIMA cần d=1.

---

## [2026-04-04] Memory Optimization Strategy
- HOT (mỗi phiên): CONTEXT.md + TODO.md ≤ 100 dòng
- WARM (khi cần): DECISIONS.md, LESSONS_LEARNED.md, RUNS_LOG.md
- COLD (1 lần): SKILL.md, guides/*
- **Compact trigger**: LESSONS_LEARNED > 80 dòng → archive entries > 7 ngày

## [2026-04-04] EDA Strategy — Hybrid Imputation
- So sánh 4 strategies: Segment(1.113) < ML(1.084) < Hybrid(**1.066**) < ExtInterp(1.321)
- ⭐ **Hybrid** (Spline≤6h + KNN 6-24h) = best. Cubic spline alone gây noise.
- **Rule**: Test set = REAL data ONLY

## [2026-04-04] IEEE Citation Style
- Dùng chuẩn IEEE cho trích dẫn (theo QĐ 1799 CTU)

## [2026-04-04 21:52] Multi-Horizon — ML thắng ở horizon dài
- **v2 (FIXED)**: 1h=MASE **1.492** (❌), 6h=MASE **0.745** (-25.5% ✅), 24h=MASE **0.842** (-15.8% ✅)
- Feature shift: 1h→lag, 6h→temporal patterns, 24h→multivariate
- Giả thuyết ĐÚNG: ML tạo giá trị khi autocorrelation giảm

## [2026-04-04 21:52] Pipeline Audit v2 — 2 Critical Bugs Fixed
- **Bug #1**: h=1 target=y[t] → fix: `shift(-h)` cho MỌI horizon
- **Bug #2**: Persistence=lag_Xh=y[t-X] → fix: dùng `df[TARGET_COL]` trực tiếp
- **Impact**: LightGBM h=1 MASE: 1.012 → 1.492. Persistence MAE nhất quán across scripts.
- **Rule**: LUÔN `shift(-h)` cho target + `df[TARGET_COL]` cho Persistence. KHÔNG dùng lag features.

## [2026-04-04] statsmodels + gappy DatetimeIndex
- Drop NaN → mất freq → dùng `.values` (numpy). `forecast()` trả numpy → `[-1]` not `iloc[-1]`.

## [2026-04-05] Hyperparameter Tuning Strategy
- **ML**: Optuna Bayesian (TPE), 50-100 trials, TimeSeriesSplit(5), minimize MAE. Mỗi horizon tune riêng.
- **DL**: Manual config + Early Stopping (patience=10). Optuna DL quá tốn kém với 7742-row dataset.
- **Lưu trữ**: `research/best_models_configs.json` = single source of truth → dùng cho bảng biểu luận văn.
- **Fine-tune**: ML → sửa search space trong `multi_horizon_eval.py`. DL → sửa constants trong `dl_multi_horizon.py`.

## [2026-04-05] SHAP Explainability
- **LightGBM**: TreeExplainer (exact, <0.3s/horizon). Top feature: pm25_lag_1h (1h), pm25_roll_24h_mean (6h).
- **GRU**: Permutation importance (5 features × 5 rounds). MPS GPU + mini-batch (256). 50 epochs + ReduceLROnPlateau.
- **Insight**: SHAP rank ≠ built-in rank → SHAP chính xác hơn vì tính interaction effects.
- **Plots**: `research/figures/shap/` — bar, beeswarm, dependence (LightGBM) + permutation bar (GRU).

## [2026-04-05] TFT — Simplified Temporal Fusion Transformer
- **Quyết định**: Implement Simplified TFT trong pure PyTorch (không dùng pytorch-forecasting) để control architecture.
- **Architecture**: GRN + GLU + Multi-head Attention (4 heads) + Static Encoder. 25,089 params.
- **Sizing**: hidden_dim=32 (nhỏ hơn GRU 64) vì dataset 7.5K rows không đủ cho TFT lớn.
- **Kết quả**: Best ML/DL tại h=1 (MASE=1.029), competitive tại h=6/24 (0.822/0.812).
- **Kết luận**: Attention giúp khai thác short-term patterns tốt hơn GRU, nhưng cần >50K rows để full potential.

## [2026-04-05] Dashboard — Scientific Observatory Design
- **Theme**: Dark (#0E1117) + teal accent (#00D4AA) + glassmorphism KPI cards.
- **Pages**: 6 (Overview, Multi-Horizon, SHAP, PI, EDA, Hyperparameters).
- **Charts**: Plotly (bar, scatter, line) với custom theme. TFT integrated vào tất cả charts.

## [2026-04-05] Prediction Intervals Strategy
- **3 methods**: Conformal (agnostic), Quantile Regression (LightGBM native), MC Dropout (GRU).
- **Best**: Quantile coverage 86.2% (1h). MC Dropout quá narrow (36.8%) do dropout=0.2 nhỏ.
- **Kết luận**: Quantile = recommended. MC Dropout cần calibration hoặc tăng dropout rate.

## [2026-04-05] AI Chatbot — RAG + Local LLM
- **Architecture**: RAG (ChromaDB + multilingual embeddings) → Context injection → LM Studio (local LLM)
- **Embedding**: `paraphrase-multilingual-MiniLM-L12-v2` — hỗ trợ Vietnamese, 50+ languages
- **LLM**: Gemma 4 E4B (4.5GB Q4) hoặc Qwen3-8B, chạy qua LM Studio port 8888
- **System prompt**: Ưu tiên phương pháp luận → quy trình → quyết định → kết quả (chuẩn bị phản biện)
- **Knowledge sources**: 11 markdown docs + 9 experiment dirs + standardized_metrics.json → ~243 chunks
- **Gotcha**: `all-MiniLM-L6-v2` = English-only → câu hỏi Việt match kém. PHẢI dùng multilingual.

## [2026-04-11] RC Integration Strategy
- **Quyết định**: Tích hợp từ Research Code (RC) → TSF: Fourier features, interaction features, log1p, rolling range, linear models, sklearn tree-based + ensemble.
- **Nguyên tắc**: Không copy code — chỉ port methodology. Mọi feature phải tuân thủ anti-leakage (dùng `pm25_lag_1h`).
- **Log transform**: Áp dụng `log1p` target cho toàn bộ pipeline (linear + LightGBM). DL chưa retrain.
- **CV feature (std/mean)**: Bỏ khỏi default do risk NaN khi std=0. Thử sau với safeguard.
- **Kết quả v2**: LightGBM MAE ↓14.2% (1h). Fourier `sin_day_2` = #2 feature importance.

## [2026-04-12] Sklearn Ensemble — Performance vs v1
- **RandomForest**: 300 trees, max_depth=12. MASE 6h=0.706, 24h=0.798. Ngang ElasticNet.
- **GradientBoosting**: 300 trees, lr=0.05. Thua RF (0.721 vs 0.706 ở 6h).
- **Stacking**: ElasticNet+RF+GB→Ridge meta, CV=5. **Tệ nhất** trong group (6h=0.735, 24h=0.857).
- **Ensemble_Weighted**: Grid-search weights (step=0.1). Best sklearn (6h=0.705, 24h=0.797). RF=80%+GB=20%.
- **Kết luận**: RF là backbone chính. Stacking không mang lợi ích với base models tương đồng.

## [2026-04-12] Snapshot Versioning Protocol
- **Format**: `dashboard_runs/{version}_{date}.json`
- **Fields bắt buộc**: `version`, `timestamp`, `parent_version`, `changes: {what, why, result, conclusion}`, `feature_set`, `data.results`
- **Rule**: KHÔNG GHI ĐÈ snapshot cũ. Mỗi run tạo file mới.
- **Dashboard**: Tab "So Sánh Phiên Bản" tự đọc snapshots, hiển thị feature diff + MAE/MASE comparison.

## Archived Lessons (pre-2026-04-05)
- **Data Leakage**: `diff/pct_change/ratio/aqi_cat` chứa y[t] → fix: `shift(1).diff()`, dùng `pm25_lag_1h`. R²>0.99 = red flag.
- **CSV Loading chậm**: Dùng `usecols`, cân nhắc parquet. Luôn `flush=True`.
- **Stationarity**: PM2.5 = trend-stationary (ADF ✅, KPSS ❌). Luôn chạy CẢ hai.
- **Unit test phải update**: Thay đổi logic → update test assertions. `shift(1).diff()` NaN tại iloc[1].
- **Kill processes**: `pkill -f "python.*scripts/"` trước pipeline run.
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

## Archived Lessons (2026-04-05 → 2026-04-29)
- **Streamlit Port**: Lỗi port 8501 bị chiếm → `lsof -ti :8501 | xargs kill -9` rồi restart.
- **File Watcher**: `local_sources_watcher` scan `transformers` chậm → tắt watcher trong `config.toml` hoặc dùng "poll".
- **Stacking**: Ensemble kém khi models tương đồng → dùng Weighted Ensemble.
- **Eval Policy**: RC MAE thấp hơn TSF vì có data imputed → LUÔN test on real only.
- **DL Features**: >100 features hại DL ở 1h → dùng PCA hoặc lag_1h đơn giản.
- **Log Transform**: log1p giúp GRU ở 6h nhưng hại LSTM → Test riêng cho từng model/horizon.
- **CV Feature**: std/mean explodes khi mean≈0 → Cần clamp/clip safeguard.
- **OMP Crash**: PyTorch + LightGBM gây crash trên M1 → `n_jobs=1` + `KMP_DUPLICATE_LIB_OK=TRUE`.
- **Look-Ahead Bias**: STL fit full data gây leak → Fit TRAIN ONLY, extrat 24h pattern cho test.
- **Fourier**: Đã có fourier thì không cần explicit deseasonalizing.
- **Outlier**: IQR*3 chặt PM2.5 ở 54 (dưới mức unhealthy) → Dùng domain bounds [0, 500].
- **DL Persistence Bug**: Offset trật nhịp gây MAE=0 → Thống nhất dùng Unified Persistence.
- **Snapshot Versioning**: Xóa v7 thừa, đổi v8 thành v7, chỉ lưu khi có kết quả mới. Không ghi đè snapshot cũ.
- **DevTools Warnings**: Framework issues từ Streamlit (labels, form ID) → bỏ qua, không fix.
- **Theme Inversion**: Không hardcode HEX color trên Streamlit UI → Dùng CSS vars (text-color, background-color).
- **Plotly Text Color (2026-04-29)**: Plotly SVG text KHÔNG nhận CSS Variable, `st.get_option("theme.base")` trả `None` ở Auto mode. Khắc phục bằng màu Kẽm 500 (`#71717A`) - màu Middle Gray tương phản tốt trên cả nền sáng/tối thay vì dò Theme.
- **VTF Framework**: Gom toàn bộ rcParams, bbox, Plotly_template vào `src/viz/theme.py`.

## [2026-04-30] Content-Driven Reporting Framework
- **Quyết định**: Tách toàn bộ textual content (Insights, Lessons, Literature Comparison) khỏi source code UI (`app.py`, `info_cards.py`).
- **Architecture**: `ContentManager` (`src/reporting/content.py`) đọc dữ liệu từ `research/experiments/dashboard_content.json`.
- **Lý do**: Zero-Hardcode policy. Khi version model thay đổi, nội dung insights cũng cần thay đổi theo version (version-aware content). Đảm bảo tính minh bạch, linh hoạt trong khoa học.
- **Rule**: Code UI chỉ để render Layout và Visualization. Mọi Text/Narrative phải lấy từ `ContentManager`.

## [2026-05-03] Phase 9 — Dashboard Full Audit & Sync
- **Vấn đề**: Sau 9 phiên bản pipeline, Dashboard UI vẫn chứa dữ liệu cũ (hardcoded) từ v7/v8 (row counts, model names, SHAP cache). Sankey diagram logic sai cho v9 multi-resolution.
- **Quyết định**: Tạo audit plan chi tiết (`docs/PENDING_PLAN.md`) với 6 hạng mục cần fix.
- **Ground Truth**: v9_multi_resolution.json là nguồn dữ liệu duy nhất. MASE là metric xếp hạng chính.
- **Đã fix**: Row counts (app.py, citations.py, explainability_hub.py), ranking default MAE→MASE (pages.py).
- **Chưa fix**: DB info cards, Sankey redesign, SHAP 24h cache, CI chart colors, AVP 6h/24h cache verification.

## [2026-05-03] Cross-Machine Portability Protocol
- **Vấn đề**: Antigravity brain artifacts (`~/.gemini/antigravity/brain/<conversation-id>/`) KHÔNG đi theo source code khi copy sang máy khác.
- **Giải pháp**: Lưu toàn bộ plan và context vào project directory:
  - `docs/PENDING_PLAN.md` — Plan chi tiết, checklist, ground truth
  - `docs/MEMORY_HOT.md` — Trạng thái hiện tại + pointers
  - `docs/LESSONS_LEARNED.md` — Bug patterns, anti-patterns
  - `docs/DECISIONS_LOG.md` — Context kỹ thuật chi tiết
- **Rule**: Khi có plan/task quan trọng → LUÔN lưu vào `docs/` thay vì chỉ ở brain artifacts.

## [2026-05-05] Phase 1-3 EDA Deep Insights
- **Conditional Distribution (Violin Plot)**: Quy?t d?nh d�ng Violin thay v� Boxplot khi ph�n t�ch nhi?t d? (<26, 26-30, >30) d? th?y r� hi?n tu?ng ph�n ph?i k�o d�i (long-tail pollution) ? nhi?t d? th?p � bi?u hi?n r� r�ng c?a hi?n tu?ng ngh?ch nhi?t.
- **Mutual Information vs Correlation**: B? sung MI Heatmap v� h? s? Pearson truy?n th?ng kh�ng b?t du?c tuong quan phi tuy?n. MI x�c nh?n Nhi?t d? v� �i?m suong d?n d?t PM2.5, ph� h?p v?i ki?n th?c domain knowledge (ngh?ch nhi?t).
- **Weekend vs Weekday**: D�ng Boxplot d? ph�n t�ch ph�t th?i theo ng�y trong tu?n. Kh�ng c� s? gi?m s�t d�ng k? v�o T7/CN, ch?ng t? PM2.5 d?n t? ngu?n h?n h?p (giao th�ng + sinh ho?t + ph�t t�n t? noi kh�c) thay v� ch? giao th�ng c�ng s?.
- **Imputation Validation**: So s�nh m?t d? PM2.5 sau di?n khuy?t v?i g?c. Phuong ph�p Hybrid (Spline+KNN) b?o to�n g?n nhu ho�n h?o ph�n ph?i g?c, ch?ng minh l?a ch?n phuong ph�p di?n khuy?t l� d�ng d?n.
