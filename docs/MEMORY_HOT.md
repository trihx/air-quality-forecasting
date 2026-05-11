# PROJECT CONTEXT

## Trạng thái (2026-05-04)
- **Status**: ✅ Phase 9 Dashboard Full Audit & Sync — HOÀN TẤT.
- **Current**: 🔶 Phase 10: Thesis V9 Sync & Final Polish (ĐANG THỰC HIỆN).
- ✅ **Ablation Study (Phase 10)**: Hoàn tất thí nghiệm v10. Phát hiện "False Sense of Accuracy" do IQR cắt mất 66 đỉnh ô nhiễm. Xác nhận Domain Bounds (v9) là chiến lược tiền xử lý BẮT BUỘC cho PM2.5.
- ✅ **EDA Enhancement (Phase 1-3)**: Bổ sung MI Heatmap, Conditional Violin, Boxplot (Weekend), và Imputation Analysis. Sửa lỗi hiển thị Log-Scale Granger Causality.
- **Next Step**: Biên soạn thesis cuối cùng → nộp.
- **Tiến độ v9_multi_resolution**: 
  - ✅ Phase 5: Multi-Resolution (15m, 30m, 1h), Segment-aware training (Fair vs Expert DL).
  - ✅ Phase 6: Deep Learning Retrain (LSTM, GRU, TFT). Phá vỡ bẫy Persistence ở 1h.
  - ✅ Phase 7 & 8: Dynamic UI, PostgreSQL Seed. Zero hardcoded UI policy được thực thi triệt để.
  - ✅ Phase 9: Dashboard Full Audit — JSON Fallback, Zinc 500 palette, Sankey 3-branch, SHAP/AVP cache verified.
  - 🔶 Phase 10: Thesis row counts V9, cleanup, docs sync, Ablation Study.
- **Best models (MASE unified)**:
  - 1h: GRU_v9_15m (0.667) — phá vỡ autocorrelation trap!
  - 6h: Ensemble_Weighted_v9_30m (0.382) ⭐
  - 24h: Ensemble_Weighted_v9_30m (0.469) ⭐
- **Dữ liệu**: 209K records → Clean(Domain Bounds cho PM2.5, IQR cho biến phụ) → Resample(15m/30m/1h). Dùng segmenting loại bỏ false continuity.
- **Tài liệu**: `THESIS_DRAFT_CTU_1799.md`, `KNOWLEDGE_BASE.md`. Plan chi tiết: `docs/PENDING_PLAN.md`.

## 🚨 DATA INTEGRITY & UI RULES (TUYỆT ĐỐI)
1. **KHÔNG halucinate số liệu** — mọi metric PHẢI lấy từ JSON/log output.
2. **FIT trên TRAIN ONLY** — Áp dụng cho STL, PCA, BoxCox, Scaler. KHÔNG fit full data.
3. **Leakage audit** — MASE < 0.1 hoặc R² > 0.99 = red flag.
4. **Test set = REAL data only** — `is_imputed==0` filter BẮT BUỘC.
5. **Shift(1) cho feature dùng target** — diff, pct_change, ratio, rolling.
6. **Zero-Hardcode UI Policy** — Toàn bộ text dài/insights phải lưu vào `dashboard_content.json` và load qua `ContentManager`. UI file CHỈ làm Layout.
7. **Visual Theme Framework (VTF)** — Plotly chart phải dùng chung cấu hình từ `src/viz/theme.py`. Màu text dùng Kẽm 500 (`#71717A`) tương thích Light/Dark. Tuyệt đối không để chart bị lỗi overlay (legend đè data).
8. **MASE là metric chính** — Mọi ranking, KPI, insights PHẢI dùng MASE (không phải MAE) để xếp hạng. Snapshot adapter đã chuẩn hóa.
9. **encoding="utf-8" BẮT BUỘC** — Mọi `open()` text mode trên Windows PHẢI có `encoding="utf-8"` để tránh `UnicodeDecodeError`.
10. **Trích dẫn Academic (IEEE_REFS)** — LUÔN bổ sung field `quote` (trích NGHIÊM NGẶT từ abstract/sách gốc, CẤM hallucinate) và `location` (Abstract, Chapter, Page) khi thêm tài liệu mới vào `citations.py`.

## Pipeline & Architecture
**Data Pipeline:** `Raw (209K) → Clean ([0,500], S-ESD) → Resample (15m/30m/1h) → Impute (Spline+KNN) → Features (119) → Split 80/10/10 (temporal) → Target: shift(-h) → Model → Eval (MAE+MASE+Brier/F1)`
**System Architecture:** `Streamlit (Thin UI) ↔ FastAPI (Backend 17 routes) ↔ PostgreSQL (DB 5 tables)`
**Data Fallback:** `API (PostgreSQL) → JSON export (db_export/) → Default string`

## Snapshot Versioning (dashboard_runs/)
- v1_baseline → v8_conformal: Legacy (1h-only pipeline)
- v9_multi_resolution: Multi-Res (15m/30m/1h), Fair/Expert DL, Ensemble Weighted, Dashboard Dynamic (Final)

## Execution
`uv run streamlit run app.py` | `uv run pytest tests/ -v` | `lsof -ti :8501 | xargs kill -9`

11. **Auto-Update Thesis Knowledge** -> Bất cứ khi nào Agent giải đáp thắc mắc, phân tích insight biểu đồ, hay giải thích quy trình làm việc cho User, Agent BẮT BUỘC phải tự động cập nhật và ghi chép lại những kiến thức này vào file docs/THESIS_EXPLANATIONS.md để lưu trữ tri thức phục vụ luận văn mà không cần User phải nhắc nhở.
