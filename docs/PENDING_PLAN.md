# 📋 PENDING PLAN — Dashboard V9 Full Audit & Sync
> **Ngày tạo:** 2026-05-03
> **Hoàn thành:** 2026-05-04
> **Trạng thái:** ✅ HOÀN TẤT (Tất cả 6 hạng mục đã được xử lý trong Phase 9)
> **Mục tiêu:** Đồng bộ toàn bộ Dashboard UI với kết quả nghiên cứu V9 thực tế.

---

## 🔴 Vấn đề tổng quát

Dashboard đã trải qua 9 phiên bản pipeline (v1→v9), nhưng nhiều nơi trên UI vẫn còn **dữ liệu cứng (hardcoded)** hoặc **cache cũ** từ các phiên bản trước. Cần audit toàn diện và sửa dứt điểm.

### Ground Truth (V9 — nguồn duy nhất)
| Horizon | Best Model | MASE | MAE | Ghi chú |
|---------|-----------|------|-----|---------|
| **1h** | GRU_v9_15m | 0.667 | 2.94 | Phá vỡ Autocorrelation Trap! |
| **6h** | Ensemble_Weighted_v9_30m | 0.382 | 3.49 | ⭐ Best overall |
| **24h** | Ensemble_Weighted_v9_30m | 0.469 | 3.42 | Long-range champion |

### Pipeline Data (V9)
- **Raw:** 209,397 records (~2 phút/mẫu, 3.1 năm, Sa Đéc Đồng Tháp)
- **Resample:** 3 resolutions — 15m (~110K), 30m (~55K), 1h (~27K)
- **Features:** 119 columns (anti-leakage, shift(1))
- **Split:** 80/10/10 temporal, test = real data only
- **Models:** 41 models across Persistence, ARIMA, ML (LightGBM, RF, GB, ElasticNet, Stacking), DL (GRU, LSTM, TFT Fair+Expert), Ensemble

---

## ✅ Đã làm (trong session 2026-05-03)

1. **`pages.py`**: Chuyển ranking metric mặc định từ MAE → MASE (index=1) tại trang "Actual vs Predicted".
2. **`app.py`**: Sửa pipeline architecture text — row counts từ `7,742` → `15m: ~110K, 30m: ~55K, 1h: ~27K`.
3. **`src/frontend/citations.py`**: Cập nhật row counts tương tự.
4. **`src/explainability_hub.py`**: Sửa "After Impute" stat từ `7,742` → `~110K (15m)`.

---

## ✅ Đã hoàn tất (Phase 9 — 2026-05-04)

### ✅ 1. Database Info Cards (PostgreSQL)
**File liên quan:** `src/api/models.py` (InfoCard model), `scripts/seed_info_cards.py`
**Cách tiếp cận:** Viết script `scripts/fix_db_content.py` để update trực tiếp trong DB.

| Card Key | Vấn đề | Cần sửa thành |
|----------|--------|---------------|
| `overview_highlights` | Ghi "TFT_1h phá vỡ Autocorrelation" | "GRU_v9_15m (MASE=0.667) phá vỡ Autocorrelation Trap" |
| `eda_findings` hoặc `eda_lessons` | Chưa có "The Why" cho v9 pipeline | Thêm giải thích tại sao chọn multi-resolution (15m bắt spikes nhưng nhiễu, 1h mịn nhưng bẫy autocorr, 30m tối ưu) |
| Persistence dominance | Nhiều nơi vẫn ghi Persistence thắng ở 1h | Cập nhật: GRU_v9_15m đã phá vỡ giới hạn này |

### ✅ 2. Quét sạch hardcode `7,742` / `7742`
**Đã sửa:** `app.py`, `citations.py`, `explainability_hub.py`
**Cần kiểm tra thêm:** Grep toàn bộ project, đặc biệt `dashboard_content.json`, DB seeds, và thesis draft.

### ✅ 3. Vẽ lại Sankey Diagram
**File:** `src/explainability_hub.py` (dòng ~285-420)
**Vấn đề:** Hiện tại gộp 15m+30m+1h lại trước Feature Eng. → Sai logic.
**Cần:** Tách thành 3 nhánh song song sau bước Clean:
```
IoT (209K) → Clean → ┬─ 15m (~110K) → FE → Split → Models ─┐
                      ├─ 30m (~55K)  → FE → Split → Models ─┤→ Evaluation → Best
                      └─ 1h  (~27K)  → FE → Split → Models ─┘
```

### ✅ 4. SHAP Feature Importance (24h)
**File:** `src/explainability_hub.py` + `research/cache/shap_summary_*.json`
**Vấn đề:** lag1h đang là top feature của 24h → rất khả nghi (nên là lag24h hoặc rolling_mean_24h).
**Cần:**
- Kiểm tra cache file `shap_summary_24h.json` xem có bị ghi đè bởi 1h không.
- Nếu cache sai → regenerate bằng script `scripts/v9_cache_shap.py`.
- Kiểm tra UI loading logic (đúng horizon key chưa).

### ✅ 5. Màu sắc Khoảng tin cậy (Confidence Intervals)
**File:** `pages.py` hoặc `src/viz/charts.py`
**Vấn đề:** Chỉ số (Coverage, Width) màu trắng → không thấy trên nền trắng (Light mode).
**Cần:** Đổi sang `#71717A` (Kẽm 500 — đã là chuẩn VTF của project) hoặc dùng CSS `var(--text-color)`.

### ✅ 6. AVP Cache cho 6h và 24h
**File:** `research/cache/avp_6h.json`, `avp_24h.json`
**Script:** `scripts/update_avp_cache_v9.py`
**Cần:** Verify cache chứa đúng v9 models. Nếu thiếu → chạy lại script.

---

## 🔧 Lệnh thực thi (khi sẵn sàng)

```bash
# 1. Grep kiểm tra hardcode còn sót
grep -rn "7,742\|7742" --include="*.py" --include="*.json" --include="*.md" .

# 2. Chạy fix DB content
uv run python scripts/fix_db_content.py

# 3. Regenerate SHAP cache (nếu cần)
uv run python scripts/v9_cache_shap.py

# 4. Regenerate AVP cache
uv run python scripts/update_avp_cache_v9.py

# 5. Rebuild Docker
docker-compose up -d --build

# 6. Verify
uv run streamlit run app.py
```

---

## 📁 Files chính cần chỉnh sửa

| File | Mục đích |
|------|----------|
| `app.py` | Trang Tổng Quan (Overview) — KPI cards, Pipeline Architecture |
| `pages.py` | Actual vs Predicted, Confidence Intervals |
| `src/explainability_hub.py` | Sankey diagram, SHAP display, Pipeline stats |
| `src/pipeline_walkthrough.py` | 7-step pipeline walkthrough |
| `src/reporting/engine.py` | ReportingEngine — KPI, insights generation |
| `src/reporting/content.py` | ContentManager — loads from DB/JSON |
| `src/info_cards.py` | Info card registry, version badge |
| `src/snapshot_adapter.py` | Snapshot normalization, MASE extraction |
| `src/frontend/citations.py` | Academic citation helpers |
| `research/experiments/dashboard_content.json` | Fallback content (khi DB không available) |
| `research/experiments/dashboard_runs/v9_multi_resolution.json` | V9 snapshot — **SOURCE OF TRUTH** |
| `research/experiments/standardized_metrics.json` | Pre-computed metrics |
| `research/cache/avp_*.json` | AVP prediction cache |
| `research/cache/shap_summary_*.json` | SHAP feature importance cache |
