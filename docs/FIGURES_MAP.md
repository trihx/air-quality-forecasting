# Bảng Ánh Xạ Hình ↔ Thesis Section (FIGURES_MAP)

> **Mục đích:** Giúp anh Trí chèn hình vào Word đúng vị trí. Mỗi dòng = 1 hình cần chèn.
> **Tổng:** 73 hình (EDA: 26, Figures: 29, Diagnostics: 18)

---

## Chương 3: Phương Pháp Nghiên Cứu

| Hình | Mô tả | File Path | Section |
|------|-------|-----------|---------|
| 3.1 | Kiểm định tính dừng — Raw PM2.5 | `research/diagnostics/stationarity/stationarity_raw_pm2.5.png` | §3.6 |
| 3.2 | Kiểm định tính dừng — 1st Diff (d=1) | `research/diagnostics/stationarity/stationarity_1st_diff_(d=1).png` | §3.6 |
| 3.3 | Kiểm định tính dừng — Seasonal Diff (d=24h) | `research/diagnostics/stationarity/stationarity_seasonal_diff_(d=24h).png` | §3.6 |
| 3.4 | Kiểm định tính dừng — Log PM2.5 | `research/diagnostics/stationarity/stationarity_log_pm2.5.png` | §3.6 |
| 3.5 | Kiểm định tính dừng — Log 1st Diff | `research/diagnostics/stationarity/stationarity_log_1st_diff.png` | §3.6 |

## Chương 4: Kết Quả và Thảo Luận

### §4.1 Data Storytelling (EDA)

| Hình | Mô tả | File Path | Section |
|------|-------|-----------|---------|
| 4.1 | Bẫy tự tương quan — Autocorrelation Memory | `research/eda/visualizations/01a_autocorrelation_memory.png` | §4.1.1 |
| 4.2 | Phân tán theo horizon | `research/eda/visualizations/01b_horizon_scatter_dispersion.png` | §4.1.1 |
| 4.3 | Phân phối PM2.5 đuôi dài | `research/eda/visualizations/02a_pm25_fat_tailed_distribution.png` | §4.1.2 |
| 4.4 | Đỉnh dị thường PM2.5 | `research/eda/visualizations/02b_pm25_erratic_spikes.png` | §4.1.2 |
| 4.5 | Rolling Correlation (Concept Drift) | `research/eda/visualizations/03a_rolling_correlation.png` | §4.1.3 |
| 4.6 | Hexbin đa biến | `research/eda/visualizations/03b_hexbin_multivariate.png` | §4.1.3 |
| 4.7 | Missing Data Barcode | `research/eda/visualizations/04a_missing_barcode.png` | §4.1.4 |
| 4.8 | Giới hạn phục hồi dữ liệu | `research/eda/visualizations/04b_recovery_limits.png` | §4.1.4 |

### §4.5 Residual Diagnostics

| Hình | Mô tả | File Path | Section |
|------|-------|-----------|---------|
| 4.9 | Diagnostics GRU h=1 | `research/diagnostics/statistical_tests/figures/diagnostics_1h_GRU.png` | §4.5 |
| 4.10 | Diagnostics LightGBM h=1 | `research/diagnostics/statistical_tests/figures/diagnostics_1h_LightGBM.png` | §4.5 |
| 4.11 | Diagnostics GRU h=6 | `research/diagnostics/statistical_tests/figures/diagnostics_6h_GRU.png` | §4.5 |
| 4.12 | Diagnostics LightGBM h=6 | `research/diagnostics/statistical_tests/figures/diagnostics_6h_LightGBM.png` | §4.5 |
| 4.13 | Diagnostics GRU h=24 | `research/diagnostics/statistical_tests/figures/diagnostics_24h_GRU.png` | §4.5 |
| 4.14 | Diagnostics LightGBM h=24 | `research/diagnostics/statistical_tests/figures/diagnostics_24h_LightGBM.png` | §4.5 |
| 4.15 | Diagnostics Persistence h=6 | `research/diagnostics/statistical_tests/figures/diagnostics_6h_Persistence.png` | §4.5 |

### §4.6 SHAP Explainability

| Hình | Mô tả | File Path | Section |
|------|-------|-----------|---------|
| 4.16 | SHAP Bar — h=1 | `research/figures/shap/shap_bar_1h.png` | §4.6 |
| 4.17 | SHAP Bar — h=6 | `research/figures/shap/shap_bar_6h.png` | §4.6 |
| 4.18 | SHAP Bar — h=24 | `research/figures/shap/shap_bar_24h.png` | §4.6 |
| 4.19 | SHAP Beeswarm — h=1 | `research/figures/shap/shap_beeswarm_1h.png` | §4.6 |
| 4.20 | SHAP Beeswarm — h=6 | `research/figures/shap/shap_beeswarm_6h.png` | §4.6 |
| 4.21 | SHAP Beeswarm — h=24 | `research/figures/shap/shap_beeswarm_24h.png` | §4.6 |
| 4.22 | GRU Permutation Importance — h=1 | `research/figures/shap/gru_permutation_1h.png` | §4.6 |
| 4.23 | GRU Permutation Importance — h=6 | `research/figures/shap/gru_permutation_6h.png` | §4.6 |
| 4.24 | GRU Permutation Importance — h=24 | `research/figures/shap/gru_permutation_24h.png` | §4.6 |

### §4.8 Prediction Intervals

| Hình | Mô tả | File Path | Section |
|------|-------|-----------|---------|
| 4.25 | Conformal Prediction — LightGBM h=1 | `research/figures/prediction_intervals/pi_conformal_prediction_LightGBM_1h.png` | §4.8 |
| 4.26 | Conformal Prediction — LightGBM h=6 | `research/figures/prediction_intervals/pi_conformal_prediction_LightGBM_6h.png` | §4.8 |
| 4.27 | Conformal Prediction — LightGBM h=24 | `research/figures/prediction_intervals/pi_conformal_prediction_LightGBM_24h.png` | §4.8 |

### §4.10 Ablation Study

| Hình | Mô tả | File Path | Section |
|------|-------|-----------|---------|
| 4.28 | Ablation Outlier Impact | `research/figures/ablation_outlier_impact.png` | §4.10 |

---

## Ghi chú sử dụng

- **Chèn vào Word:** Copy hình từ đường dẫn → paste vào đúng section → thêm caption "Hình X.Y: [Mô tả]"
- **SHAP Dependency Plots** (15 hình bổ sung trong `research/figures/shap/shap_dep_*.png`): Chỉ chèn khi cần minh họa chi tiết, có thể đưa vào phụ lục
- **Stationarity plots** (5 hình): Chèn đại diện 2-3 hình vào §3.6, còn lại đưa phụ lục
