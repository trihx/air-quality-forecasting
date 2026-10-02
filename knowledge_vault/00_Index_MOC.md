---
title: "Map of Content (MOC): Bản Đồ Tri Thức Thứ Hai Đề Án Dự Báo PM2.5 Sa Đéc"
tags: [moc, time-series, pm25, sa-dec, index, architecture]
aliases: ["Index MOC", "Second Brain Index", "Bản Đồ Tri Thức"]
domain: architecture
created: "2026-10-02"
status: completed
---

# 🧠 Bản Đồ Tri Thức Thứ Hai (Second Brain MOC): Hệ Thống Dự Báo PM2.5 Đa Độ Phân Giải

> **Đề án Thạc sĩ:** Ứng dụng Business Intelligence để phân tích dữ liệu môi trường cho một huyện (TP. Sa Đéc, Đồng Tháp)  
> **Tác giả:** Hoàng Xuân Trí (trihx) — M2522016 | **CBHD:** TS. Nguyễn Minh Khiêm  
> **Ngành:** Hệ thống Thông tin (Mã ngành: 8480104) — Trường ĐH Cần Thơ (QĐ 1799/QĐ-ĐHCT)

---

## 🎯 Sợi Chỉ Đỏ & Tuyên Bố Học Thuật Cốt Lõi
Hệ thống giải quyết trọn vẹn nghịch lý dữ liệu quan trắc IoT môi trường thực tế tại cấp huyện: **tỷ lệ khuyết thiếu cao (74% qua 38 tháng)** và **phân phối hạt mịn đuôi nặng (Fat-tailed)** bằng một kiến trúc Business Intelligence 3 tầng hoàn chỉnh:
1. **Kỷ luật kỹ nghệ dữ liệu thép:** Loại bỏ 19.810 giờ khuyết dài chống sinh dữ liệu ảo (anti-hallucination), bảo tồn 66 đỉnh ô nhiễm bằng Domain Bounds [0, 500] µg/m³, và kỷ luật `shift(1)` triệt tiêu 100% rò rỉ dữ liệu.
2. **Khung đa độ phân giải:** Xác lập điểm ngọt Pareto 30 phút và chứng minh mạng GRU 15 phút phá vỡ bẫy tự tương quan của dữ liệu chuỗi 1h.
3. **Định lượng độ bất định & Khả năng giải thích:** Conformal Prediction kết hợp ACI đạt độ phủ thực nghiệm 90.2% và Tree SHAP phát hiện điểm chuyển pha vi khí hậu 14–17 µg/m³.

---

## 🗺️ Cấu Trúc Bản Đồ Tri Thức 7 Chuyên Đề

```mermaid
mindmap
  root((PM2.5 Second Brain))
    Pipeline & Data Engineering
      [[01_Pipeline_7_Steps]]
      [[02_Tiered_Imputation_and_Data_Sparsity]]
      [[03_Outlier_Trap_and_Domain_Bounds]]
      [[04_Multi_Resolution_Framing]]
    Data Integrity & Anti-Leakage
      [[01_Anti_Leakage_Shift1_Discipline]]
      [[02_Temporal_Split_and_Test_on_Real_Only]]
    Feature Engineering
      [[01_Feature_Store_119_Features]]
      [[02_Stationarity_ADF_KPSS]]
    Models & Architectures
      [[01_Benchmark_Models_Overview]]
      [[02_Autocorrelation_Trap_and_GRU_15m]]
      [[03_Sweet_Spot_30m_Ensemble]]
      [[04_Hyperparameter_Tuning_Optuna]]
    Evaluation & Uncertainty
      [[01_Metrics_Standard_MASE_over_RMSE]]
      [[02_Diebold_Mariano_Hypothesis_Testing]]
      [[03_Uncertainty_Quantification_CQR_ACI]]
    Explainability & Domain Insights
      [[01_Tree_SHAP_vs_Permutation_Importance]]
      [[02_Threshold_Tipping_Point_14_17]]
      [[03_Diurnal_Cycle_and_Sa_Dec_Context]]
    Defense Playbook & FAQ
      [[01_Master_Defense_QnA_Hoi_Dong]]
      [[02_Slides_Narrative_and_Key_Arguments]]
      [[03_Limitations_and_Actionable_Solutions]]
```

---

### 1. 🏗️ Chuyên Đề 1: Kỹ Nghệ Dữ Liệu & Quy Trình 7 Bước
* [[01_Pipeline_7_Steps]]: Quy trình nghiên cứu khép kín từ 209.594 bản ghi thô qua 7 chặng kỹ thuật.
* [[02_Tiered_Imputation_and_Data_Sparsity]]: Chiến lược phục hồi 3 tầng: Spline (≤6h), KNN (6–24h) và dũng cảm loại bỏ 19.810h khuyết dài >24h.
* [[03_Outlier_Trap_and_Domain_Bounds]]: Phân tích bẫy IQR 3.0 gọt nhầm 66 đỉnh ô nhiễm và giải pháp Domain Bounds [0, 500] µg/m³.
* [[04_Multi_Resolution_Framing]]: Khung 3 độ phân giải (15m, 30m, 1h) và phát hiện điểm ngọt Pareto 30 phút.

### 2. 🛡️ Chuyên Đề 2: Liêm Chính Dữ Liệu & Chống Rò Rỉ (Anti-Leakage)
* [[01_Anti_Leakage_Shift1_Discipline]]: Kỷ luật bắt buộc `shift(1)` cho 100% lag/rolling/diff; bóc tách vụ việc $R^2=1.000$ ảo về $0.267$ thực chất.
* [[02_Temporal_Split_and_Test_on_Real_Only]]: Phân tách thời gian 80:10:10 và nguyên tắc bất biến: chỉ tính điểm kiểm định trên dữ liệu đo thật (`is_imputed == 0`).

### 3. ⚙️ Chuyên Đề 3: Kỹ Nghệ Đặc Trưng (Feature Engineering)
* [[01_Feature_Store_119_Features]]: Kho đặc trưng 119 chiều thuộc 6 nhóm (Lag, Rolling, EWMA, Fourier, Calendar, Domain interactions).
* [[02_Stationarity_ADF_KPSS]]: Kiểm định tính dừng kết hợp ADF (Trend-stationary) và KPSS; chứng minh tính phi chuẩn (Skewness 2.0046, Kurtosis 6.1458).

### 4. 🤖 Chuyên Đề 4: Kiến Trúc Mô Hình & Tối Ưu Hóa
* [[01_Benchmark_Models_Overview]]: Đối chuẩn 11 kiến trúc mô hình (Persistence, ARIMA, SARIMAX, ElasticNet, RF, LightGBM, LSTM, GRU, TFT, Ensemble).
* [[02_Autocorrelation_Trap_and_GRU_15m]]: Giải mã bẫy tự tương quan $r=0.86$ ở chuỗi giờ và đột phá của mạng GRU 15 phút đạt MASE = 0.667.
* [[03_Sweet_Spot_30m_Ensemble]]: Sự thống trị của mô hình Ensemble Weighted tại độ phân giải 30 phút (MASE 0.382 ở 6h và 0.469 ở 24h).
* [[04_Hyperparameter_Tuning_Optuna]]: Quy trình tìm kiếm siêu tham số bằng Optuna TPE qua 50 trials trên tập Validation.

### 5. 📊 Chuyên Đề 5: Tiêu Chuẩn Đánh Giá & Định Lượng Bất Định
* [[01_Metrics_Standard_MASE_over_RMSE]]: Luận giải tại sao MASE là tiêu chuẩn vàng; mẫu số chuẩn hóa đồng nhất $MAE_{Persistence\_1h} = 1.821$ µg/m³.
* [[02_Diebold_Mariano_Hypothesis_Testing]]: Kiểm định ý nghĩa thống kê Diebold-Mariano với hiệu chỉnh Harvey-HLN; giải trình dấu thống kê âm.
* [[03_Uncertainty_Quantification_CQR_ACI]]: Định lượng độ bất định bằng Conformal Quantile Regression kết hợp Adaptive Conformal Inference (ACI) đạt độ phủ 90.2%.

### 6. 🔍 Chuyên Đề 6: Khả Năng Giải Thích (XAI) & Động Lực Học Môi Trường
* [[01_Tree_SHAP_vs_Permutation_Importance]]: Phân luồng giải thích: Tree SHAP chính xác cho LightGBM vs Permutation Importance cho mạng nơ-ron sâu.
* [[02_Threshold_Tipping_Point_14_17]]: Phát hiện điểm chuyển pha khí quyển phi tuyến 14–17 µg/m³ từ SHAP Dependence Plot.
* [[03_Diurnal_Cycle_and_Sa_Dec_Context]]: Phân tích chu kỳ ngày đêm, đỉnh phát thải 6h sáng, cực tiểu 12h trưa, giao thông thủy bộ và mùa đốt rơm rạ Sa Đéc.

### 7. 🛡️ Chuyên Đề 7: Cẩm Nang Bảo Vệ & Phản Biện Hội Đồng
* [[01_Master_Defense_QnA_Hoi_Dong]]: Bộ 20 câu hỏi - đáp chất vấn chuyên sâu chia theo 3 vai trò (Chủ tịch, Phản biện 1, Phản biện 2).
* [[02_Slides_Narrative_and_Key_Arguments]]: Kịch bản 24 slides thuyết trình, bản đồ thời gian vàng 18 phút và 4 tuyên bố học thuật đanh thép.
* [[03_Limitations_and_Actionable_Solutions]]: 5 hạn chế cốt lõi của đề án và các giải pháp công nghệ đột phá tương ứng (Hardware Ring-Buffer, Neural ODE, PINN, Kohler Correction, Conformal Calibration).

---
*Ghi chú: Toàn bộ tri thức trong vault này được kết nối hai chiều (`[[...]]`) phục vụ việc tra cứu tức thời của Trợ lý AI và nghiên cứu chuyên sâu.*\n