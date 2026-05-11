"""Append new chapters to THESIS_EXPLANATIONS.md."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
target = PROJECT_ROOT / "docs" / "THESIS_EXPLANATIONS.md"

NEW_CONTENT = """

---

## Chương 6: Phân Tích Q-Q Plot — Kiểm Tra Tính Chuẩn (Normality)

### 6.1. Ý nghĩa của Q-Q Plot
Q-Q Plot (Quantile-Quantile Plot) so sánh phân phối thực tế của dữ liệu với phân phối chuẩn lý thuyết (Normal/Gaussian). Nếu dữ liệu tuân theo phân phối chuẩn hoàn hảo, tất cả các điểm dữ liệu sẽ nằm trùng khít lên đường tham chiếu (đường nét đứt đỏ).

### 6.2. Biểu đồ bên trái (Raw PM2.5) — Phân phối KHÔNG chuẩn
- **Hiện tượng:** Đường chấm xanh dương uốn cong lồi xuống ở đoạn giữa và vút hẳn lên cao ở bên phải, hoàn toàn trệch khỏi đường đỏ.
- **Ý nghĩa thống kê:** Dữ liệu PM2.5 gốc **không hề có phân phối chuẩn**. Nó bị lệch phải rất nặng (Right-skewed) và có "đuôi béo" (Fat-tailed / Heavy-tailed).
- **Insight thực tế:** Phản ánh đúng bản chất vật lý của ô nhiễm không khí: Phần lớn thời gian (>80%) PM2.5 ở mức thấp-trung bình (an toàn), nhưng thỉnh thoảng có các đợt bùng phát cực kỳ nghiêm trọng (60-150 µg/m³). Các đỉnh nhọn này kéo dài cái "đuôi" của phân phối.

### 6.3. Biểu đồ bên phải (Log-Transformed PM2.5) — Tiến gần phân phối chuẩn
- **Hiện tượng:** Sau Log-transform, đường chấm xanh lá bám rất sát đường đỏ (chỉ hơi lệch ở 2 đầu mút).
- **Ý nghĩa thống kê:** Phép Log đã "nén" các giá trị cực đoan, giúp dữ liệu tiến gần đến phân phối chuẩn.

### 6.4. Insight đưa vào luận văn (Chương 3 & 5)
> **Biện luận tại sao mô hình tuyến tính thất bại:**
> *"Dựa vào Q-Q Plot, dữ liệu PM2.5 gốc có phân phối fat-tailed, vi phạm nghiêm trọng giả định phân phối chuẩn của các mô hình thống kê truyền thống (ARIMA, Linear Regression). Đây là lý do cốt lõi khiến các mô hình này dự báo rất kém ở các đợt bùng phát ô nhiễm. Do đó, việc sử dụng các mô hình Deep Learning (LSTM, GRU) — vốn không đòi hỏi giả định phân phối chuẩn — là lựa chọn bắt buộc."*

> **Biện luận về Data Transformation:**
> *"Mặc dù Log-Transform giúp tiến gần phân phối chuẩn, thực nghiệm cho thấy áp dụng đồng loạt lại làm GIẢM hiệu năng dự báo ở các điểm cực trị vì Log-transform đã 'cào bằng' các đỉnh ô nhiễm. Nghiên cứu quyết định giữ nguyên phân phối gốc (Fat-tailed) và chỉ dùng StandardScaler/MinMaxScaler."*

---

## Chương 7: Bẫy Outlier Removal (Outlier Removal Trap) & Ablation Study

### 7.1. Phát hiện vấn đề
Khi phân tích biểu đồ "Pre vs Post Imputation", phát hiện rằng tại thời điểm 17:00 ngày 17/03/2022:
- **Raw (resample trung bình giờ):** PM2.5 = 43.0 µg/m³
- **Imputed (sau cleaning pipeline):** PM2.5 = 40.74 µg/m³

Sự chênh lệch do trong giờ đó, cảm biến ghi nhận 30 giá trị (mỗi 2 phút), trong đó có 3 đỉnh cao: 56.0, 63.0, và 71.0 µg/m³. Phương pháp IQR đã tính toán ngưỡng cắt toàn cục là ~54.0 µg/m³ và **xóa nhầm 3 giá trị thật** thành NaN, khiến trung bình giờ giảm từ 43.0 xuống 40.74.

### 7.2. Nguyên nhân gốc rễ
PM2.5 có đặc tính phân phối **fat-tailed** (đuôi béo). Phương pháp thống kê thuần túy (IQR, Z-score) giả định dữ liệu có phân phối gần chuẩn, nên:
- IQR x 3 chỉ cho phép giá trị tối đa ~54 µg/m³
- Mọi giá trị > 54 bị coi là "outlier" và bị xóa
- Thực tế đây là các đợt bùng phát ô nhiễm thật sự

### 7.3. Giải pháp đã áp dụng
Chuyển sang **Domain Bounds** theo chuẩn WHO AQI: [0, 500] µg/m³. Chỉ loại bỏ các giá trị ngoài phạm vi vật lý (lỗi cảm biến), giữ nguyên toàn bộ các đỉnh ô nhiễm thật.

### 7.4. Ablation Study Design
**Mục tiêu:** So sánh Domain Bounds (v9 hiện tại) vs IQR-truncated (mô phỏng lỗi cũ).

**7 mô hình đại diện x 3 horizons x 1 resolution (30m):**
1. Persistence (Baseline)
2. ElasticNet (Linear ML)
3. LightGBM (Tree-based ML)
4. GRU (RNN DL)
5. LSTM (RNN DL)
6. TFT (Transformer DL)
7. Ensemble_Weighted (Ensemble)

**Giả thuyết:**
- MAE trên dữ liệu IQR sẽ **thấp hơn** (vì đã gọt đỉnh khó đoán) -> "False Sense of Accuracy"
- MASE ranking giữa các mô hình sẽ **không thay đổi**
- Extreme Event Accuracy sẽ **tệ hơn nhiều** vì mô hình chưa bao giờ "thấy" đỉnh ô nhiễm

### 7.5. Ý nghĩa trong luận văn
> *"Bài học đắt giá: Việc áp dụng máy móc phương pháp loại nhiễu thống kê thuần túy (IQR, Z-score) lên dữ liệu ô nhiễm không khí là sai lầm. Bụi PM2.5 có đặc tính phân phối đuôi dài, các đợt bùng phát ô nhiễm có thể bị thuật toán xóa nhầm vì tưởng là lỗi cảm biến, dẫn đến làm mịn dữ liệu khiên cưỡng và bỏ sót các cảnh báo nguy hiểm thật sự."*

### 7.6. Kết luận "30 phút là tối ưu" — Đã kiểm chứng
Trong bảng xếp hạng MASE Unified trên toàn bộ 41 models x 3 horizons:
- **30m chiếm 10/15 vị trí top-5** (67%)
- 15m chiếm 4/15 (27%), 1h chỉ 1/15 (7%)
- Kết luận này vững chắc và không cần chạy lại multi-resolution.

---

## Chương 8: Gợi Ý Viết Luận Văn Theo Từng Chương (Thesis Writing Prompts)

> **Hướng dẫn sử dụng:** Khi anh viết đến chương nào, hãy hỏi Agent:
> "Em nhắc anh nội dung nên viết ở Chương X" để Agent tra cứu file này và gợi ý chi tiết.

### Chương 1: Giới thiệu
- Trình bày tầm quan trọng của dự báo PM2.5 đối với sức khỏe cộng đồng tại ĐBSCL
- Nêu rõ **khoảng trống nghiên cứu**: Chưa có nghiên cứu nào áp dụng ML/DL cho IoT PM2.5 tại Sa Đéc, Đồng Tháp
- Phạm vi: Multi-Resolution (15m, 30m, 1h) x Multi-Horizon (1h, 6h, 24h) x 30+ models

### Chương 2: Tổng quan Tài liệu
- Literature Review: 14 bài SOTA đã verified (xem PIPELINE_REFERENCES.md)
- Lý thuyết: ARIMA/SARIMA, LSTM, GRU, TFT, Ensemble Methods
- Giải thích tại sao chọn MASE làm metric chính (Hyndman 2006)

### Chương 3: Phương pháp Nghiên cứu — Data & Preprocessing
- **Data Collection:** IoT sensor tại Sa Đéc (209K records, 2022-2025)
- **Cleaning Pipeline:** Physical Bounds -> Outlier Detection -> Resample -> Interpolation -> Drop NaN
- **BAY OUTLIER REMOVAL:** Giải thích chuyển từ IQR sang Domain Bounds [0-500] (xem Chương 7 ở trên)
- **Q-Q PLOT:** Dùng biểu đồ Q-Q để biện luận tại sao PM2.5 không tuân phân phối chuẩn -> DL phù hợp hơn (xem Chương 6)
- **Tiered Imputation:** Spline (gap <=6h) -> KNN (6-24h) -> Drop (>24h)
- **Anti-Leakage:** shift(1) bắt buộc cho diff, pct_change, rolling features
- **Feature Engineering:** 119 tabular features (lags, rolling, Fourier, calendar)

### Chương 3: Phương pháp Nghiên cứu — Models & Training
- **Segment-aware training:** Giải quyết False Continuity do missing data
- **Fair vs Expert Pipeline:** Fair (119 tabular features) vs Expert (raw 5 variables -> DL tự trích xuất)
- **Test-on-Real-Only:** is_imputed == 0 filter bắt buộc trong test set
- **Data Split:** 80/10/10 temporal split (không random!)

### Chương 4: Kết quả Thực nghiệm
- **Best models:** GRU_15m (1h, MASE=0.667), Ensemble_30m (6h, MASE=0.382), Ensemble_30m (24h, MASE=0.469)
- **Key finding:** 30 phút là resolution tối ưu (10/15 top-5 positions)
- **Persistence trap:** Autocorrelation ~0.97 ở 1h khiến Naive gần như bất khả chiến bại
- **Ablation Study:** (KẾT QUẢ SẼ CẬP NHẬT SAU KHI CHẠY v10)
- **SHAP Analysis:** Horizon shift effect, Threshold bùng phát ô nhiễm ở ~17-18 ug/m3

### Chương 5: Bàn luận & Kết luận
- **Đóng góp khoa học:** Multi-Resolution x Multi-Horizon methodology cho IoT PM2.5
- **Đóng góp kỹ thuật:** Anti-Leakage 4 tầng + Tiered Imputation + Test-on-Real-Only
- **Hạn chế:** Data Sparsity (89 ngày/năm mù), đơn trạm, 4 biến phụ
- **BÀI HỌC OUTLIER TRAP:** Đoạn biện luận quan trọng (xem Chương 7.5)
- **Hướng phát triển:** FD-1 -> FD-6 (xem Dashboard Conclusion page)
- **External Data Policy:** Không merge vào pipeline chính (bias hệ thống PM2.5: IoT ~13.7 vs CAMS ~22.2)
"""

with open(target, "a", encoding="utf-8") as f:
    f.write(NEW_CONTENT)

print(f"Appended {len(NEW_CONTENT)} chars to {target}")
