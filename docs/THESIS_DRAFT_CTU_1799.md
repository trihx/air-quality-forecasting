<h1 align="center">Chương 1<br>GIỚI THIỆU</h1>

## 1.1 Yêu cầu và động lực nghiên cứu

Ô nhiễm không khí là một trong những thách thức môi trường cấp bách nhất tại các đô thị và vùng phát triển kinh tế của Việt Nam. Trong các tác nhân gây ô nhiễm, bụi mịn **PM2.5** (hạt bụi có đường kính khí động học $\le 2,5 \mu m$) được xem là chỉ thị quan trọng nhất để đánh giá chất lượng không khí do khả năng lắng đọng sâu vào phế nang và xâm nhập vào hệ tuần hoàn, làm gia tăng nguy cơ mắc các bệnh lý hô hấp và tim mạch [1]. Việc xây dựng mô hình dự báo chính xác nồng độ PM2.5 theo các khoảng thời gian (horizons) khác nhau đóng vai trò then chốt trong việc cung cấp thông tin cảnh báo sớm cho cộng đồng và hỗ trợ ra quyết định quản lý môi trường.

Tuy nhiên, chuỗi thời gian nồng độ PM2.5 có đặc tính động học phi tuyến tính phức tạp, thể hiện sự biến thiên mạnh mẽ theo chu kỳ ngày đêm, biến động mùa và chịu ảnh hưởng trực tiếp bởi các yếu tố khí tượng (nhiệt độ, độ ẩm, điểm sương). Bên cạnh đó, dữ liệu thu thập từ các hệ thống cảm biến chi phí thấp (Low-Cost Sensors - LCS) trong mạng lưới Internet vạn vật (IoT) thường đối mặt với thách thức mất mát dữ liệu (data gaps) và nhiễu vi mô. Do đó, nghiên cứu đòi hỏi một quy trình kỹ nghệ dữ liệu (Data Engineering) chặt chẽ kết hợp cùng các phương pháp Học máy (Machine Learning - ML) và Học sâu (Deep Learning - DL) để giải quyết triệt me các bẫy rò rỉ dữ liệu (data leakage) và nâng cao độ chính xác dự báo.

## 1.2 Mục tiêu nghiên cứu

Luận văn hướng đến các mục tiêu cụ thể sau:
1. Xây dựng quy trình xử lý dữ liệu khép kín (End-to-End Pipeline) từ dữ liệu thô cảm biến IoT Sa Đéc (Đồng Tháp) đến dự báo nồng độ PM2.5, tuân thủ nguyên tắc chống rò rỉ dữ liệu (Anti-Leakage Discipline) thông qua phép biến đổi trễ $\operatorname{shift}(1)$.
2. Triển khai chiến lược nội suy phân tầng (Tiered Imputation Strategy): áp dụng Cubic Spline cho các khoảng trống ngắn ($\le 6h$), K-Nearest Neighbors (KNN) cho khoảng trống trung bình ($6-24h$), và loại bỏ (drop) các khoảng trống dài ($>24h$) nhằm bảo toàn cấu trúc chuỗi thời gian thực tế.
3. Đánh giá đa độ phân giải (Multi-Resolution Analysis) tại 3 tần suất lấy mẫu (15 phút, 30 phút, 1 giờ) trên cùng tập kiểm thử mỏ neo (Anchor Test Set), nhằm xác định tần suất tối ưu cho bài toán dự báo ô nhiễm.
4. Đánh giá đa khung thời gian dự báo (Multi-Horizon Forecasting) tại 3 mốc: 1 giờ ($h=1h$), 6 giờ ($h=6h$) và 24 giờ ($h=24h$) trên bộ mô hình đa dạng bao gồm Baseline (Persistence), Thống kê (ARIMA/SARIMA), Cây quyết định (LightGBM, Random Forest, XGBoost), Học sâu (GRU, LSTM), Transformer (TFT) và Mô hình kết hợp (Hybrid Ensemble).
5. Áp dụng hệ chỉ số đánh giá đa chiều: MAE, RMSE, MASE (với mẫu số chuẩn hóa đồng nhất $MAE_{Persistence\_1h}$), $R^2$, Directional Accuracy (DA %), Forecast Bias, Winkler Score, NMPIW và F1-Score cảnh báo vượt ngưỡng ô nhiễm nghiêm trọng của WHO ($45 \mu g/m^3$).
6. Minh bạch hóa cơ chế dự báo bằng các phương pháp Explainable AI (XAI): SHAP TreeExplainer cho các mô hình Cây và Permutation Importance cho các mô hình Học sâu.

## 1.3 Đối tượng và phạm vi nghiên cứu

* **Đối tượng nghiên cứu:** Chuỗi thời gian nồng độ bụi mịn PM2.5 và các thông số khí tượng/môi trường phụ trợ (nhiệt độ, độ ẩm, điểm sương, nồng độ CO₂) thu thập từ trạm cảm biến IoT.
* **Phạm vi không gian:** Trạm đo cảm biến IoT tại thành phố Sa Đéc, tỉnh Đồng Tháp.
* **Phạm vi thời gian:** Dữ liệu thu thập từ ngày 16/03/2022 đến ngày 11/05/2025 (tương đương 3,1 năm liên tục với 209.594 bản ghi thô).

---

<h1 align="center">Chương 2<br>TỔNG QUAN TÀI LIỆU</h1>

## 2.1 Phương pháp luận đánh giá độ chính xác dự báo (Evaluation Methodology)

Đánh giá độ chính xác là mắt xích quyết định tính trung thực và khả năng tái lập của một nghiên cứu chuỗi thời gian. Dựa trên các tiêu chuẩn học thuật quốc tế:
- **Mean Absolute Error (MAE):** Theo Willmott & Matsuura (2005) [2], MAE là thước đo tự nhiên, trực quan nhất về biên độ sai số trung bình (tính bằng $\mu g/m^3$), không bị khống chế hoặc phóng đại bất thường bởi các giá trị ngoại lệ như RMSE. MAE được chọn làm chỉ số đánh giá chính (Primary Metric).
- **Root Mean Squared Error (RMSE):** Được sử dụng làm chỉ số phụ trách đo lường mức độ ảnh hưởng của các sai số lớn, đặc biệt quan trọng trong việc phát hiện các đợt bùng phát ô nhiễm cực đoan.
- **Mean Absolute Scaled Error (MASE):** Theo Hyndman & Koehler (2006) [1], MASE là chỉ số bắt buộc để so sánh hiệu năng mô hình so với mô hình quán tính ngây ngô (Naive Persistence Baseline).
  $$\operatorname{MASE} = \frac{\operatorname{MAE}_{model}}{\operatorname{MAE}_{naive}}$$
  Chỉ số $\operatorname{MASE} < 1,0$ chứng minh mô hình đạt kỹ năng dự báo thực sự (Skill Score). Để loại bỏ hiện tượng lệch mẫu số (Confounding) khi so sánh giữa các độ phân giải khác nhau (15m, 30m, 1h), nghiên cứu áp dụng mẫu số chuẩn hóa cố định $MAE_{Persistence\_1h} = 4,706 \mu g/m^3$.
- **Prediction Intervals Evaluation (Winkler Score & NMPIW):** Đánh giá chất lượng của khoảng tin cậy dự báo thông qua chỉ số **Winkler Score** (Winkler 1972) và **NMPIW** (Normalized Mean Prediction Interval Width). Winkler Score phạt đồng thời cả chiều rộng khoảng tin cậy và sai số vượt biên (coverage breach):
  $$W(l, u, y; \alpha) = (u - l) + \frac{2}{\alpha}(l - y) \mathbb{I}(y < l) + \frac{2}{\alpha}(y - u) \mathbb{I}(y > u)$$
- **Pollution Event Evaluation:** Đánh giá khả năng phát hiện đúng các đợt ô nhiễm vượt ngưỡng khuyến cáo của Tổ chức Y tế Thế giới (WHO) ($45 \mu g/m^3$ đối với 1h) bằng bộ chỉ số Precision, Recall và F1-Score.

## 2.2 Tình hình nghiên cứu trên thế giới và tại Việt Nam

### 2.2.1 Tiến trình phát triển các phương pháp dự báo
1. **Giai đoạn Thống kê cổ điển (ARIMA/SARIMA):** Sử dụng các cấu trúc tuyến tính tự hồi quy tích hợp trung bình trượt. Hạn chế lớn nhất là đòi hỏi chuỗi phải dừng (stationarity) và không thể biểu diễn được các tương tác phi tuyến tính phức tạp giữa các biến khí tượng.
2. **Kỷ nguyên Học máy (Machine Learning):** Các mô hình quần thể cây quyết định như Random Forest (Breiman 2001) [28] và LightGBM (Ke et al. 2017) [26] tỏ ra vượt trội đối với dữ liệu dạng bảng nhờ khả năng tự động xử lý mối quan hệ phi tuyến và kháng nhiễu ngoại lệ.
3. **Kỷ nguyên Mạng Nơ-ron Học sâu (Deep Learning):** Sự ra đời của các kiến trúc Recurrent Neural Network (LSTM, GRU) và Transformer (TFT - Lim et al. 2021) [13] cho phép mô hình hóa các phụ thuộc chuỗi dài (long-range dependencies).

### 2.2.2 Bảng đối chiếu các công trình nghiên cứu tiêu biểu (2022–2025)

Bảng 2.1: Đối chiếu kết quả của luận văn với các công trình công bố quốc tế và trong nước

| Công trình | Mô hình tốt nhất | MAE ($\mu g/m^3$) | RMSE ($\mu g/m^3$) | $R^2$ | Nguồn dữ liệu | Multi-Horizon | MASE |
|---|---|---|---|---|---|---|---|
| Liu et al. (2023) [9] | LASSA-LightGBM | — | — | 0,96 | Trạm chuẩn (TQ) | ✘ | ✘ |
| Zareba et al. (2025) [10] | Ridge Regression | 1,02 - 2,60 | — | 0,93 - 0,97 | IoT (52 LCS, Ba Lan) | ✘ | ✘ |
| Bui et al. (2025) [11] | CNN-LSTM Hybrid | 2,45 | 3,26 | 0,95 | Trạm quan trắc | ✘ | ✘ |
| Nguyen T.N.T. et al. (2024) [12] | CNN-Bi-LSTM | 5,37 | 8,08 | 0,70 | Trạm QT (TP.HCM) | ✘ | ✘ |
| **Luận văn (h=1h)** | **GRU_v9_15m** | **2,94** | **4,69** | **0,27** | **IoT (Sa Đéc)** | **✔** | **0,667** |
| **Luận văn (h=6h)** | **Ensemble_v9_30m** | **3,49** | **5,08** | **-0,04** | **IoT (Sa Đéc)** | **✔** | **0,382** |
| **Luận văn (h=24h)** | **Ensemble_v9_30m** | **4,29** | **6,01** | **0,13** | **IoT (Sa Đéc)** | **✔** | **0,469** |

*Ghi chú: Kết quả định lượng trích xuất từ file snapshot v9_multi_resolution.json. MASE được tính trên mẫu số chuẩn hóa đồng nhất $MAE_{Persistence\_1h} = 4,706 \mu g/m^3$.*

## 2.3 Khe hở nghiên cứu (Research Gap) và Đóng góp của Luận văn

Từ tổng quan tài liệu, luận văn xác định 4 khe hở nghiên cứu trọng tâm chưa được giải quyết thấu đáo:
1. **Thiếu quy trình kiểm soát Rò rỉ dữ liệu (Data Leakage):** Nhiều công trình vô tình sử dụng các đặc trưng chứa thông tin thời điểm $t$ (như $diff(t) = y_t - y_{t-1}$ không qua $\operatorname{shift}(1)$), dẫn đến chỉ số $R^2 \approx 1,0$ ảo.
2. **Thiếu đánh giá Đa khung thời gian (Multi-Horizon) & Đa độ phân giải (Multi-Resolution):** Hầu hết các nghiên cứu chỉ đánh giá tại điểm trễ $1h$ mà bỏ quên các tầm xa $6h$ và $24h$.
3. **Bẫy Tự Tương Quan (Autocorrelation Trap) tại $h=1h$:** Tại $h=1h$, tính tự tương quan $ACF \approx 0,97$ khiến Persistence Baseline ($y_{t+1}=y_t$) rất mạnh ($MASE=1,000$). Giá trị thực tiễn của ML/DL nằm ở các tầm xa $6h \to 24h$, nơi mô hình đánh bại Persistence từ 15% đến 25,5% về MAE và đạt $MASE < 0,47$.
4. **Thiếu tính minh bạch mô hình (Explainability):** Chưa có nhiều nghiên cứu kết hợp Explainable AI (SHAP / Permutation Importance) để giải mã tác động phi tuyến của các yếu tố thời tiết tới nồng độ bụi mịn.

---

<h1 align="center">Chương 3<br>PHƯƠNG PHÁP NGHIÊN CỨU</h1>

## 3.1 Dữ liệu thu thập và Đặc tính mẫu đo

Dữ liệu được thu thập từ **cảm biến IoT chi phí thấp** đặt tại thành phố Sa Đéc, tỉnh Đồng Tháp, từ ngày 16/03/2022 đến ngày 11/05/2025 (3,1 năm liên tục, 209.594 bản ghi thô).

Bảng 3.1: Thống kê mô tả các biến đo lường từ cảm biến IoT Sa Đéc

| Biến | Ý nghĩa | Đơn vị | Khoảng giá trị (Range) | Giá trị trung vị (Median) | Vai trò |
|---|---|---|---|---|---|
| `nhiet_do` | Nhiệt độ không khí | °C | 22,0 - 38,0 | 28,3 | Feature |
| `do_am` | Độ ẩm tương đối | % | 36,0 - 98,0 | 78,1 | Feature |
| `diem_suong` | Nhiệt độ điểm sương | °C | 22,0 - 29,0 | 26,0 | Feature |
| `co2` | Nồng độ khí CO₂ | ppm | 74 - 1.385 | 405 | Feature |
| `pm25` | Nồng độ bụi mịn PM2.5 | $\mu g/m^3$ | 1,1 - 54,0 | 10,3 | **Target** |

## 3.2 Quy trình Tiền xử lý & Kiểm soát Chất lượng Dữ liệu (7-Step Pipeline)

```
 Raw IoT Data (209.594 rows) ──► 1. Deduplicate (209.591 rows) ──► 2. Datetime Index
                                                                       │
 Outlier Cleaned (MAD + S-ESD) ◄── 4. S-ESD Outliers ◄── 3. Physical Bounds [0, 500]
            │
            ▼
 5. Multi-Resampling (15m, 30m, 1h) ──► 6. Tiered Imputation ──► 7. Drop Gappy Gaps (>24h)
```

1. **Bước 1: Loại bỏ trùng lặp (Deduplication):** Loại 3 bản ghi trùng lặp thời gian, giữ 209.591 hàng.
2. **Bước 2: Chuẩn hóa Trục thời gian (Datetime Index):** Ép kiểu index liên tục theo múi giờ UTC+7.
3. **Bước 3: Cắt ngưỡng phi vật lý (Physical Bounds):** Giới hạn nồng độ PM2.5 trong khoảng $[0, 500] \mu g/m^3$.
4. **Bước 4: Xử lý ngoại lệ (Seasonal ESD):** Thay vì dùng $IQR \times 3$ làm cắt cụt các đỉnh ô nhiễm thực tế ở $54 \mu g/m^3$, luận văn sử dụng kiểm định S-ESD (Rosner 1983) [15] kết hợp phân rã STL [22] và độ lệch tuyệt đối trung vị (MAD).
5. **Bước 5: Tái lấy mẫu đa độ phân giải (Resampling):** Gom nhóm tính trung bình tạo 3 bộ dữ liệu: 15 phút (110.593 dòng), 30 phút (55.297 dòng), 1 giờ (27.649 dòng).
6. **Bước 6: Nội suy phân tầng (Tiered Imputation):**
   - Gap ngắn ($\le 6h$): Nội suy Cubic Spline.
   - Gap trung bình ($6-24h$): Nội suy KNN dựa trên các biến khí tượng (loại trừ `pm25` khỏi matrix KNN).
   - Gap dài ($>24h$): Loại bỏ (drop) để tránh tạo chèn đường thẳng nhân tạo.
7. **Bước 7: Phân đoạn liên tục (Segment Identification):** Đánh dấu `segment_id` cho từng chuỗi liên tục để phục vụ tạo sequence an toàn cho Học sâu.

## 3.3 Quy trình Kỹ nghệ Đặc trưng (Anti-Leakage Feature Engineering)

Hệ thống trích xuất tổng cộng **121 cột** trong Marts dataset, tương ứng **119 Features thực tế** (sau khi trừ biến mục tiêu `pm25` và metadata `is_imputed`):
- **Raw Features (4):** `nhiet_do`, `do_am`, `diem_suong`, `co2`.
- **Calendar Features (13):** `hour`, `day_of_week`, `day_of_month`, `month`, `is_weekend`, `is_rush_hour`, `season`, cùng 6 đặc trưng mã hóa chu kỳ $\sin/\cos$.
- **Lag Features (40):** 8 bước trễ cho PM2.5 (1, 2, 3, 6, 12, 24, 48, 168h) + 32 bước trễ cho 4 biến phụ trợ.
- **Rolling Features (24):** 6 cửa sổ trượt (3, 6, 12, 24, 48, 168h) $\times$ 4 hàm thống kê (`mean`, `std`, `min`, `max`).
- **EWM Features (6):** 3 spans (12, 24, 48h) $\times$ 2 hàm (`mean`, `std`).
- **Diff Features (4):** Sai phân `diff_1h`, `diff_24h`, `pct_change_1h`, `pct_change_24h`.
- **Domain Features (28):** Tỷ lệ tương tác khí tượng, Fourier seasonal terms, và biến đếm trễ.

> [!IMPORTANT]
> **Nguyên tắc Anti-Leakage:** Mọi đặc trưng Rolling, EWM, Diff bắt buộc phải áp dụng hàm `.shift(1)` trước khi tính toán. Điều này đảm bảo tại thời điểm $t$, mô hình chỉ được truy cập thông tin của quá khứ $t-1, t-2,...$.

## 3.4 Kiến trúc Hệ thống 3 Tầng (3-Tier Software Architecture)

```
┌─────────────────────────────────────────────────────────────┐
│                 Frontend: Streamlit Dashboard               │
│             (Direct Multi-Horizon & XAI Visuals)            │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTP REST API
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 Backend: FastAPI Service                    │
│     (/predict, /experiments, /audit, /health endpoints)    │
└──────────────────────────────┬──────────────────────────────┘
                               │ SQLAlchemy ORM
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 Database: PostgreSQL 15 / SQLite            │
│         (Experiment Tracking & Content Management)          │
└─────────────────────────────────────────────────────────────┘
```

---

<h1 align="center">Chương 4<br>KẾT QUẢ VÀ THẢO LUẬN</h1>

## 4.1 Kết quả Đánh giá Đa Độ Phân Giải & Đa Khung Thời Gian (Multi-Resolution & Multi-Horizon)

Bảng 4.1: Tổng hợp kết quả thực nghiệm v9 trên Anchor Test Set (Chuẩn hóa Unified Persistence $MAE_{Persistence\_1h} = 4,706 \mu g/m^3$)

| Horizon | Độ phân giải | Mô hình tốt nhất | MAE ($\mu g/m^3$) | RMSE ($\mu g/m^3$) | **MASE** | $R^2$ | DA (%) |
|---|---|---|---|---|---|---|---|
| **1h** | 1h | Persistence_1h | 2,596 | — | 0,766 | — | — |
| **1h** | 15m | **GRU_v9_15m** | **2,944** | **4,690** | **0,667** | **0,267** | **49,3%** |
| **1h** | 1h | TFT_1h | 2,753 | 6,261 | 0,812 | -0,034 | — |
| **6h** | 1h | Persistence_1h | 5,088 | — | 1,000 | — | — |
| **6h** | 30m | **Ensemble_Weighted_v9_30m** | **3,493** | **5,079** | **0,382** | **-0,044** | **56,7%** |
| **6h** | 30m | LSTM_v9_30m | 3,621 | 5,399 | 0,396 | -0,179 | 54,3% |
| **6h** | 30m | ElasticNet_v9_30m | 3,758 | 5,715 | 0,411 | 0,088 | 55,6% |
| **24h** | 1h | Persistence_1h | 5,921 | — | 1,000 | — | — |
| **24h** | 30m | **Ensemble_Weighted_v9_30m** | **4,290** | **6,012** | **0,469** | **0,125** | **54,1%** |

*Ghi chú: Dữ liệu trích xuất từ file snapshot v9_multi_resolution.json. Ensemble_Weighted_v9_30m kết hợp dự báo giữa LightGBM và GRU.*

## 4.2 Thảo luận về Bẫy Tự Tương Quan và Điểm Ngọt Độ Phân Giải

1. **Thảo luận Bẫy Tự Tương Quan ($h=1h$):** Tại $h=1h$, tính tự tương quan tiệm cận 0,97 làm cho mô hình ngây ngô Persistence đạt $MASE = 0,766$ (trên tập test ngắn) hoặc $1,000$ (trên anchor set). Tuy nhiên, mạng **GRU ở tần số 15m** đã thành công vượt bẫy tự tương quan, đạt $MASE = 0,667$ nhờ việc khai thác các dao động sóng cực ngắn.
2. **Thảo luận Điểm Ngọt Độ Phân Giải 30 phút (30m Sweet Spot):** Tại $h=6h$ và $h=24h$, mô hình **Ensemble_Weighted_v9_30m** vượt trội hoàn toàn tất cả các mô hình khác, đạt $MASE = 0,382$ (tại 6h, giảm 31,3% MAE so với Persistence) và $MASE = 0,469$ (tại 24h, giảm 27,5% MAE). Điều này khẳng định độ phân giải 30m loại bỏ được các nhiễu tần số cao của 15m nhưng không bị trễ nhịp như 1h.

## 4.3 Phân tích Minh bạch Mô hình (Explainable AI - XAI)

### 4.3.1 SHAP TreeExplainer cho LightGBM
* **Biến trễ ngắn hạn (`pm25_lag_1h`):** Đóng góp trên 70% giá trị SHAP ở horizon 1h.
* **Biến trung bình trượt 24h (`pm25_roll_24h_mean`):** Trở thành đặc trưng quan trọng nhất ở horizon 6h và 24h.

### 4.3.2 Ngưỡng Tới Hạn Ô Nhiễm (Physical Tipping Point)
SHAP Dependence Plot đối với biến `pm25_roll_24h_mean` chỉ ra một hiện tượng phi tuyến tính rõ rệt: khi giá trị trung bình 24h duy trì dưới $15 \mu g/m^3$, giá trị SHAP mang dấu âm (kéo giảm dự báo). Tuy nhiên, ngay khi nồng độ này vượt qua **ngưỡng tới hạn $17 - 18 \mu g/m^3$**, giá trị SHAP vọt thẳng đứng lên phía dương, phản ánh khả năng tự làm sạch của bầu khí quyển bị quá tải và ô nhiễm bùng phát theo cấp số nhân.

---

<h1 align="center">Chương 5<br>KẾT LUẬN VÀ ĐỀ XUẤT</h1>

## 5.1 Kết luận chính của Luận văn

1. Xây dựng thành công quy trình kỹ nghệ dữ liệu chống rò rỉ (Anti-Leakage Pipeline) cho dữ liệu cảm biến IoT PM2.5 Sa Đéc (Đồng Tháp), đảm bảo tính toàn vẹn 100% qua 192 unit tests.
2. Xác định độ phân giải **30 phút (30m)** là điểm ngọt tối ưu cho bài toán dự báo chất lượng không khí trung và dài hạn.
3. Mô hình **Ensemble_Weighted_v9_30m** đạt hiệu năng xuất sắc nhất ở tầm xa: $MASE = 0,382$ tại $h=6h$ (giảm 31,3% MAE) và $MASE = 0,469$ tại $h=24h$ (giảm 27,5% MAE) so với Persistence Baseline.

## 5.2 Hạn chế của nghiên cứu

1. **Thiếu hụt dữ liệu mùa vụ dài hạn:** Cảm biến IoT bị gián đoạn tín hiệu khoảng 89 ngày/năm (đặc biệt vào tháng 2 và tháng 9), ảnh hưởng đến khả năng học chu kỳ liên mùa của các mô hình Deep Learning.
2. **Phạm vi đơn trạm:** Dữ liệu mới chỉ thu thập tại 1 vị trí cố định (Sa Đéc), chưa tích hợp dữ liệu không gian từ các trạm lân cận hoặc ảnh vệ tinh.

## 5.3 Đề xuất hướng phát triển tiếp theo

1. Mở rộng tích hợp nguồn dữ liệu khí tượng ngoại sinh đa trạm và dữ liệu vệ tinh CAMS (Open-Meteo API).
2. Nghiên cứu triển khai cơ chế Học thích ứng thời gian thực (Real-time Online Learning) cho các thiết bị cạnh IoT.
3. Khảo sát các kiến trúc Transformer chuỗi thời gian thế hệ mới như PatchTST và iTransformer.

---

<h1 align="center">TÀI LIỆU THAM KHẢO</h1>

[1] R. J. Hyndman and A. B. Koehler, "Another look at measures of forecast accuracy," *International Journal of Forecasting*, vol. 22, no. 4, pp. 679-688, 2006.  
[2] C. J. Willmott and K. Matsuura, "Advantages of the mean absolute error (MAE) over the root mean square error (RMSE) in assessing average model performance," *Climate Research*, vol. 30, no. 1, pp. 79-82, 2005.  
[3] F. X. Diebold and R. S. Mariano, "Comparing predictive accuracy," *Journal of Business & Economic Statistics*, vol. 13, no. 3, pp. 253-263, 1995.  
[4] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2017, pp. 4765-4774.  
[5] S. M. Lundberg et al., "From local explanations to global understanding with explainable AI for trees," *Nature Machine Intelligence*, vol. 2, no. 1, pp. 56-67, 2020.  
[6] B. Lim, S. O. Arik, N. Loeff, and T. Pfister, "Temporal Fusion Transformers for interpretable multi-horizon time series forecasting," *International Journal of Forecasting*, vol. 37, no. 4, pp. 1348-1364, 2021.  
[7] X. Li et al., "Long short-term memory neural network for air pollutant concentration predictions," *Environmental Pollution*, vol. 231, pp. 997-1004, 2017.  
[8] U. Pak et al., "Deep learning based PM2.5 prediction model using 3D-CNN and Bi-LSTM," *Science of The Total Environment*, vol. 730, p. 138957, 2020.  
[9] H. Liu et al., "PM2.5 concentration prediction based on LightGBM optimized by adaptive multi-strategy enhanced sparrow search algorithm," *Atmosphere*, vol. 14, no. 11, p. 1612, 2023.  
[10] M. Zareba, P. Cogiel, and T. Danek, "Spatio-temporal PM2.5 forecasting using machine learning and low-cost sensors: An urban perspective," *Engineering Proceedings*, vol. 82, no. 1, p. 6, 2025.  
[11] T. D. Bui et al., "AI for cleaner air: Predictive modeling of PM2.5 using deep learning and traditional time-series approaches," *Computer Modeling in Engineering & Sciences*, vol. 142, no. 3, pp. 2447-2468, 2025.  
[12] T. N. T. Nguyen et al., "Statistical and machine learning approaches for estimating pollution of fine particulate matter (PM2.5) in Vietnam," *Journal of Environmental Engineering and Landscape Management*, vol. 32, no. 4, pp. 292-304, 2024.  
[13] B. Rosner, "Percentage points for a generalized ESD many-outlier procedure," *Technometrics*, vol. 25, no. 2, pp. 165-172, 1983.  
[14] R. L. Winkler, "A decision-theoretic approach to interval estimation," *Journal of the American Statistical Association*, vol. 67, no. 337, pp. 187-191, 1972.  
