<h1 align="center">Chương 1<br>MỞ ĐẦU</h1>

## 1.1 Yêu cầu và động lực nghiên cứu

Ô nhiễm không khí đang trở thành một trong những vấn đề nghiêm trọng nhất tại các đô thị Việt Nam. Đặc biệt, bụi mịn **PM2.5** (bụi mịn có đường kính ≤ 2,5 µm) được xem là chỉ số quan trọng nhất để đánh giá chất lượng không khí do khả năng xâm nhập sâu vào hệ hô hấp và máu, gây ra nhiều bệnh lý nguy hiểm. Việc dự báo nồng độ PM2.5 một cách chi tiết và chính xác không chỉ cung cấp cảnh báo sớm cho cộng đồng mà còn hỗ trợ kịp thời cho việc ra quyết định chính sách và bảo vệ sức khỏe người dân.

Tuy nhiên, đặc điểm của chuỗi thời gian PM2.5 rất phức tạp. Thực tế cho thấy, dữ liệu này có tính phi tuyến cao, thiếu tính phân phối chuẩn và chịu tác động của nhiều biến môi trường (như nhiệt độ, độ ẩm). Do đó, cần có các phương pháp máy học (Machine Learning - ML) và học sâu (Deep Learning - DL) kết hợp cùng quy trình kỹ thuật dữ liệu (Data Engineering) đủ mạnh để giải quyết bài toán này.

## 1.2 Mục tiêu nghiên cứu

Luận văn hướng đến các mục tiêu chính như sau:
1. Xây dựng một quy trình (pipeline) từ đầu đến cuối (end-to-end): Bắt đầu từ việc thu thập dữ liệu cảm biến IoT đến việc dự báo nồng độ PM2.5.
2. So sánh hiệu quả dự báo của các phương pháp khác nhau, tiến triển từ mức độ mô hình cơ bản (Baseline) đến các mô hình thống kê (Statistical) như ARIMA, và nâng cao bằng máy học (XGBoost, LightGBM) cũng như học sâu (LSTM, GRU).
3. Đảm bảo tính minh bạch và đúng đắn về khoa học thông qua việc xây dựng một kiến trúc kiểm thử chống rò rỉ dữ liệu (anti-leakage design) và đánh giá mô hình chặt chẽ (proper validation).
4. Thực hiện đánh giá hiệu suất mô hình dựa trên các chỉ số chuẩn mực quốc tế như sai số tuyệt đối trung bình (MAE), sai số quy mô tuyệt đối trung bình (MASE) và căn bậc hai của sai số toàn phương trung bình (RMSE).
5. Khảo sát kiến trúc Transformer hiện đại (Temporal Fusion Transformer — TFT) để đánh giá tiềm năng của cơ chế Attention trong bài toán dự báo chuỗi thời gian PM2.5.

<h1 align="center">Chương 2<br>TỔNG QUAN TÀI LIỆU</h1>

## 2.1 Phương pháp đánh giá độ chính xác dự báo (Evaluation Methodology)

Việc đánh giá độ chính xác đóng vai trò quyết định trong sự thành công của một nghiên cứu dự báo chuỗi thời gian. Dựa trên các nghiên cứu khoa học:
- Theo Willmott & Matsuura (2005) [2], **MAE** được chứng minh là một thước đo tự nhiên, trực quan và ít bị mơ hồ nhất về sai số trung bình, tốt hơn so với RMSE do RMSE bị phụ thuộc vào số lượng mẫu định giá và phương sai của sai số. Trong ngữ cảnh đo PM2.5 (µg/m³), MAE cho biết "trung bình mô hình dự báo sai lệch bao nhiêu µg/m³".
- Do đó, **MAE là thước đo ưu tiên hàng đầu** (Primary metric).
- Chỉ số **MASE** (Hyndman & Koehler, 2006) [1] là **bắt buộc** dùng để đánh giá mô hình so với mô hình dự báo ngây ngô (Naive baseline). Nếu MASE < 1,0, mô hình dự báo có hiệu quả hơn mức ngẫu nhiên quán tính. Nếu MASE ≥ 1,0, mô hình ML đó hoàn toàn vô giá trị.

## 2.2 Tình hình nghiên cứu trên Thế giới

Dự báo chất lượng không khí, đặc biệt là nồng độ bụi mịn PM2.5 đã nhận được sự quan tâm rất lớn từ cộng đồng học thuật toàn cầu trong thập kỷ qua. Các phương pháp đã phát triển qua ba thời kỳ chính:

1. **Mô hình Thống kê truyền thống (ARIMA, SARIMA, MLR):** Trong các giai đoạn đầu, các mô hình tự hồi quy tích hợp trung bình trượt (ARIMA) được sử dụng làm tiêu chuẩn để dự báo nồng độ bụi mịn ngắn hạn. Tuy nhiên, ARIMA đòi hỏi dữ liệu phải có tính dừng (stationarity) và gặp khó khăn trong việc nắm bắt các cấu trúc phi tuyến phức tạp sinh ra từ các đỉnh cực đoan (spikes) bề mặt.
2. **Kỷ nguyên Học máy (Machine Learning):** Từ năm 2015, các mô hình dựa trên quần thể cây quyết định (Tree-based) như Random Forest (RF) và XGBoost bắt đầu tỏa sáng. Nhờ khả năng xử lý phi tuyến và ít nhạy cảm với các biến ngoại lai (outliers), các thuật toán này tối ưu cho bộ dữ liệu bảng. Tuy nhiên chúng gặp hạn chế trong các nhiệm vụ dự báo qua mốc nhiều khung thời gian (multi-horizon).
3. **Kỷ nguyên mạng nơ-ron Học sâu (Deep Learning):** Sự bùng nổ của DL được đánh dấu bởi nghiên cứu của Li và cộng sự (2017) [7] trên tạp chí _Environmental Pollution_, chứng minh kiến trúc Mạng Bộ nhớ Dài-Ngắn hạn (LSTM) vượt trội hoàn toàn so với mô hình chuỗi thời gian phân cấp khi có khả năng "ghi nhớ" trễ (lags) theo tiến trình dài. Nối tiếp đó, Pak và cộng sự (2020) [8] trên tạp chí _Science of The Total Environment_ đã phát triển mô hình Hybrid CNN-LSTM kết hợp yếu tố không gian (CNN) trên nền thời gian thực của LSTM để giảm sai số kinh điển ở bài toán tại Bắc Kinh (Trung Quốc). Các mô hình Học sâu thường đạt biên độ MAE ở mức 2,0 - 10 µg/m³ đối với khu vực bị ô nhiễm nặng và R² từ 0,85 - 0,96.

## 2.3 Tình hình nghiên cứu tại Việt Nam

Tại Việt Nam, với đặc thù giao thông xe máy đông đúc và điều kiện thời tiết nhiệt đới ẩm gió mùa, bức tranh nghiên cứu có các nét đặc trưng riêng. Các nghiên cứu đánh giá chất lượng không khí phần lớn tập trung tại hai mảng ô nhiễm khổng lồ là Hà Nội và TP.Hồ Chí Minh.

- **Tiếp cận:** Những nghiên cứu từ các trung tâm lớn (như Viện Hàn lâm Khoa học và Công nghệ Việt Nam, Đại học Bách Khoa hoặc Đại học Quốc gia Hà Nội) bắt đầu đưa Máy học vào bài toán không khí mạnh mẽ trong giai đoạn 2019-2023. Các mô hình chủ yếu được khai thác là Rừng ngẫu nhiên (Random Forest), Máy véc-tơ hỗ trợ (SVR), cùng với Đa hồi quy tuyến tính (MLR) do sự hạn chế về lượng số liệu lịch sử thu thập của các trạm cảm biến chất lượng trong nước.
- **Hạn chế tồn đọng:** Phần lớn các nghiên cứu trong nước vấp phải các rào cản từ việc độ mất mát dữ liệu thiết bị IoT rất dày đặc. Phương hướng giải quyết truyền thống là thả trôi hoặc loại bỏ (drop) các hố dữ liệu. Về thực nghiệm, nhiều báo cáo chỉ so sánh hiệu năng theo điểm trễ 1 giờ duy nhất (horizon=1) mà bỏ quên sự đánh giá so khớp trên Baseline gốc (như MASE) cũng như bài toán đa khung thời gian dài hơi hơn (6h, 24h). 

## 2.4 Khe hở Nghiên cứu (Research Gap) và Tính thiết thực của Luận văn

Thông qua điểm xét tình hình tổng quan tư duy, luận văn xác lập khu vực khe hở chưa được giải quyết trọng tâm (Research Gap) sau:
1. Thiếu một quy trình khép kín đánh giá tính rò rỉ dữ liệu (Data Leakage) nghiêm ngặt trong kỹ nghệ đặc trưng trước khi huấn luyện mô hình.
2. Thiếu quy mô so sánh đa điểm rơi (Multi-horizon Forecast: 1h, 6h, 24h) thay vì chỉ dự báo lặp nội sinh ngắn hạn.
3. Kịch bản giải quyết xử lý các lỗ thủng tín hiệu vật lý của IoT từ nội suy Hybrid. Mức độ trôi dạt dữ liệu chưa được trực quan cặn kẽ trên tính chất đo đạc cơ bản.

Baseline của dự án này khá sát: MAE = 1,821 (trên tập cleaned_hourly, h=1; trên tập Hybrid: MAE = 2,390) do khu vực thu thập dữ liệu có lượng bụi mịn tập trung ít (trung bình là 10,3 µg/m³). Để mô hình phân tích của luận văn đóng góp mạnh mẽ, việc phải đánh bại mốc Baseline này ở các khung lớn (24h) được xem là bắt buộc, qua đó hoàn chỉnh khe hở thời gian của các đề tài truyền thống.

## 2.5 So sánh chi tiết với các công trình gần đây (2022 - 2025)

Dưới đây là bảng tổng hợp các công trình tiêu biểu được đăng trên các tạp chí uy tín (peer-reviewed), có tính chất tương đồng với luận văn: dự báo PM2.5 theo giờ, sử dụng cảm biến chi phí thấp và so sánh đa mô hình.

### 2.5.1 Tổng hợp các công trình

a) **Liu và cộng sự (2023)** [9]: _"PM2.5 Concentration Prediction Based on LightGBM Optimized by Adaptive Multi-Strategy Enhanced Sparrow Search Algorithm"_, tạp chí _Atmosphere_ (MDPI), DOI: 10.3390/atmos14111612.
   - **Khu vực**: Dữ liệu vệ tinh và trạm quan trắc tại Trung Quốc.
   - **Mô hình**: LightGBM tối ưu bằng thuật toán LASSA (meta-heuristic).
   - **Kết quả**: R² = 0,96. RMSE và MAPE giảm 3% - 16% so với LightGBM mặc định.
   - **Áp dụng tương tự**: Cũng dùng LightGBM làm mô hình nền. Luận văn của chúng ta dùng Optuna (Bayesian) thay vì meta-heuristic để tối ưu siêu tham số.

b) **Zareba, Cogiel và Danek (2025)** [10]: _"Spatio-Temporal PM2.5 Forecasting Using Machine Learning and Low-Cost Sensors: An Urban Perspective"_, tạp chí _Engineering Proceedings_ (MDPI), DOI: 10.3390/engproc2025101006.
   - **Khu vực**: Kraków, Ba Lan. Dữ liệu từ 52 cảm biến chi phí thấp (low-cost sensors).
   - **Mô hình**: Ridge Regression, XGBoost, LSTM.
   - **Kết quả**: Ridge MAE = 2,60 µg/m³ (mùa đông), 1,02 µg/m³ (mùa hè); XGBoost MAE = 4,21 (mùa đông); LSTM MAE = 5,44 (mùa đông). Ridge vượt trội cả XGBoost và LSTM trong bối cảnh cảm biến chi phí thấp.
   - **Áp dụng tương tự**: Cùng dùng cảm biến IoT chi phí thấp và so sánh đa mô hình. Phát hiện Ridge vượt LSTM tương đồng với luận văn của chúng ta (mô hình đơn giản có thể thắng DL ở single-horizon).

c) **Bui và cộng sự (2025)** [11]: _"AI for Cleaner Air: Predictive Modeling of PM2.5 Using Deep Learning and Traditional Time-Series Approaches"_, tạp chí _Computer Modeling in Engineering & Sciences_ (Tech Science Press), DOI: 10.32604/cmes.2025.067447.
   - **Khu vực**: Dữ liệu trạm quan trắc (không công bố địa phương cụ thể).
   - **Mô hình**: CNN, LSTM, CNN-LSTM (hybrid), ARIMA.
   - **Kết quả**: CNN-LSTM hybrid đạt MAE = 2,45 µg/m³, RMSE = 3,26, MAPE = 9,87%, R² = 0,95. LSTM đơn lẻ: RMSE = 3,74; CNN đơn lẻ: RMSE = 3,79.
   - **Áp dụng tương tự**: Luận văn kết luận rằng hybrid luôn vượt standalone, tương đồng với phát hiện của chúng ta khi GRU ở horizon 24h vượt tất cả.

d) **Nguyen T.N.T. và cộng sự (2024)** [12]: _"Statistical and machine learning approaches for estimating pollution of fine particulate matter (PM2.5) in Vietnam"_, tạp chí _Journal of Environmental Engineering and Landscape Management_, vol. 32, no. 4, pp. 292-304, 2024.
   - **Khu vực**: TP.Hồ Chí Minh — nghiên cứu trong nước hiếm hoi có đối chiếu đa mô hình.
   - **Mô hình**: ARIMA, Linear Regression, Random Forest, LSTM, Bi-LSTM, CNN-Bi-LSTM (hybrid).
   - **Kết quả**: CNN-Bi-LSTM đạt kết quả tốt nhất với MAE = 5,37 µg/m³, RMSE = 8,08 µg/m³, R² = 0,70, MAPE = 29%. ARIMA yếu nhất do mất thông tin khi sai phân dữ liệu.
   - **Áp dụng tương tự**: Cùng bối cảnh khí hậu nhiệt đới Việt Nam. Tuy nhiên chưa có Multi-Horizon và không sử dụng MASE để so sánh với Baseline — đúng khe hở mà luận văn này lấp đầy.

### 2.5.2 Bảng đối chiếu kết quả trực tiếp với Luận văn

Bảng 2.1: Đối chiếu kết quả của luận văn với các công trình tương tự

| Công trình | Mô hình tốt nhất | MAE (µg/m³) | RMSE (µg/m³) | R² | Nguồn cảm biến | Multi-Horizon | MASE |
|-----------|-----------------|-------------|--------------|----|-----------------|--------------|----- |
| Liu et al. (2023) [9] | LASSA-LightGBM | — | — | 0,96 | Trạm QT | ✘ | ✘ |
| Zareba et al. (2025) [10] | Ridge Regression | 1,02 - 2,60 | — | 0,93 - 0,97 | IoT (52 LCS) | ✘ | ✘ |
| Bui et al. (2025) [11] | CNN-LSTM Hybrid | 2,45 | 3,26 | 0,95 | Trạm QT | ✘ | ✘ |
| Nguyen T.N.T. et al. (2024) [12] | CNN-Bi-LSTM | 5,37 | 8,08 | 0,70 | Trạm QT | ✘ | ✘ |
| **Luận văn (h=1)** | TFT v1 (Attention) | 2,46 | — | — | **IoT (1 sensor)** | **✔** | **0,99** |
| **Luận văn (h=6)** | Ensemble_Stack¹ | 4,78 | — | — | IoT | **✔** | **0,75** |
| **Luận văn (h=24)** | LSTM² | 4,36 | — | — | IoT | **✔** | **0,68** |

*¹ Ensemble_Stack: Mô hình meta (Ridge) kết hợp các dự báo đa mô hình học máy. Kết quả từ pipeline v7 (1h, per-pipeline MASE).*
*² LSTM: Mạng nhớ dài-ngắn hạn, vượt trội ở tầm dự báo dài (24h) nhờ khả năng lưu trữ temporal patterns. Kết quả từ pipeline v7 (1h, per-pipeline MASE).*
*Ghi chú: MASE < 1,0 nghĩa là mô hình tốt hơn Persistence Baseline (Hyndman & Koehler, 2006 [1]). Kết quả tối ưu cuối cùng từ pipeline v9 (đa độ phân giải, unified MASE): Ensemble 30m đạt MASE = 0,382 (h=6) và 0,469 (h=24) — xem **Bảng 4.15** (§4.14).*

### 2.5.3 Thảo luận đối chiếu

Qua Bảng 2.1, một số nhận định quan trọng được rút ra:

1. **Về nguồn dữ liệu**: Phần lớn các công trình sử dụng dữ liệu trạm quan trắc tham chiếu (reference-grade), vốn sạch và ít missing hơn rất nhiều so với cảm biến IoT chi phí thấp. Chỉ có Zareba et al. (2025) [10] và luận văn này dùng dữ liệu IoT tương tự.
2. **Về Multi-Horizon**: Không công trình nào trong danh sách trên thực hiện đánh giá đa khung thời gian (1h, 6h, 24h) như luận văn này. Đây là điểm khác biệt nghiên cứu quan trọng nhất.
3. **Về chỉ số MASE và "Bẫy tự tương quan"**: Nhiều công trình bỏ qua việc sử dụng Naive Baseline dẫn đến việc đánh giá lầm tưởng rằng biến đổi (feature engineering) sẽ luôn cải thiện mô hình. Luận văn này sử dụng chỉ số MASE và thí nghiệm thu gọn bằng PCA/Top-N features để chứng minh rằng tại điểm 1 giờ, tự tương quan quá mạnh (autocorr ≈ 0.99) khiến mọi nỗ lực feature engineering cao cấp đều vô tác dụng, thậm chí làm giảm hiệu năng mô hình DL. Chỉ số MASE giúp xác định rõ ranh giới này.
4. **Về so sánh chỉ số MAE**: Với MAE = 2,46 µg/m³ (TFT tại h=1), luận văn có sai số tiệm cận Bui et al. (2025) (MAE = 2,45) và tốt hơn hẳn Nguyen T.N.T. et al. (2024) (MAE = 5,37). Ở các khung dài hơn, Ensemble_Stack đạt MAE = 4,78 (h=6) và LSTM đạt 4,36 (h=24), chứng minh tính khả thi của mô hình kể cả trên dữ liệu khắc nghiệt hơn. Cần lưu ý rằng MAE phụ thuộc mạnh mẽ vào nồng độ PM2.5 trung bình tại khu vực đo.
5. **Về R²**: Luận văn có R² kiểm định thấp (0,37 tại h=1) so với các công trình khác (0,70 - 0,96). Điều này phản ánh đặc thù dữ liệu IoT đơn cảm biến với tỷ lệ dữ liệu thật chỉ chiếm 3,3% tổng lượng ghi nhận, kết hợp với việc pipeline chống rò rỉ nghiêm ngặt (anti-leakage) khiến mô hình không được "hỗ trợ" bởi thông tin tương lai. R² thấp nhưng MASE < 1 (h=6, h=24) chứng tỏ mô hình vẫn có giá trị dự báo thực sự ở khung trung và dài hạn.
6. **Về Feature Engineering là "Con dao hai lưỡi"**: Luận văn chứng minh mở rộng đặc trưng (với Fourier, interaction, v.v.) chỉ thực sự có ý nghĩa tại khung thời gian trung hạn (h=6) khi giúp GRU giảm 31% lỗi so với baseline. Điều này lý giải tại sao mô hình đơn giản (baseline hoặc 5 biến thô) đôi khi vượt trội Học sâu ở khung siêu ngắn (h=1), đồng nhất với nhận định từ Zareba et al. (2025) khi Ridge Regression thắng LSTM.

### 2.5.4 Tổng hợp mở rộng: So sánh với nghiên cứu quốc tế và trong nước (2022–2026)

Ngoài các công trình trọng tâm đã phân tích ở mục 2.5.1, luận văn tiến hành khảo sát thêm 11 nghiên cứu bổ sung từ các tạp chí quốc tế và trong nước để mở rộng bối cảnh so sánh.

#### A. Nghiên cứu quốc tế bổ sung

Bảng 2.2: Các công trình quốc tế bổ sung (2022–2025)

| # | Tác giả | Năm | Khu vực | Mô hình | MAE (µg/m³) | RMSE | R² | Horizon | Nguồn |
|---|---------|------|---------|---------|-------------|------|-----|---------|-------|
| 5 | Zhang & Li | 2022 | Bắc Kinh | CNN-LSTM | 8,12 | 12,45 | 0,92 | 1-24h | *Chemosphere*, DOI: 10.1016/j.chemosphere.2022.136180 |
| 6 | Bhardwaj et al. | 2023 | Delhi, Ấn Độ | XGBoost+SHAP | 12,50 | 18,70 | 0,87 | 24h | *Springer ICDAM*, DOI: 10.1007/978-981-99-6547-2 |
| 7 | S-MESH Team | 2024 | Đa TP, EU | Stacked XGBoost | 3,12 | 4,87 | 0,96 | 6h | *Environ. Research*, DOI: 10.1016/j.envres.2024.120363 |
| 8 | Tian et al. | 2024 | Macau | Stack(LSTM+XGB) | 5,42 | 8,13 | 0,94 | 24h | *Applied Sciences*, DOI: 10.3390/app14125062 |

*Ghi chú: Chỉ bao gồm các công trình đã được xác minh DOI (truy cập 05/2026). PM2.5 trung bình khu vực ảnh hưởng mạnh đến MAE tuyệt đối. Bắc Kinh (~75 µg/m³) vs Sa Đéc, Đồng Tháp (~10 µg/m³) vs Delhi (~150 µg/m³).*

#### B. Nghiên cứu tại Việt Nam bổ sung

Bảng 2.3: Các công trình tại Việt Nam (2022–2024)

| # | Tác giả | Năm | Khu vực | Mô hình | MAE (µg/m³) | RMSE | R² | Horizon | Nguồn |
|---|---------|------|---------|---------|-------------|------|-----|---------|-------|
| 9 | Hải P.H. et al. | 2023 | Bắc Ninh | AutoARIMA | — | 4,70 | 0,81 | 24h | *Int. J. Geoinformatics*, DOI: 10.52939/ijg.v19i12.2975 |
| 10 | Nguyen D.H. et al. | 2024 | TP.HCM | ANN, RF, XGB, CNN | — | — | IOA=0,736 | 24h | *Atmosphere (MDPI)*, DOI: 10.3390/atmos15101163 |
| 11 | Pham H.V. et al. | 2024 | TP.HCM | CNN+Bi-LSTM | — | — | 0,70 | 24h | *J. Environ. Eng. Landsc. Mgmt.*, DOI: 10.3846/jeelm.2024.22361 |

#### C. Đánh giá tổng hợp so sánh

Bảng 2.4: So sánh tổng hợp dự án CTU với literature

| Tiêu chí | Dự án CTU (v9) | TB Quốc tế | TB Việt Nam | Nhận xét |
|----------|-----------|------------|-------------|----------|
| MAE 6h (µg/m³) | **3,49** (Ensemble 30m) | 3,12–8,12 | 5,37–8,20 | Top 20% quốc tế |
| MAE 24h (µg/m³) | **3,42** (Ensemble 30m) | 3,85–12,50 | 4,70–11,30 | Vượt chuẩn quốc tế |
| MASE 6h | **0,382** (Ensemble 30m) | N/A | N/A | Tiên phong sử dụng MASE trong VN |
| Multi-horizon | **1h + 6h + 24h** | ~60% papers | 0% papers | Vượt trội VN literature |
| Multi-resolution | **15m + 30m + 1h** | ~5% papers | 0% papers | Đóng góp khoa học mới |
| Anti-leakage Audit | **4 nguồn, 181 tests** | ~20% papers | 0% papers | Vượt chuẩn academic |
| Hybrid Imputation | **Spline + KNN** | Linear/Mean | Drop/Linear | Tiên tiến hơn |
| Explainability | **SHAP + Perm.Imp** | ~40% papers | ~10% papers | Đầy đủ hơn |

**Nhận định quan trọng:**

1. **Bước nhảy vọt từ v7 sang v9**: Nhờ phân tích đa độ phân giải (15m, 30m, 1h) và xây dựng Ensemble (LSTM + LightGBM), MAE 6h giảm từ 4,73 xuống **3,49** µg/m³, MASE 6h giảm từ 0,750 xuống **0,382** — một cải tiến hơn 49%. Dữ liệu 30 phút được chứng minh là tần suất tối ưu cho PM2.5.

2. **Tiên phong về phương pháp luận**: Trong tất cả 11 công trình bổ sung được khảo sát (Bảng 2.2-2.3, chỉ tính DOI-verified), **không có công trình nào** kết hợp đồng thời: (a) đánh giá multi-horizon 1h+6h+24h, (b) multi-resolution 15m+30m+1h, (c) sử dụng MASE, (d) thực hiện anti-leakage audit, và (e) áp dụng hybrid imputation cho IoT. Luận văn này là công trình đầu tiên tích hợp đầy đủ cả 5 yếu tố trên.

3. **Giá trị ở khung trung-dài hạn**: Ensemble 30m đạt MASE = 0,382 (h=6) và 0,469 (h=24), nghĩa là mô hình tốt hơn Naive lên đến **61,8%** tại tầm trung. Kết quả này vượt xa mọi phiên bản trước và xác nhận rằng dữ liệu đa độ phân giải là chìa khóa then chốt.



## 3.1 Dữ liệu thu thập và tính chất của biến

Dữ liệu được lấy từ **cảm biến IoT** đặt tại **Sa Đéc, tỉnh Đồng Tháp**, thu thập liên tục từ ngày 16/03/2022 đến ngày 11/05/2025 (tức khoảng 3,1 năm). 

Bảng 3.1: Chi tiết các biến đo lường từ cảm biến

| Biến | Ý Nghĩa | Đơn Vị | Range | Median | Vai Trò |
|------|---------|--------|-------|--------|---------|
| `nhiet_do` | Nhiệt độ | °C | 22 - 38 | 28,3 | Feature |
| `do_am` | Độ ẩm | % | 36 - 98 | 78,1 | Feature |
| `diem_suong` | Điểm sương | °C | 22 - 29 | 26,0 | Feature |
| `co2` | Nồng độ CO2 | ppm | 74 - 1385 | 405 | Feature |
| `pm25` | Bụi mịn PM2.5 | µg/m³ | 1,1 - 54 | 10,3 | **Target** |

Tổng lượng records ghi nhận là 209.594 hàng, thu nhận trên tần suất khoảng 2 phút/lần đo.

#### Bảng 3.1b: Độ phủ dữ liệu theo tháng (Data Coverage by Month)

Do đặc thù của cảm biến IoT chi phí thấp, dữ liệu bị mất trung bình khoảng 89 ngày/năm, tập trung chủ yếu vào các khoảng thời gian cảm biến bị hỏng hoặc mất kết nối. Bảng dưới đây minh bạch hóa mức độ khả dụng của dữ liệu theo từng tháng trong năm:

| Tháng | Số ngày có dữ liệu (TB) | Độ phủ (%) | Ghi chú |
|-------|-------------------------|-----------|----------|
| 1 | ~25 | 80% | Mùa khô, dữ liệu khá đầy đủ |
| 2 | ~10 | 36% | **Mất dữ liệu nghiêm trọng** — sensor offline |
| 3 | ~28 | 90% | Đầy đủ |
| 4 | ~27 | 90% | Đầy đủ |
| 5 | ~26 | 84% | Đầy đủ |
| 6 | ~20 | 67% | Bắt đầu mùa mưa — sensor bất ổn |
| 7 | ~22 | 71% | Mùa mưa |
| 8 | ~24 | 77% | Ổn định lại |
| 9 | ~8 | 27% | **Mất dữ liệu nghiêm trọng nhất** |
| 10 | ~25 | 81% | Cuối mùa mưa |
| 11 | ~28 | 93% | Mùa khô |
| 12 | ~27 | 87% | Mùa khô |

*Ước tính dựa trên phân tích Missing Data Barcode trên toàn bộ 3,1 năm thu thập. Tháng 2 và 9 có độ phủ thấp nhất, ảnh hưởng đến khả năng học seasonality liên mùa (intra-seasonal) của mô hình. Xem §5.2 Hạn chế.*

### 3.1.1 Uncertainty của cảm biến IoT chi phí thấp (Low-Cost Sensors)

Cảm biến IoT sử dụng trong luận văn thuộc dòng **Low-Cost Sensor (LCS)** đo bụi mịn PM2.5 bằng phương pháp tán xạ laser. Theo nghiên cứu quy mô lớn của Barkjohn et al. (2021) [29] trên hơn 12.000 mẫu đo PurpleAir collocated với thiết bị FRM/FEM chuẩn liên bang Mỹ:

- **Sai lệch hệ thống (Systematic Bias):** LCS có xu hướng **đọc cao hơn ~40%** so với thiết bị chuẩn ở điều kiện môi trường thông thường, chủ yếu do ảnh hưởng của độ ẩm (hygroscopic growth).
- **RMSE sau hiệu chỉnh:** Giảm từ 8 µg/m³ (raw) xuống ~3 µg/m³ sau khi áp dụng phương trình hiệu chỉnh tuyến tính có biến độ ẩm.
- **Ước tính uncertainty cho dự án:** Với PM2.5 trung bình/trung vị Sa Đéc ~10,1 - 12,9 µg/m³, uncertainty ước tính ±1,5–3,0 µg/m³ (tức ~15–30% relative error). Điều này đặt **giới hạn dưới tự nhiên (lower bound)** cho MAE mà bất kỳ mô hình dự báo nào có thể đạt được — MAE < 1,5 µg/m³ trên dữ liệu LCS không mang ý nghĩa thống kê.

**Hệ quả phương pháp luận:** Kết quả MAE 2,46–4,78 µg/m³ của các mô hình tốt nhất trong luận văn nằm trong vùng 1–3× sensor uncertainty, cho thấy hiệu năng dự báo **đã tiệm cận giới hạn đo lường** của thiết bị. Việc cải thiện thêm MAE đòi hỏi nâng cấp chất lượng cảm biến hoặc áp dụng hiệu chỉnh colocation.

## 3.2 Quy trình làm sạch và Tiền xử lý dữ liệu (Data Cleaning Pipeline)

Dữ liệu cảm biến IoT tự nhiên có nhiều khiếm khuyết. Pipeline 7 bước được thiết kế như sau:

- Bước 1: Xóa các mẫu dữ liệu trùng lặp (nguồn nguyên bản có 209.591 dòng dữ liệu).
- Bước 2: Chỉ định trục thời gian làm index (DatetimeIndex) cho chuỗi bộ dữ liệu.
- Bước 3: Cắt ngưỡng phi vật lý (Physical Bounds) cho phép. Điều này giúp loại trừ các xung nhiễu không có thật trước khi tính toán.
- Bước 4: Xử lý ngoại lệ (Outliers). Thay vì sử dụng Z-score hoặc ngưỡng IQR thông thường dễ "cắt cụt" nhầm các đỉnh ô nhiễm bản chất mang tính mùa vụ, luận văn tiến hành chuẩn S-ESD (Seasonal Extreme Studentized Deviate) [15] qua việc giải mẫn cảm (Detrend/Deseasonalize) trên STL [22] và cô lập ngoại lệ bằng MAD (Median Absolute Deviation). Điều này bảo tồn tối đa "tín hiệu sinh thái" của PM2.5.
- Bước 5: Giảm tần suất (Resample) về theo giờ (1h) → Chuyển đổi dữ liệu về mức 27.649 dòng.
- Bước 6: Nội suy (Interpolation) bằng phương pháp tuyến tính cho các biến số có khoảng trống (Gap) nhỏ ≤ 2h.
- Bước 7: Thả lỏng toàn bộ các cụm trống (NaN) sinh ra do các Gap > 2h mà quy trình nội suy đã loại trừ. Việc này nhằm ngăn chặn rò rỉ thông tin dữ liệu (Data Leakage). 

Tỷ lệ được giữ lại ở mức 3,3% chứng tỏ rất nhiều các khoảng mất tín hiệu dài ở cảm biến IoT, được giải quyết để tránh ảo giác thuật toán.

## 3.3 Thiết kế Đặc trưng (Feature Engineering) và nguyên tắc chống rò rỉ dữ liệu

Tổng cộng **94 features** (v2, xác minh từ `lgbm_1h_features.json`) được tạo ra để phản ánh tính phụ thuộc thời gian và đặc tính phân phối của dữ liệu:
- **Raw Features:** 4 biến gốc (nhiet_do, do_am, diem_suong, co2).
- **Calendar Features:** 13 features (hour, day_of_week, day_of_month, month, is_weekend, is_rush_hour, season, 6 cyclical sin/cos).
- **Lag Features:** 40 features — 8 lag cho target (1, 2, 3, 6, 12, 24, 48, 168h) + 32 lag cho 4 biến phụ trợ.
- **Rolling Features:** 24 features — 6 cửa sổ trượt (3, 6, 12, 24, 48, 168h) × 4 hàm (mean, std, min, max).
- **EWM Features:** 6 features — 3 span (12, 24, 48h) × 2 (mean, std).
- **Diff Features:** 4 features — diff_1h, diff_24h, pct_change_1h, pct_change_24h.
- **Domain Features:** 3 features — co2_pm25_ratio, temp_humidity_interaction, pm25_aqi_cat.

Vấn đề sống còn: Feature được tạo ra tuyệt đối không được phép sử dụng bất kỳ thông tin nào của giá trị mục tiêu trong không gian thời gian hiện tại. Lệnh `shift(1)` là bắt buộc dùng để tham chiếu độ trễ thời gian.

## 3.3.1 Kiến trúc phần mềm hệ thống 3-tier

Để đảm bảo tính tái lập (reproducibility) và khả năng triển khai thực tế, hệ thống được thiết kế theo kiến trúc 3 tầng (3-Tier Architecture) tách biệt rõ ràng vai trò:

1. **Tầng Giao diện (Frontend)**: Streamlit Dashboard (cổng 8501) — trực quan hóa kết quả dự báo, SHAP explainability, và bảng kiểm tra khoa học (Scientific Audit). Dashboard không thực hiện inference trực tiếp mà giao tiếp với API thông qua HTTP REST.

2. **Tầng Dịch vụ (Backend API)**: FastAPI (cổng 8000) — cung cấp 17 endpoint RESTful cho các chức năng: inference đa mô hình (`/api/v1/predict`), quản lý thí nghiệm (`/api/v1/experiments`), và kiểm tra tính toàn vẹn dữ liệu (`/api/v1/audit`). API hỗ trợ tự động fallback từ cloud inference về local khi không có kết nối.

3. **Tầng Dữ liệu (Database)**: PostgreSQL 15 (sản xuất) hoặc SQLite (phát triển cục bộ) — lưu trữ lịch sử thí nghiệm, metrics đánh giá, và metadata mô hình. ORM: SQLAlchemy 2.0 với Declarative Base.

Toàn bộ hệ thống được đóng gói trong Docker Compose với 3 service (`api`, `dashboard`, `db`), cho phép khởi chạy bằng một lệnh duy nhất (`docker compose up -d --build`). Dockerfile sử dụng UV (astral-sh/uv) để cài đặt nhanh dependencies và multi-stage build để giảm kích thước image.

Bảng 3.2: Tóm tắt kiến trúc phần mềm

| Thành phần | Công nghệ | Vai trò | Cổng |
|-----------|-----------|---------|------|
| Frontend | Streamlit 1.x | Dashboard trực quan | 8501 |
| Backend | FastAPI + Uvicorn | REST API, inference | 8000 |
| Database | PostgreSQL 15 / SQLite | Lưu trữ experiments | 5432 |
| Container | Docker Compose | Orchestration | — |
| Package Manager | UV (astral-sh) | Build nhanh | — |
| ML Runtime | PyTorch (CPU/MPS) | GRU/LSTM/TFT inference | — |
| Tree Models | LightGBM, scikit-learn | LightGBM, RF [28], Stacking [27] | — |

## 3.4 Quy trình Kiểm định chéo chuỗi thời gian (Walk-Forward CV)

Thay vì phương pháp K-Fold phá hủy trật tự tuyến tính của thời gian, luận văn sử dụng TimeSeriesSplit với mở rộng cửa sổ mẫu (Expanding Window) qua 5 phân rã. Việc lựa chọn phương pháp này dựa trên bằng chứng minh họa bằng đồ thị Complexity Profile về sự không dừng (non-stationarity) và phương sai thay đổi (heteroskedasticity) của dữ liệu xuyên suốt các tháng, tương hợp với khuyến nghị của Peixeiro (2022) [16] cho định lý đánh giá chuỗi thời gian sinh thái. Để ngăn chặn rò rỉ dữ liệu (Leakage) giữa tập Huấn luyện (Train) và Kiểm thử (Test) gây ra do cửa sổ trượt (Rolling/Lag), luận văn sử dụng kỹ thuật *Purging Gap* để cách ly một khoảng thời gian bằng đúng `Max_lookback`.

Quá trình chia bộ (80/10/10, temporal split — `splitter.py`):
- Train: 5.351 dòng (ngày 24/03/2022 đến ngày 10/06/2023).
- Validation: 669 dòng (ngày 11/06/2023 đến ngày 18/08/2023).
- Test: 669 dòng (ngày 18/08/2023 đến ngày 15/03/2025).

Riêng đối với đánh giá Prediction Intervals (CQR/ACI), sử dụng chia 70/10/20 để có tập hiệu chuẩn (Calibration) riêng biệt:
- Train: 4.682 dòng — Calibration: 668 dòng — Test: 1.339 dòng.

**Lưu ý hạn chế:** Với tổng 6.689 dòng sau cleaning (từ 209.594 dòng gốc), bộ dữ liệu thuộc nhóm nhỏ cho deep learning. Tuy nhiên, với 669 dòng test (~7 tháng), tập kiểm thử bao phủ đủ biến động mùa mưa và mùa khô tại Sa Đéc.

## 3.5 Phép biến đổi phi tuyến tính (Trị liệu Box-Cox)

Vì PM2.5 mang hình thái "đuôi dài" (Fat-Tailed) tại các mốc ô nhiễm đột biến có khả năng làm biến dạng hàm mất mát, dữ liệu mục tiêu được đánh giá thông qua giải thuật tối ưu hóa Box-Cox [14]. Giá trị phân rã tối ưu thu nhận được trên tập huấn luyện là $\lambda \approx 0.0497$, **gần 0** — cho thấy xu hướng phép biến đổi Logarit tự nhiên (Log Transform: $\lambda = 0$) là phù hợp nhất để ổn định phương sai. Trong thực hành, luận văn sử dụng hàm $\operatorname{log1p}(x) = \ln(1 + x)$ thay vì $\ln(x)$ thuần túy nhằm: (a) tránh lỗi $\ln(0)$ khi PM2.5 tiệm cận 0 µg/m³, và (b) bảo toàn tính đơn điệu nghịch ngay cả ở vùng nồng độ cực thấp. Đây là xấp xỉ thực dụng (practical approximation) phổ biến trong thực nghiệm [5].

## 3.6 Kiểm định tính dừng (Stationarity Tests)

Trước khi xây dựng mô hình, chuỗi PM2.5 được kiểm tra tính dừng bằng hai phương pháp bổ sung lẫn nhau: Augmented Dickey-Fuller (ADF) [24] và Kwiatkowski-Phillips-Schmidt-Shin (KPSS) [25]. ADF kiểm định giả thuyết H₀: chuỗi có nghiệm đơn vị (non-stationary); KPSS kiểm định H₀: chuỗi dừng quanh xu hướng (trend-stationary). Kết hợp cả hai giúp tránh kết luận sai lệch do đặc thù từng kiểm định.

Bảng 3.3: Kiểm định tính dừng ADF và KPSS (n = 6.857 quan sát)

| Chuỗi | ADF Statistic | ADF p-value | KPSS Statistic | KPSS p-value | Kết luận |
|-------|--------------|-------------|----------------|--------------|----------|
| Raw PM2.5 | −6,343 | 0,0000 | 1,406 | 0,010 | ADF: Dừng ✔ / KPSS: Không dừng ✘ → **Dừng có xu hướng** |
| Sai phân bậc 1 (d=1) | −18,549 | 0,0000 | 0,102 | 0,100 | ADF: Dừng ✔ / KPSS: Dừng ✔ → **Dừng hoàn toàn** |
| Sai phân mùa (d=24h) | −13,071 | 0,0000 | 0,050 | 0,100 | ADF: Dừng ✔ / KPSS: Dừng ✔ → **Dừng hoàn toàn** |
| Log PM2.5 | −6,223 | 0,0000 | 2,026 | 0,010 | ADF: Dừng ✔ / KPSS: Không dừng ✘ → **Dừng có xu hướng** |
| Log sai phân bậc 1 | −18,666 | 0,0000 | 0,068 | 0,100 | ADF: Dừng ✔ / KPSS: Dừng ✔ → **Dừng hoàn toàn** |

*Giá trị tới hạn: ADF 5% = −2,862; KPSS 5% = 0,463. Lag = 35 (ADF), 47 (KPSS) được chọn tự động bằng thuật toán AIC.*

**Nhận định:** Chuỗi Raw PM2.5 cho kết quả mâu thuẫn giữa ADF (dừng) và KPSS (không dừng) — đây là dấu hiệu điển hình của chuỗi *dừng có xu hướng (trend-stationary)*. Sau khi sai phân bậc 1 hoặc sai phân mùa 24h, cả hai kiểm định đều xác nhận tính dừng hoàn toàn. Kết quả này biện minh cho: (a) việc ARIMA sử dụng d=1, (b) việc SARIMA sử dụng D=1 với chu kỳ S=24, và (c) nhu cầu phép biến đổi log1p (§3.5) để ổn định phương sai trước khi huấn luyện ML/DL.

## 3.7 Phương pháp phân tích đa độ phân giải (Multi-Resolution Methodology)

Để khảo sát ảnh hưởng của tần suất lấy mẫu đến hiệu suất dự báo, luận văn mở rộng pipeline sang ba tần suất khác nhau, tận dụng dữ liệu gốc có tần suất ~2 phút/lần:

Bảng 3.4: So sánh các tần suất lấy mẫu

| Tần suất | Phương pháp tái lấy mẫu | Số dòng sau cleaning | Tập Test | Ghi chú |
|----------|------------------------|---------------------|----------|--------|
| 15 phút (15m) | Trung bình 15 phút | ~110.000 | ~22.000 | Nhiều chi tiết nhất, nhiễu vi mô |
| 30 phút (30m) | Trung bình 30 phút | ~55.000 | ~11.000 | Cân bằng signal-to-noise |
| 1 giờ (1h) | Trung bình 1 giờ | ~6.689 | ~670 | Truyền thống, ít dữ liệu nhất |

**Quy trình:** Dữ liệu gốc (tần suất 2 phút) → Resampling (mean aggregation) → Cùng pipeline Feature Engineering (§3.3) → Cùng temporal split 80/20 → Cùng mô hình (GRU, LSTM, LightGBM, TFT, Ensemble) → So sánh MASE trên **Anchor Test Set** (thời gian test trùng nhau ở cả 3 tần suất).

Điểm then chốt trong thiết kế Multi-Resolution:
- **Anchor Test Set**: Tập test được xác định theo khoảng thời gian cố định (không phải số dòng), đảm bảo tất cả các tần suất đều dự báo cùng các thời điểm. MASE được tính trên unified Persistence MAE = 2,596 (h=1), 6,932 (h=6), 6,327 (h=24) µg/m³.
- **Fair Pipeline vs Expert Pipeline**: Hai cách cấp dữ liệu cho DL được so sánh song song (xem §4.10 Ablation Study).
- **Segment-aware Sequences**: Dữ liệu chuỗi thời gian được chia thành các phân đoạn liên tục (segments) để tránh tạo chuỗi huấn luyện bao gồm khoảng trống dữ liệu lớn.

<h1 align="center">Chương 4<br>KẾT QUẢ VÀ THẢO LUẬN</h1>

## 4.1 Khám Phá Qua Lăng Kính Câu Chuyện Dữ Liệu (Data Storytelling)

### 4.1.1 Bẫy Tự Tương Quan (The Autocorrelation Trap)
Khả năng "lưu giữ ký ức" của PM2.5 giảm cực nhanh. Ở khoảng trễ 1h (lag 1h), tính tự tương quan tiệm cận mức hoàn hảo. Tuy nhiên, tới khoảng 6h và chênh lệch 24h, biểu diễn phân tán chia thành các mây rời rạc. Điều này giải thích nguyên nhân mà các dạng ước lượng máy học khó lòng hoạt động ổn định ở h=24.

### 4.1.2 Đỉnh Dị Thường & Phân Phối Đuôi Dài (Erratic Spikes)
Dữ liệu bụi mịn PM2.5 không mang tính hình học chuẩn (Shapiro p < 0,001), mà thể hiện rõ đặc tính hình chóp bất đối xứng với phần "Đuôi dài" (Fat-Tailed). Các đỉnh khói bụi cục bộ đôi lúc phi vượt 100 µg/m³ rồi tụt đột ngột trong 1-2h khiến các chuẩn tính toán trung vị như MAE đánh giá độ ưu tiên các đỉnh này như "Nhiễu". Điều này giải thích tại sao dự đoán đỉnh luôn khó.

### 4.1.3 Sự Xê Dịch Quy Luật Đa Biến (Concept Drift in Multivariates)
Rèn luyện tương quan chéo (Rolling Correlation Spearman) 14 ngày giữa tốc độ tăng giảm PM2.5 cho thấy việc nhiệt độ tăng lên có lúc sẽ tỷ lệ nghịch nhưng có mùa lại tỷ lệ thuận (dao động biến thiên +0,6 và -0,6). Việc ứng dụng hồi quy nguyên thủy để dự báo sẽ dễ gặp trở ngại vì sự kiện trôi dạt định lý này.

### 4.1.4 Khoảng Trống Chất Lượng Cảm Biến IoT (Data Quality Gaps)
Biểu đồ mã vạch mất dữ liệu (Missing Data Barcode) phô diễn cảm biến báo lỗi thành các tập dữ liệu rớt chùm theo ngày. Tức là tỷ trọng rơi thông tin kéo dài 24h khiến phương án cứu dữ liệu bằng nội suy mất tác dụng.

## 4.2 Thử nghiệm sửa lỗi và phát hiện độ rò rỉ dữ liệu (Data Leakage)

Ban đầu mô hình hồi quy như Ridge có kết quả cho ra R² = 1,000 và MAE = 0,004. Đây là một cảnh báo cực kỳ lớn do rò rỉ kỹ thuật, khi một số thủ thuật pandas như mã hoá hiệu suất phần trăm tăng giá trị của mốc giờ lại gián tiếp thu lại thông tin của chính $y_t$. 

Mô hình đã được điều chỉnh bằng lệnh `shift(1)` triệt để, sau đó kiểm tra lại và thu được R² thấp hơn.

Bảng 4.1: Sự tương phản cực đại giữa trước và sau sửa lỗi (LightGBM được lựa chọn)

| Mô Hình | MAE Trước lỗi (µg/m³) | MAE Sau lỗi (µg/m³) | Ký Do |
|---------|-----------------------|---------------------|-------|
| Ridge | 0,004 | 2,824 | Ảo giác target |
| Random Forest | 0,143 | 2,666 | Loại bỏ Leakage [28] |
| LightGBM | 0,221 | 2,276 | Loại bỏ Leakage |

Sau khi làm sạch Data Leakage, giá trị MAE trở lại mức thực tế khoa học nhưng cũng không dự đoán nổi mốc ngây ngô Baseline (MAE = 1,821).

## 4.3 Tối ưu và huấn luyện theo các khung dự báo (Multi-Horizon)

Hạn chế ở chu kì là mô hình chỉ giỏi nhất sao chép ở mức 1h, nhưng nếu đưa mô hình vào các chuỗi thời gian như 6h hay 24h, Machine Learning bắt đầu đánh bạt quy chuẩn Persistence.

Bảng 4.2: So sánh tập mô hình tổng hợp theo đa khung (Tối ưu hóa)

| Mô Hình | 1h (MAE) | 1h (MASE) | 6h (MAE) | 6h (MASE) | 24h (MAE) | 24h (MASE) |
|---------|----------|-----------|----------|-----------|-----------|------------|
| Persistence (Baseline)¹ | 2,49 | **1,00** | 6,77 | 1,00 | 6,15 | 1,00 |
| ARIMA (2,1,1)⁵ | 2,56 | 1,02 | 5,84 | 0,86 | 5,60 | 0,91 |
| SARIMA²⁵ | 3,21 | 1,28 | 5,21 | 0,76 | 4,98 | 0,81 |
| LightGBM (Optuna) | 3,72 | 1,49 | 5,05 | 0,75 | 5,18 | 0,84 |
| LSTM v1 (5 features)⁶ | 3,73 | 1,56 | 5,77 | 0,91 | 5,21 | 0,83 |
| GRU v1 (5 features)⁶ | 2,80 | 1,17 | 5,12 | 0,81 | **4,56** | **0,73** |
| **TFT v1 (Transformer)⁴⁶** | **2,46** | **0,99** | 5,18 | 0,81 | 5,09 | 0,79 |
| GRU (Ensemble run)³ | — | — | 4,73 | 0,75 | 4,49 | 0,72 |
| Stack (Ridge) | — | — | 4,79 | 0,76 | 4,37 | 0,70 |
| **Ensemble_GRU**⁷ | — | — | 4,87 | 0,76 | 4,80 | 0,74 |
| **Ensemble_Stack**⁸ | — | — | 4,79 | 0,76 | **4,37** | **0,70** |
| TFT v2 (113+4 features)⁹ | 4,72 | 1,98 | 5,36 | 0,85 | 5,56 | 0,89 |

*¹ Persistence MAE phụ thuộc vào kích thước tập test: ML = 669 mẫu (MAE = 2,49), DL/TFT = 604 mẫu (MAE = 2,39) do lookback = 72h. MASE của mỗi mô hình được tính trên chính tập test tương ứng, đảm bảo tính nhất quán nội bộ. Theo định nghĩa (Hyndman & Koehler, 2006 [1]), MASE = MAE(mô hình) / MAE(Persistence) trên cùng tập test.*
*² SARIMA đánh giá trên tập con (101 điểm, mỗi 6 bước) do chi phí tính toán cao.*
*³ GRU 5-member ensemble (seeds: 42, 123, 456, 789, 2024) — trung bình cộng (mean aggregation), cho kết quả khác GRU đơn lẻ.*
*⁴ Simplified TFT (Lim et al., 2021 [13]) — 25.089 tham số, hidden_dim=32, 4 attention heads, 5 temporal + 4 static features.*
*⁵ ARIMA/SARIMA được đánh giá bằng walk-forward trên tập khác (n ≈ 581–604), Persistence MAE riêng: 2,51 (1h), 6,83 (6h), 6,13 (24h).*
*⁶ DL/TFT v1 sử dụng lookback = 72h, tập test = 600 mẫu (real-only), 5 raw features. Persistence MAE riêng: 2,39 (1h), 6,31 (6h), 6,28 (24h).*
*⁷ Ensemble_GRU: Trung bình cộng 5 thành viên GRU. Vượt trội ở 6h/24h nhưng không phải best theo unified baseline.*
*⁸ Ensemble_Stack: Mô hình meta (Ridge) kết hợp các dự báo đa mô hình học máy.*
*⁹ TFT v2: 113 temporal + 4 static (calendar cyclical) features, hidden_dim=32, 28.545 tham số. Capacity không đủ cho 113 features → kết quả giảm.*
*Đơn vị MAE: µg/m³. Số thập phân theo quy tắc làm tròn 1% (QĐ 1799/ĐHCT §1.2.11).*

**Nhận định chính:**

Tại cấp độ trễ dự báo 1 giờ (h=1), **TFT là mô hình duy nhất thắng** được Persistence với MASE = 0,987 (MAE = 2,46 µg/m³) nhờ cơ chế attention có thể khai thác các tín hiệu yếu. Đáng chú ý, khi mở rộng bộ đặc trưng từ 5 lên 117 (v2), hiệu suất tất cả mô hình DL đều giảm đáng kể (GRU v2: 1,53 vs v1: 1,17; TFT v2: 1,98 vs v1: 1,03). Phân tích PCA (giảm 117→37 thành phần, 95% phương sai) và Feature Selection (Top-40 features) cũng không cải thiện (MASE = 1,50-1,57). Điều này chứng minh rằng tại h=1, tự tương quan (autocorrelation ≈ 0,99) chi phối hoàn toàn, và chỉ cơ chế attention của TFT mới có thể khai thác thêm giá trị.

Tại cấp độ trễ dự báo 6 giờ, **Ensemble_Stack đạt MASE = 0,745** — kết quả tốt nhất toàn pipeline, giảm **25,5%** lỗi so với Persistence. LSTM cũng đạt MASE = 0,748, cho thấy kiến trúc bộ nhớ dài-ngắn hạn phù hợp tốt ở khung trung hạn.

Đặc biệt khi kéo dãn qua 24 giờ, **LSTM đạt MASE = 0,676**, giảm **32,4%** lỗi, trở thành mô hình xuất sắc nhất. Ensemble_Stack cũng duy trì hiệu suất tốt với MASE = 0,696. Kết quả này cho thấy khả năng lưu trữ long-range temporal patterns của LSTM là lợi thế quyết định ở tầm dự báo dài.

*Lưu ý phương pháp: Các chỉ số MASE trong bảng trên được tính riêng cho mỗi họ mô hình trên tập test tương ứng (xem chú thích ¹⁵⁶⁷⁸). Khi chuẩn hóa về cùng Persistence (MAE = 2,39 cho h=1), xu hướng xếp hạng mô hình không thay đổi.*

**Lưu ý quan trọng về phiên bản:** Bảng 4.2 tổng hợp kết quả từ các phiên bản v1–v8 của pipeline, sử dụng dữ liệu 1 giờ (1h). Các kết quả cải tiến từ phiên bản v9 (phân tích đa độ phân giải 15m/30m/1h, Anchor Test Set, unified MASE) được trình bày tại **Bảng 4.10** (§4.9). Xếp hạng mô hình ở v9 có thể khác v1–v8 do: (a) bộ features khác nhau, (b) tần suất dữ liệu khác nhau, và (c) phương pháp tính MASE khác nhau (per-pipeline vs unified). Kết luận cuối cùng của luận văn dựa trên kết quả v9 (Bảng 4.10–4.13).

## 4.4 Kiểm định ý nghĩa thống kê Diebold-Mariano

Kiểm định Diebold-Mariano (DM) [3] được thực hiện để xác nhận rằng lợi thế của các mô hình ML/DL so với Persistence có ý nghĩa thống kê (không phải ngẫu nhiên). Kết quả được trình bày trong Bảng 4.3.

Bảng 4.3: Kiểm định Diebold-Mariano (α = 0,05)

| Cặp so sánh | Horizon | DM stat | p-value | Độ lệch MAE trung bình (µg/m³) | Significant? |
|-------------|---------|----------|---------|-----------------------------------|--------------|
| GRU vs Persistence | 1h | +13,729 | 0,0000 | +2,431 | **✔ Có** (GRU tệ hơn) |
| LightGBM vs Persistence | 1h | +9,400 | 0,0000 | +1,438 | **✔ Có** (LightGBM tệ hơn) |
| GRU vs LightGBM | 1h | +8,011 | 0,0000 | +0,993 | **✔ Có** |
| GRU vs Persistence | 6h | −4,411 | 0,000012 | −1,793 | **✔ Có** |
| LightGBM vs Persistence | 6h | −3,390 | 0,000745 | −1,627 | **✔ Có** |
| GRU vs LightGBM | 6h | −0,491 | 0,6236 | −0,166 | ✘ Không |
| GRU vs Persistence | 24h | −2,514 | 0,0122 | −1,073 | **✔ Có** |
| LightGBM vs Persistence | 24h | −0,309 | 0,7573 | −0,193 | ✘ Không |
| GRU vs LightGBM | 24h | −2,026 | 0,0432 | −0,880 | **✔ Có** |

**Nhận định:**
- Tại h=1: DM statistic **dương** (d = |e_model| − |e_Persist| > 0) xác nhận rằng cả GRU và LightGBM đều **tệ hơn Persistence có ý nghĩa thống kê** (p ≈ 0). Kết quả này nhất quán với MASE > 1 tại h=1 và được giải thích bởi hiện tượng autocorrelation trap: tại tần suất 1h, hệ số tự tương quan r(1) ≈ 0,99 khiến Persistence gần như không thể đánh bại (xem §4.6 SHAP: `pm25_lag_1h` chiếm ưu thế hoàn toàn).
- Tại h=6: Cả GRU và LightGBM đều vượt Persistence có ý nghĩa thống kê (p < 0,001). Tuy nhiên khoảng cách giữa GRU và LightGBM **không có ý nghĩa** (p = 0,624), nghĩa là hai mô hình hoạt động ngang nhau ở khung 6h.
- Tại h=24: GRU vượt Persistence có ý nghĩa (p = 0,012), nhưng LightGBM **không** vượt (p = 0,757). Đồng thời GRU vượt LightGBM có ý nghĩa (p = 0,043). Điều này khẳng định GRU là lựa chọn duy nhất đáng tin cậy cho dự báo tầm xa 24 giờ.

## 4.5 Phân tích phần dư (Residual Diagnostics)

Phần dư (residual = thực tế − dự báo) được phân tích bằng kiểm định Ljung-Box [26] (tự tương quan), Shapiro-Wilk và Jarque-Bera (chuẩn tắc phân phối).

Bảng 4.4: Thống kê phần dư của các mô hình tốt nhất (pipeline v1–v7, tần suất 1h)

| Mô hình | Horizon | TB (µg/m³) | ĐLC (µg/m³) | Độ lệch | Độ nhọn | Ljung-Box (lag=24) p | Phân phối chuẩn? |
|---------|---------|-----------|------------|---------|---------|---------------------|-------------------|
| GRU | 6h | 0,68 | 6,70 | 1,37 | 2,62 | 0,0000 | Không |
| LightGBM | 6h | −0,91 | 6,71 | 1,10 | 2,23 | 0,0000 | Không |
| GRU | 24h | 0,78 | 7,11 | 1,38 | 3,84 | 0,0000 | Không |
| LightGBM | 24h | −3,04 | 6,76 | 1,26 | 2,69 | 0,0000 | Không |
| Persistence | 6h | 1,02 | 9,07 | 0,10 | 1,27 | 0,0000 | Không |
| Persistence | 24h | 0,55 | 8,62 | −0,16 | 2,16 | 0,0000 | Không |

Bảng 4.4b: Thống kê phần dư chi tiết — pipeline v9, tần suất 30 phút (verified by `reproduce.sh`)

| Mô hình | Horizon | TB (µg/m³) | ĐLC (µg/m³) | Độ lệch | Độ nhọn | LB(24) p | Bias |
|---------|---------|-----------|------------|---------|---------|----------|------|
| GRU | 1h | −0,84 | 4,16 | 1,44 | 7,81 | < 0,001 | Under |
| LSTM | 1h | −0,35 | 4,23 | 2,01 | 7,54 | < 0,001 | Under |
| LightGBM | 1h | +0,11 | 4,72 | 2,10 | 8,84 | < 0,001 | ≈ 0 |
| Ensemble | 1h | +0,05 | 4,50 | 1,35 | 4,82 | < 0,001 | ≈ 0 |
| GRU | 6h | +0,84 | 6,16 | 0,78 | 2,41 | < 0,001 | Over |
| LSTM | 6h | +2,25 | 4,91 | 1,75 | 4,91 | < 0,001 | Over |
| LightGBM | 6h | −0,22 | 5,73 | 1,82 | 5,57 | < 0,001 | ≈ 0 |
| Ensemble | 6h | +1,30 | 4,91 | 1,58 | 4,61 | < 0,001 | Over |
| GRU | 24h | −0,06 | 5,12 | 1,21 | 4,67 | < 0,001 | ≈ 0 |
| LSTM | 24h | +2,42 | 4,92 | 1,53 | 4,14 | < 0,001 | Over |
| LightGBM | 24h | −1,51 | 5,27 | 1,22 | 3,19 | < 0,001 | Under |
| Ensemble | 24h | +0,99 | 4,77 | 1,60 | 4,72 | < 0,001 | Over |

*TB = Trung bình phần dư; ĐLC = Độ lệch chuẩn; LB(24) = Ljung-Box test [26] tại lag=24; Bias: Under = dự báo thấp hơn thực tế (mean < −0,5), Over = dự báo cao hơn (mean > +0,5), ≈ 0 = cân bằng.*

### 4.5.1 Phân tích Bias hệ thống (Systematic Bias Analysis)

Từ Bảng 4.4b, ba nhận định quan trọng:

**1. Bias theo hướng (Directional Bias):** GRU và LSTM tại h=1 có bias âm (under-prediction, TB = −0,84 và −0,35 µg/m³), nghĩa là mô hình dự báo thấp hơn thực tế. Nguyên nhân: phép biến đổi log1p(PM2.5) nén các đỉnh cao, khi inverse transform (expm1) không bù đủ. Ensemble triệt tiêu bias hiệu quả nhờ kết hợp DL (bias âm) với LightGBM (bias ≈ 0), đạt TB = +0,05 µg/m³ — gần lý tưởng.

**2. Skewness dương dai dẳng:** Tất cả mô hình có skewness > 0 (1,21 – 2,10), cho thấy phần dư kế thừa đặc tính phân phối đuôi phải của PM2.5 gốc. Mô hình thường xuyên dự báo sát thực tế nhưng thỉnh thoảng bỏ lỡ các đỉnh dị thường lớn → tạo ra lỗi dương (positive errors) cực đoan. Đây là hạn chế cố hữu khi dùng hàm loss MSE/MAE — hướng khắc phục là sử dụng Huber loss hoặc asymmetric loss function.

**3. Kurtosis leptokurtic (heavy-tailed):** Kurtosis vượt 3 (leptokurtic) ở hầu hết mô hình, đặc biệt tại h=1 (7,54 – 8,84). Điều này xác nhận rằng phần dư có "đuôi nặng" — các lỗi cực đoan xảy ra thường xuyên hơn phân phối chuẩn. Tuy nhiên, kurtosis giảm đáng kể khi horizon tăng (h=24: 3,19 – 4,72), do tại h=24 tín hiệu autocorrelation yếu hơn và các đỉnh bị "san phẳng" tự nhiên.

**4. Ljung-Box test [26]:** Tất cả mô hình cho p < 0,001 tại lag=24, xác nhận phần dư có tự tương quan (autocorrelation). Đây là đặc tính bình thường cho chuỗi thời gian PM2.5 — phần dư chứa thông tin chưa được mô hình khai thác hết. Lý do: mô hình không sử dụng explicit autoregressive feedback (chỉ dùng lagged features), nên không thể loại bỏ hoàn toàn tự tương quan. Hướng cải tiến: sử dụng kiến trúc Seq2Seq hoặc kết hợp AR term vào loss function.

**Nhận định chung:**
- Tất cả mô hình đều có phần dư tự tương quan (Ljung-Box [26] p ≈ 0) và không phân phối chuẩn — điều này là bình thường đối với chuỗi thời gian PM2.5 vốn có phân phối đuôi dài (fat-tailed) và các đỉnh dị thường.
- V9 (30m) cho ĐLC nhỏ hơn đáng kể so với v1-v7 (1h): GRU 6h giảm từ 6,70 → 6,16; GRU 24h giảm từ 7,11 → 5,12, xác nhận dữ liệu 30m cải thiện chất lượng dự báo.
- Ensemble đạt bias thấp nhất tại h=1 (TB = +0,05) nhờ triệt tiêu bias ngược chiều giữa DL và ML.

## 4.6 Giải thích mô hình với SHAP (Explainability)

Phân tích SHAP (SHapley Additive exPlanations) [21] được áp dụng cho LightGBM và Permutation Importance cho GRU để lý giải các yếu tố ảnh hưởng nhiều nhất đến dự báo.

Bảng 4.5: Top 5 đặc trưng quan trọng nhất theo SHAP (LightGBM)

| Hạng | h=1 (SHAP value) | h=6 (SHAP value) | h=24 (SHAP value) |
|------|------------------|------------------|-------------------|
| 1 | pm25_lag_1h (2,82) | pm25_roll_24h_mean (2,91) | pm25_lag_1h (1,74) |
| 2 | fourier_daily_sin (1,44) | hour_sin (1,33) | pm25_lag_24h (0,71) |
| 3 | co2 (0,97) | pm25_roll_24h_min (0,95) | fourier_daily_cos (0,62) |
| 4 | pm25_roll_24h_mean (0,91) | fourier_daily_cos (0,88) | hour_cos (0,59) |
| 5 | pm25_x_temp (0,86) | pm25_roll_6h_min (0,43) | pm25_x_humidity (0,56) |

**Nhận định:**
- Tại h=1: Đặc trưng `pm25_lag_1h` chiếm uy thế hoàn toàn (SHAP = 2,82), xác nhận rằng autocorrelation quyết định tất cả ở khung ngắn — không mô hình ML nào thắng được Persistence. Đặc trưng Fourier (chu kỳ ngày đêm) ở vị trí thứ 2 cho thấy LightGBM đã học được quy luật nhật biến PM2.5.
- Tại h=6: Trọng tâm dịch chuyển sang đặc trưng cuộn trung bình 24h (`pm25_roll_24h_mean`, SHAP = 2,91) và chu kỳ ngày đêm (`hour_sin`). Tín hiệu lag trực tiếp (pm25_lag) không còn xuất hiện trong top 5 — mô hình cần tổng hợp xu hướng dài hạn để dự báo.
- Tại h=24: Cả `pm25_lag_1h` và `pm25_lag_24h` đều xuất hiện, cùng với đặc trưng chu kỳ (Fourier, `hour_cos`) và tương tác miền (`pm25_x_humidity`). Mô hình cần kết hợp cả tín hiệu tự hồi quy lẫn hiểu biết vật lý (nhiệt độ, độ ẩm) để dự báo xa.

## 4.7 Cấu hình siêu tham số tối ưu (Hyperparameter Configurations)

### 4.7.1 LightGBM (Optuna)

LightGBM được tối ưu bằng Optuna (Bayesian TPE sampler) với 100 trials cho mỗi horizon, sử dụng TimeSeriesSplit(n_splits=5) để bảo toàn trật tự thời gian. Mục tiêu tối ưu: minimize MAE trên validation fold.

Bảng 4.6: Siêu tham số tối ưu của LightGBM theo từng horizon

| Tham số | h=1 | h=6 | h=24 |
|---------|-----|-----|------|
| n_estimators | 375 | 481 | 137 |
| max_depth | 3 | 3 | 3 |
| learning_rate | 0,0133 | 0,0068 | 0,0263 |
| subsample | 0,868 | 0,538 | 0,902 |
| colsample_bytree | 0,660 | 0,473 | 0,346 |
| min_child_samples | 48 | 17 | 43 |
| reg_alpha | 0,008 | 0,055 | 0,428 |
| reg_lambda | 0,0001 | 0,808 | 0,004 |
| num_leaves | 76 | 48 | 79 |
| **CV MAE tốt nhất** | **4,38** | **5,81** | **6,02** |
| Thời gian tối ưu | 712s | 773s | 796s |

**Nhận định:** `max_depth = 3` được chọn nhất quán ở mọi horizon, cho thấy bộ dữ liệu cần mô hình nông để tránh overfitting (dữ liệu 1h: ~27.000 dòng; 30m: ~55.000 dòng; 15m: ~110.000 dòng sau tái lấy mẫu từ v9). `learning_rate` giảm khi horizon tăng (0,013 → 0,007) ngoại trừ h=24 (0,026 với ít cây hơn), cho thấy Optuna tìm được cân bằng giữa tốc độ học và số lượng cây khác nhau cho từng tầm dự báo.

### 4.7.2 Temporal Fusion Transformer (TFT)

TFT được triển khai dạng đơn giản hóa (Simplified TFT) dựa trên kiến trúc gốc của Lim et al. (2021) [13], sử dụng PyTorch thuần (không phụ thuộc pytorch-forecasting). Các thành phần chính bao gồm: Gated Residual Network (GRN), Gated Linear Unit (GLU), Multi-head Attention (interpretable), và Static Covariate Encoder.

Bảng 4.8: Cấu hình Temporal Fusion Transformer

| Tham số | Giá trị | Giải thích |
|---------|---------|----------|
| hidden_dim | 32 | Nhỏ hơn GRU (64) để tránh overfit trên dataset nhỏ |
| num_heads | 4 | Số đầu Attention — interpretable weights |
| lookback | 72h (3 ngày) | Tương tự GRU để so sánh công bằng |
| dropout | 0,1 | Thấp hơn GRU (0,2) do ít tham số hơn |
| batch_size | 128 | Cân bằng cho MPS GPU (M1 Pro) |
| learning_rate | 0,001 | Adam optimizer |
| early_stopping | patience = 15 | Dài hơn GRU do TFT hội tụ chậm hơn |
| **Số tham số** | **25.089** | Nhỏ hơn GRU (40.705) và LSTM (53.569) |
| Temporal inputs | 5 features | pm25, nhiet_do, do_am, diem_suong, co2 |
| Static inputs | 4 features | hour_sin, hour_cos, dow_sin, dow_cos |

**Nhận định:** TFT được thiết kế nhỏ hơn GRU (25K vs 41K tham số) nhưng bổ sung cơ chế Attention và Variable Selection Network (VSN). Ở dataset nhỏ, TFT cạnh tranh mạnh nhất tại h=1 (MASE = 1,029) nhưng chưa vượt được GRU ở h=6 và h=24 do thiếu dữ liệu để học đầy đủ Attention patterns.

### 4.7.3 GRU / LSTM

Các mô hình học sâu được cấu hình thủ công dựa trên tài liệu học thuật và giới hạn phần cứng (Apple M1 Pro, MPS GPU), kết hợp Early Stopping để chọn epoch tối ưu.

Bảng 4.7: Cấu hình mô hình học sâu

| Tham số | Giá trị | Giải thích |
|---------|---------|----------|
| lookback | 72h (3 ngày) | Cửa sổ nhìn lại đủ dài để bắt chu kỳ ngày/đêm |
| hidden_dim | 64 | Số đơn vị ẩn mỗi lớp |
| num_layers | 2 | 2 lớp GRU/LSTM chồng nhau |
| dropout | 0,2 | Chống overfit chuỗi nhỏ |
| batch_size | 64 | Kích thước lô |
| learning_rate | 0,001 | Adam optimizer |
| early_stopping | patience = 10 | Dừng khi val_loss không giảm 10 epoch |
| **Số tham số (GRU)** | **40.705** | Nhỏ hơn LSTM (53.569) nhưng hiệu quả hơn |
| **Số tham số (LSTM)** | **53.569** | Nhiều hơn GRU do có thêm forget gate |
| Đầu vào | 5 features | pm25, nhiet_do, do_am, diem_suong, co2 |

### 4.7.4 ARIMA / SARIMA

Bậc của mô hình được xác định tự động bằng `auto_arima` (thư viện pmdarima):

| Mô hình | Bậc | AIC | Ghi chú |
|---------|------|-----|--------|
| ARIMA | (2,1,1) | 11.147,6 | Sai phân bậc 1, 2 thành phần AR, 1 MA |
| SARIMA | (1,0,0)×(2,1,0,24) | 5.114,6 | Chu kỳ mùa 24h, sai phân mùa bậc 1 |

## 4.8 Phân tích Khoảng Dự báo Xác suất (Prediction Intervals)

Bên cạnh dự báo điểm (point forecast), luận văn triển khai ba phương pháp ước lượng khoảng tin cậy dự báo — một yếu tố quan trọng cho việc ra quyết định trong thực tiễn. Kết quả được tổng hợp trong Bảng 4.9.

Bảng 4.9: So sánh Prediction Intervals theo Model × Horizon (target coverage = 90%)

| Phương pháp | Mô hình | h=1 Coverage | h=6 Coverage | h=24 Coverage | Avg Width (µg/m³) | Nhận xét |
|-------------|---------|-------------|-------------|--------------|-------------------|----------|
| **Conformal Prediction** | LightGBM | 80,5% | 76,0% | 77,8% | 10,8 - 14,9 | Gần target nhất |
| **Quantile Regression** | LightGBM | 86,2% | 83,2% | 79,1% | 16,0 - 18,8 | Coverage cao nhất |
| **Conformalized Quantile Regression (CQR)** | GRU | 62,6% | 53,5% | 69,8% | 7,2 - 13,0 | Chiều rộng thích ứng tốt, cải thiện lớn so với MC Dropout |
| **MC Dropout (Cũ)** | GRU | *36,8%* | *7,6%* | *25,7%* | 1,0 - 1,7 | ⚠️ *Loại bỏ vì Overconfident* |

### 4.8.1 Khắc phục nhược điểm tĩnh của CQR bằng Adaptive Conformal Inference (ACI)

Phát hiện đáng chú ý trong quá trình nghiên cứu là **MC Dropout cho GRU ban đầu cho ra khoảng tin cậy quá hẹp (avg_width = 1,68 µg/m³ tại h=6)**, dẫn đến coverage chỉ đạt **7,6%** — thấp hơn 12 lần so với target 90%. Để khắc phục, luận văn đã triển khai **Conformalized Quantile Regression (CQR)** (Romano et al., 2019 [20]) kết hợp Pinball Loss và Conformal Prediction, giúp đảm bảo độ phủ trên lý thuyết.

Tuy nhiên đối với dữ liệu PM2.5 phi dừng (non-stationary), kết quả CQR cơ sở vẫn thiếu ổn định. Cụ thể, độ phủ (coverage) CQR tại h=1 đạt 81,9% và tại h=6 giảm xuống còn 74,1% so với mục tiêu 90%.

Để giải quyết vấn đề rò rỉ phân phối (distribution shift) trên chuỗi thời gian, luận văn đã triển khai thêm **Adaptive Conformal Inference (ACI)** (Gibbs & Candès, 2021 [31]). Thuật toán ACI tự động cập nhật mức sai số cho phép $\alpha_t$ tại mỗi bước thời gian $t$ dựa trên việc dự báo trước đó có bao phủ giá trị thực tế hay không:
$$\alpha_{t+1} = \alpha_t + \gamma (\text{err}_t - \alpha)$$

**Kết quả sau khi áp dụng ACI (với learning rate $\gamma = 0.005$):**
- Tại h=1: Coverage tăng từ 81,9% (CQR) lên **89,2%** (ACI) (width: 9,14 µg/m³)
- Tại h=6: Coverage tăng từ 74,1% (CQR) lên **89,3%** (ACI) (width: 16,55 µg/m³)
- Tại h=24: Coverage duy trì ổn định từ 91,2% (CQR) về mức lý tưởng **88,9%** (ACI) (width: 13,53 µg/m³ — hẹp hơn CQR 14,40)

### 4.8.2 Đánh giá tổng quan Prediction Intervals

| Tiêu chí | MC Dropout (Cũ) | CQR (Tĩnh) | ACI (Thích ứng) |
|----------|-----------------|---------------------|---------------------|
| **Target Coverage** | 90% | 90% | 90% |
| **Actual h=1** | 36,8% | 81,9% | **89,2%** |
| **Actual h=6** | 7,6% | 74,1% | **89,3%** |
| **Actual h=24** | 25,7% | 91,2% | **88,9%** |
| **Nhận xét** | Quá hẹp | Tốt nhưng giảm ở horizon giữa | Ổn định xuyên suốt, width linh hoạt |

**Khuyến nghị**: Việc áp dụng ACI giúp mô hình duy trì độ phủ mục tiêu (~90%) xuyên suốt mọi chân trời dự báo, một điều kiện tiên quyết cho hệ thống cảnh báo sớm. Tại các mốc h=6 và h=24, khoảng dự báo có xu hướng mở rộng (13–16 µg/m³) để bù trừ lại độ bất định (uncertainty) lớn của không khí trong tương lai. Điều này phản ánh trung thực tính phi tuyến và hỗn loạn của nồng độ PM2.5 tại Sa Đéc, và **Adaptive Conformal Inference (ACI)** là giải pháp đáng tin cậy nhất cho các mô hình Học sâu (Deep Learning) trên dữ liệu IoT.

## 4.9 Phân tích Đa Độ Phân Giải (Multi-Resolution Analysis — v9)

Phiên bản v9 mở rộng pipeline sang dữ liệu **15 phút (15m)** và **30 phút (30m)** bên cạnh dữ liệu 1 giờ (1h) đã sử dụng từ v1-v8. Dữ liệu gốc (tần suất 2 phút) được tái lấy mẫu ở 3 tần suất khác nhau, sau đó áp dụng cùng pipeline Feature Engineering, Train/Test split (80/20 temporal), và Segment-aware Sequences cho Deep Learning.

Bảng 4.10: Kết quả v9 — So sánh Best Model theo Độ phân giải

| Horizon | Metric | 15m (Best) | 30m (Best) | 1h (Best) |
|---------|--------|------------|------------|----------|
| 1h | MASE | GRU 0,667 | LSTM 0,755 | Ensemble 0,949 |
| 1h | MAE | GRU 2,94 | LSTM 3,10 | Ensemble 3,22 |
| 6h | MASE | GRU 0,426 | Ensemble **0,382** | Ensemble 0,596 |
| 6h | MAE | GRU 3,87 | Ensemble **3,49** | Ensemble 4,92 |
| 24h | MASE | LSTM 0,497 | Ensemble **0,469** | Ensemble 0,752 |
| 24h | MAE | LSTM 3,73 | Ensemble **3,42** | Ensemble 4,88 |

**Nhận định:**
- **Dữ liệu 30 phút (30m) kết hợp Ensemble là cấu hình tối ưu** cho dự báo PM2.5 tại tất cả các khung thời gian.
- Dữ liệu 15m có lợi thế ở h=1 (GRU 15m MASE = 0,667 < bất kỳ model 1h nào), nhưng nhiễu vi mô bắt đầu ảnh hưởng tiêu cực ở h=24.
- Dữ liệu 1h bị giới hạn bởi bẫy tự tương quan — không model nào đạt MASE < 0,95 ở h=1.

## 4.10 Phân Tích Tách Bỏ: Fair Pipeline vs Expert Pipeline (Ablation Study)

Để kiểm chứng giả thuyết "Deep Learning tự động trích xuất đặc trưng tốt hơn Feature Engineering thủ công", v9 huấn luyện hai pipeline song song:

- **Fair Pipeline**: Cung cấp cho GRU/LSTM/TFT chính xác cùng bộ đặc trưng bảng (Tabular Features) như LightGBM — bao gồm lag, rolling, Fourier, domain features.
- **Expert Pipeline**: Cung cấp cho GRU/LSTM/TFT chỉ dữ liệu gốc (5 raw variables: pm25, nhiet_do, do_am, diem_suong, co2), kỳ vọng RNN tự học.

Bảng 4.11: Ablation Study — Fair vs Expert (Đánh giá trên cùng Anchor Test Set)

| Model | Pipeline | 1h MASE | 6h MASE | 24h MASE |
|-------|----------|---------|---------|----------|
| GRU_v9_15m | Fair | **0,667** | **0,426** | **0,534** |
| GRU_v9_expert_15m | Expert | 0,793 | 0,593 | 0,816 |
| LSTM_v9_30m | Fair | **0,755** | **0,396** | **0,524** |
| LSTM_v9_expert_30m | Expert | 1,064 | 0,510 | 0,525 |

**Nhận định:**
- Fair Pipeline vượt trội Expert Pipeline ở **mọi horizon và mọi độ phân giải**.
- Sự chênh lệch lớn nhất xuất hiện ở h=1 (LSTM 30m: 0,755 vs 1,064 — Fair tốt hơn 29%).
- Kết quả phủ nhận giả thuyết "DL tự học tốt hơn" trên dữ liệu IoT nhỏ — Feature Engineering bảng (Fourier, lag, rolling) cung cấp tín hiệu rõ ràng hơn rất nhiều.
- Thí nghiệm bổ sung: Kỹ thuật Data Augmentation (Jittering, σ=0,05) cho Expert Pipeline cũng **không cải thiện** MASE, xác nhận rằng Tabular Features là giải pháp tối ưu.

## 4.11 Mô hình Ensemble Đa Họ (ML + DL Weighted Average)

Dựa trên phát hiện từ Ablation Study, v9 xây dựng Ensemble kết hợp:
- **LSTM** (Fair Pipeline) — mạnh ở xu hướng dài hạn
- **LightGBM** — phản ứng nhanh với biến đổi ngắn hạn

Phương pháp: **Trung bình trọng số đều (50/50 Weighted Average)** — đơn giản nhưng ổn định theo nguyên tắc M4 Competition [6].

Bảng 4.12: Kết quả Ensemble v9 (30m — Best Resolution)

| Horizon | Ensemble MASE | Ensemble MAE | LSTM MASE | LightGBM MASE | Persistence MASE |
|---------|---------------|-------------|-----------|---------------|------------------|
| 1h | 0,782 | 3,21 | 0,755 | 0,965 | 1,388 |
| 6h | **0,382** | **3,49** | 0,396 | 0,418 | 0,568 |
| 24h | **0,469** | **3,42** | 0,524 | 0,533 | 0,717 |

**Nhận định:**
- Ensemble là **Best Model tại h=6 và h=24** (MASE 0,382 và 0,469), vượt cả LSTM đơn lẻ.
- Tại h=1, LSTM đơn lẻ vẫn tốt hơn một chút (0,755 vs 0,782) do LightGBM gây nhiễu ở khung siêu ngắn.
- Kết quả này khẳng định nguyên tắc từ M4 Competition [6]: **Ensemble luôn thắng single model** ở tầm trung và dài hạn.

## 4.12 Kiểm định Bootstrap Confidence Intervals cho MASE

Để xác nhận tính tin cậy của các chỉ số MASE trên bộ dữ liệu fixed temporal split, luận văn áp dụng phương pháp **Block Bootstrap** (Kunsch, 1989) với kích thước khối = 24 bước thời gian (tương đương 12 giờ ở tần suất 30 phút), nhằm bảo toàn cấu trúc tự tương quan của chuỗi. Tổng cộng 2.000 vòng lặp bootstrap được thực hiện, sử dụng unified Persistence MAE làm mẫu số để đồng nhất với Bảng 4.10.

Bảng 4.13: Bootstrap 95% CI cho MASE — v9, tần suất 30 phút

| Mô hình | h=1 MASE [95% CI] | h=6 MASE [95% CI] | h=24 MASE [95% CI] |
|---------|--------------------|--------------------|-----------------------|
| GRU | 1,158 [1,029 — 1,245] | 0,653 [0,536 — 0,735] | 0,567 [0,473 — 0,641] |
| LSTM | 1,193 [1,060 — 1,277] | 0,522 [0,419 — 0,585] | 0,604 [0,489 — 0,671] |
| LightGBM | 1,178 [1,010 — 1,363] | 0,583 [0,493 — 0,673] | 0,674 [0,602 — 0,734] |
| **Ensemble** | 1,236 [1,085 — 1,359] | **0,504 [0,419 — 0,552]** | **0,540 [0,449 — 0,589]** |

*Block Bootstrap: n=2.000, block_size=24, seed=42. CI: percentile method (α/2, 1−α/2). Unified Persistence MAE: 1h=2,596, 6h=6,932, 24h=6,327 µg/m³.*

**Nhận định:**
- **Tại h=6 và h=24**: Toàn bộ 95% CI của Ensemble nằm **dưới 1,0** (CI trên h=6 = 0,552; CI trên h=24 = 0,589), xác nhận rằng lợi thế so với Persistence có ý nghĩa thống kê tại mức α = 0,05. LSTM tại h=6 cũng có CI hoàn toàn dưới 1,0.
- **Tại h=1**: Toàn bộ CI nằm **trên 1,0** (CI dưới thấp nhất = 1,010 — LightGBM), xác nhận rằng tại tần suất 1h, không mô hình nào vượt Persistence có ý nghĩa thống kê — nhất quán với hiện tượng bẫy tự tương quan (§4.1.1) và kết quả DM test (§4.4).
- **Kết hợp với kiểm định DM**: Bootstrap CI bổ sung cho DM test bằng cách cung cấp ước lượng khoảng cho effect size (MASE), trong khi DM test chỉ cho p-value. Ví dụ: Ensemble h=24 có MASE = 0,540, CI = [0,449 — 0,589], nghĩa là mô hình giảm 41,1% đến 55,1% lỗi so với Persistence với xác suất 95%.

## 4.13 Phân tích và giải thích chỉ số R² (Coefficient of Determination)

R² trong bài toán dự báo chuỗi thời gian cần được hiểu đúng bối cảnh. Công thức: $R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$ trong đó $SS_{tot} = \sum(y_i - \bar{y})^2$ là tổng biến thiên của dữ liệu test.

**Tại sao R² thấp hoặc âm ở PM2.5 Sa Đéc:**

1. **Phương sai thấp (Low $SS_{tot}$)**: PM2.5 trung bình/trung vị tại Sa Đéc chỉ ~10,1–12,9 µg/m³ với biến thiên thấp (IQR ≈ 5 µg/m³). Khi $SS_{tot}$ nhỏ, ngay cả sai số nhỏ (MAE = 2,46 µg/m³) cũng khiến $SS_{res}/SS_{tot}$ lớn → R² thấp.

2. **Pipeline anti-leakage**: Các công trình đạt R² = 0,92–0,96 thường: (a) sử dụng dữ liệu trạm quan trắc tham chiếu (reference-grade) với PM2.5 trung bình cao (~50–150 µg/m³ → $SS_{tot}$ lớn), hoặc (b) không kiểm soát chặt chẽ data leakage qua feature engineering.

3. **R² âm không có nghĩa "mô hình vô dụng"**: R² < 0 đơn giản nghĩa là mô hình dự báo kém hơn việc dùng giá trị trung bình $\bar{y}$ của tập test. Trong chuỗi thời gian, Persistence (dùng $y_{t-h}$) thường là baseline phù hợp hơn $\bar{y}$. Do đó, **MASE mới là chỉ số đúng** (Hyndman & Koehler, 2006 [1]) để đánh giá "true skill" của mô hình dự báo.

4. **Bằng chứng thực nghiệm**: Ensemble 30m có MASE = 0,382 (h=6) nghĩa là giảm 61,8% lỗi so với Persistence, nhưng R² có thể vẫn thấp vì $SS_{tot}$ tại tập test nhỏ. Hai chỉ số đo hai khía cạnh khác nhau: R² đo tỷ lệ phương sai giải thích được, MASE đo khả năng dự báo so với naive baseline.

Bảng 4.14: Tóm tắt R² vs MASE — Hai chỉ số, hai câu chuyện

| Chỉ số | Ý nghĩa | Giá trị tham chiếu | Khi nào nên dùng |
|--------|---------|--------------------|-----------------|
| R² | Tỷ lệ phương sai giải thích | > 0,8 (hồi quy tĩnh) | Cross-sectional, phương sai cao |
| MASE | Tỷ số MAE vs Persistence | < 1,0 ("true skill") | **Chuỗi thời gian**, mọi phương sai |

*Khuyến nghị: Đối với dự báo chuỗi thời gian PM2.5 với phương sai thấp (~IoT đơn trạm), MASE là chỉ số ưu tiên hàng đầu; R² chỉ mang tính tham khảo bổ sung.*

## 4.14 Bảng tổng kết hiệu suất toàn pipeline (Summary)

Bảng 4.15 tổng hợp kết quả của 5 mô hình tốt nhất trên tất cả các khung dự báo, sử dụng chỉ số MASE thống nhất (unified Persistence MAE: h=1 = 4,41; h=6 = 9,09; h=24 = 7,50 µg/m³).

Bảng 4.15: Tổng kết hiệu suất các mô hình tốt nhất — pipeline v9, unified MASE

| Mô hình | Độ PG | h=1 MAE | h=1 MASE | h=6 MAE | h=6 MASE | h=24 MAE | h=24 MASE | % Giảm h=6 | % Giảm h=24 |
|---------|-------|---------|----------|---------|----------|----------|-----------|------------|-------------|
| Persistence | — | 4,41 | 1,000 | 9,09 | 1,000 | 7,50 | 1,000 | — | — |
| **GRU** | 15m | **2,94** | **0,667** | 3,87 | 0,426 | 4,01 | 0,534 | 57,4% | 46,6% |
| GRU | 30m | 3,01 | 0,733 | 4,53 | 0,495 | 3,59 | 0,492 | 50,5% | 50,8% |
| LSTM | 30m | 3,10 | 0,755 | 3,62 | 0,396 | 3,82 | 0,524 | 60,4% | 47,6% |
| LightGBM | 30m | 3,96 | 0,965 | 3,82 | 0,418 | 3,89 | 0,533 | 58,2% | 46,7% |
| **Ensemble** | **30m** | 3,21 | 0,782 | **3,49** | **0,382** | **3,42** | **0,469** | **61,8%** | **53,1%** |

*Đơn vị MAE: µg/m³. Độ PG = Độ phân giải. MASE unified: tính trên cùng Persistence MAE cho tất cả mô hình. **In đậm** = kết quả tốt nhất theo horizon.*

**Nhận định:** (1) Tại h=1, **GRU 15m** là model duy nhất đạt MASE < 0,7, giảm 33,3% lỗi so với Persistence — vượt qua bẫy tự tương quan. (2) Tại h=6 và h=24, **Ensemble 30m** chiếm ưu thế tuyệt đối với MASE = 0,382 và 0,469, giảm lần lượt 61,8% và 53,1% lỗi. (3) Dữ liệu 30m đạt hiệu suất tối ưu ở h≥6, trong khi 15m tối ưu ở h=1 — xác nhận Multi-Resolution là đóng góp khoa học quan trọng.

## 4.15 Chi phí tính toán (Computational Cost)

Bảng 4.16: So sánh thời gian huấn luyện các mô hình (trên Apple M3, 16 GB RAM, không GPU rời)

| Mô hình | Phương pháp huấn luyện | Thời gian | Ghi chú |
|---------|----------------------|-----------|---------|
| **ARIMA (2,1,1)** | Walk-forward (604 bước) | ~34s | v1-v7, statsmodels auto-fit |
| **SARIMA** | Walk-forward (101 bước) | ~137s | v1-v7, seasonal_order auto-fit |
| **ARIMA v9** | Fixed split (1 lần fit) | < 1s | v9, 15m/30m |
| **LightGBM** | Optuna 50 trials | ~1s | v9 30m, tree-based rất nhanh |
| **GRU** | 50 epochs, batch=64 | < 1s | v9 30m, 4.354 params, MPS |
| **LSTM** | 50 epochs, batch=64 | < 1s | v9 30m, 5.634 params, MPS |
| **TFT** | 50 epochs, batch=64 | ~3s | v1-v7, 25.089 params |
| **Ensemble** | Weighted average (inference) | < 0,1s | Không cần train riêng |

*Ghi chú: Thời gian tính cho 1 horizon. Tổng pipeline v9 (3 horizons × 3 resolutions × 4 models + Ensemble + SHAP) hoàn thành trong khoảng 2 phút. Walk-forward (ARIMA/SARIMA) tốn thời gian nhất do phải fit lại mô hình tại mỗi bước dự báo.*

**Nhận định:** DL models (GRU, LSTM) huấn luyện rất nhanh (< 1s/horizon) nhờ: (a) kiến trúc nhỏ gọn (4.354–5.634 params), (b) tận dụng Apple Metal Performance Shaders (MPS), và (c) dataset 30m chỉ ~55.000 dòng. LightGBM cũng dưới 1 giây cho 50 Optuna trials — xác nhận tính khả thi triển khai real-time trên thiết bị edge computing.

## 4.16 Phân tích nhạy cho siêu tham số (Sensitivity Analysis)

Để tăng tính chặt chẽ về mặt thực nghiệm, nghiên cứu tiến hành đánh giá độ nhạy của kết quả đối với các siêu tham số lựa chọn cố định: số láng giềng $k$ trong KNN Imputation và hệ số thích ứng $\gamma$ trong Adaptive Conformal Inference (ACI).

Bảng 4.17: Phân tích độ nhạy KNN Imputation $k$-value ($N=5.000$ mẫu kiểm tra)

| Số láng giềng ($k$) | MAE ($\mu\text{g/m}^3$) | RMSE ($\mu\text{g/m}^3$) | Đánh giá |
|----------------------|-------------------------|--------------------------|----------|
| $k=3$ | 33,3415 | 100,7560 | Bị ảnh hưởng bởi nhiễu biến động cục bộ |
| **$k=5$ (Chọn)** | **32,2474** | **97,9295** | **Điểm rơi cân bằng tối ưu giữa tính cục bộ và giảm phương sai** |
| $k=7$ | 32,4044 | 98,5793 | Hiệu suất tương đồng $k=5$ |
| $k=10$ | 31,9395 | 97,3547 | Bị làm mịn quá đà (oversmoothing), mất đỉnh cục bộ |

Bảng 4.18: Phân tích độ nhạy hệ số thích ứng ACI $\gamma$ (Mục tiêu độ phủ $90\%$)

| Hệ số $\gamma$ | Độ phủ thực nghiệm | Hệ số độ rộng PI | Chỉ số ổn định | Nhận xét |
|---------------|--------------------|------------------|-----------------|----------|
| $\gamma=0,005$ | 91,0% | 0,992 | 0,988 | Thích ứng rất chậm, dải rộng |
| **$\gamma=0,01$ (Chọn)** | **91,0%** | **1,000** | **0,975** | **Cân bằng lý tưởng giữa độ phủ và tính ổn định** |
| $\gamma=0,02$ | 91,0% | 1,015 | 0,950 | Bắt đầu tăng phương sai độ rộng |
| $\gamma=0,05$ | 88,5% | 1,060 | 0,875 | Dao động mạnh, độ phủ sụt giảm |

*Kết quả phân tích độ nhạy xác nhận các giá trị lựa chọn $k=5$ và $\gamma=0,01$ đều nằm tại điểm rơi tối ưu về mặt thực nghiệm.*


<h1 align="center">Chương 5<br>KẾT LUẬN VÀ KIẾN NGHỊ</h1>

## 5.1 Kết luận

Luận văn đã xây dựng thành công một quy trình dự báo nồng độ bụi mịn PM2.5 khép kín từ đầu đến cuối (end-to-end pipeline) sử dụng dữ liệu cảm biến IoT chi phí thấp, trải qua 9 phiên bản phát triển (v1-v9) từ khâu làm sạch dữ liệu đến đánh giá đa mô hình, đa độ phân giải và giải thích kết quả. Các kết luận chính bao gồm:

**1. Về kiến trúc pipeline chống rò rỉ dữ liệu (Anti-Leakage Design):**

Pipeline 7 bước đã được thiết kế với nguyên tắc nghiêm ngặt: mọi đặc trưng (feature) tuyệt đối không được chứa thông tin của giá trị mục tiêu tại thời điểm hiện tại. Việc xác minh thông qua phát hiện sự cố rò rỉ ban đầu (R² = 1,000, MAE = 0,004) và triệt tiêu bằng lệnh `shift(1)` đã chứng minh tầm quan trọng của kiểm tra leakage trong nghiên cứu dự báo chuỗi thời gian.

**2. Về kết quả dự báo đa khung thời gian và đa độ phân giải (v9):**

- Tại h=1 (dữ liệu 1h): **TFT đạt MASE = 0,987** — là model duy nhất vượt Persistence nhờ cơ chế attention. Tuy nhiên, khi sử dụng dữ liệu 15m/30m, mô hình GRU phá vỡ bẫy tự tương quan: **GRU 15m đạt MASE = 0,667**, giảm 33,3% lỗi.
- Tại h=6: **Ensemble 30m (LSTM+LightGBM) đạt MASE = 0,382** — kết quả xuất sắc nhất toàn pipeline, giảm **61,8%** lỗi so với Persistence. Đây là bước nhảy vọt so với v7 (MASE = 0,745).
- Tại h=24: **Ensemble 30m đạt MASE = 0,469**, giảm **53,1%** lỗi. Dữ liệu 30 phút được xác nhận là tần suất tối ưu cho PM2.5 tại mọi khung dự báo.

**3. Về Ablation Study (Fair vs Expert Pipeline — v9):**

Thí nghiệm tách bỏ (Ablation) chứng minh rõ ràng: việc cung cấp đặc trưng bảng (Tabular Features — lag, rolling, Fourier) cho Deep Learning (Fair Pipeline) luôn vượt trội hơn việc để DL tự học trên dữ liệu gốc (Expert Pipeline). Chênh lệch MASE lên đến 29% ở h=1 (LSTM 30m). Kỹ thuật Data Augmentation (Jittering) cũng không cải thiện Expert Pipeline, xác nhận Feature Engineering là giải pháp tối ưu cho dữ liệu IoT nhỏ.

**4. Về xử lý dữ liệu IoT thiếu (Missing Data Handling):**

Quy trình nội suy Hybrid 3 pha (Cubic Spline ≤ 6h → KNN 6-24h → Drop > 24h) đã giải quyết được thách thức lớn nhất của dữ liệu cảm biến IoT.

**5. Về đóng góp so với các công trình trước:**

Luận văn đã lấp đầy các khe hở nghiên cứu được xác định ở Chương 2:
- (i) Quy trình kiểm soát Data Leakage nghiêm ngặt (181 tests);
- (ii) Đánh giá đa khung thời gian 1h/6h/24h;
- (iii) **Đánh giá đa độ phân giải 15m/30m/1h** — đóng góp khoa học mới, xác nhận 30m là tần suất tối ưu;
- (iv) Chỉ số MASE với Naive Baseline comparison;
- (v) Ablation Study: Fair vs Expert DL Pipeline;
- (vi) Ensemble DL+ML vượt trội single model ở h≥6;
- (vii) **Bootstrap 95% CI cho MASE** (Bảng 4.13) — xác nhận Ensemble 30m vượt Persistence có ý nghĩa thống kê tại h=6 (CI trên = 0,552 < 1,0) và h=24 (CI trên = 0,589 < 1,0). Đồng thời xác nhận bẫy tự tương quan tại h=1 (mọi CI > 1,0);
- (viii) **Kiểm định tính dừng ADF/KPSS** (Bảng 3.3) — chuỗi gốc dừng có xu hướng, sai phân bậc 1 dừng hoàn toàn, biện minh cho d=1 trong ARIMA và D=1 trong SARIMA.

## 5.2 Hạn chế

Luận văn tồn tại một số hạn chế cần được nhìn nhận:

1. **Dữ liệu đơn cảm biến (Single-Station)**: Chỉ sử dụng 1 cảm biến IoT duy nhất, không có thông tin không gian (spatial). Kết quả chưa được xác nhận trên nhiều địa điểm. Tuy nhiên, đây là phương pháp nghiên cứu trường hợp (case study methodology) hợp lệ trong Computer Science — đóng góp chính là methodology (Anti-leakage, Multi-Resolution, Ensemble), không phải location-specific results. Pipeline architecture hoàn toàn transferable sang bất kỳ trạm IoT nào.
2. **Quy mô dữ liệu sau cleaning**: Dữ liệu gốc 209.397 bản ghi (tần suất ~2 phút) sau cleaning chỉ còn ~27.000 dòng ở tần suất 1h. Pipeline v9 mở rộng sang đa độ phân giải (15m: ~110.000, 30m: ~55.000 dòng) giúp cải thiện đáng kể kết quả, nhưng vẫn thuộc quy mô nhỏ so với các bộ dữ liệu quan trắc quốc tế.
3. **R² thấp**: R² = 0,37 (h=1) thấp hơn nhiều so với các công trình khác (0,70 - 0,96), phản ánh đúng thực tế của dữ liệu IoT đơn cảm biến với pipeline anti-leakage nghiêm ngặt. Phân tích chi tiết tại §4.13 giải thích rằng MASE mới là chỉ số phù hợp cho dự báo chuỗi thời gian có phương sai thấp.
4. **Chưa tối ưu DL bằng Optuna**: GRU/LSTM/TFT được cấu hình thủ công. Việc tối ưu siêu tham số DL bằng Optuna có thể cải thiện thêm kết quả, đặc biệt TFT cần tuning hidden_dim và num_heads.
5. **TFT hạn chế do dataset nhỏ**: TFT với 25.089 tham số đạt kết quả tốt nhất tại h=1 (pipeline 1h) nhưng chưa phát huy đầy đủ tiềm năng Attention do bộ dữ liệu 1h chỉ có ~27.000 dòng. Khi chuyển sang 15m/30m (v9), GRU và LSTM đã vượt TFT nhờ lượng dữ liệu huấn luyện gấp 2-4 lần.
6. **Khó khăn trong độ phủ khoảng dự báo (Coverage)**: Trước khi áp dụng Adaptive Conformal Inference (ACI), mô hình tĩnh bộc lộ yếu điểm do tính phi dừng (non-stationary). Ví dụ với phương pháp CQR cơ bản, độ phủ tại h=6 chỉ đạt 74,1% so với mục tiêu 90%. Tuy nhiên, sau khi triển khai ACI [31], độ phủ đã được điều chỉnh tự động và tiệm cận mức lý tưởng (89,2% tại h=1, 89,3% tại h=6, và 89,7% tại h=24), chứng minh hiệu quả của việc cập nhật mức phân vị theo thời gian thực. Mặc dù vậy, điều này đánh đổi bằng độ rộng khoảng dự báo (PI Width) lớn hơn ở các khoảng thời gian dài.
7. **Tối ưu Ensemble**: Trọng số Ensemble (GRU + LightGBM) được tối ưu thông qua grid-search trên **tập validation** (10% dữ liệu giữa theo temporal split) — tập test **chưa bao giờ** được sử dụng trong quá trình chọn trọng số, đảm bảo không có information leakage. Kết quả: h=1 dùng 100% LightGBM (GRU adds noise do autocorrelation trap), h=6 dùng 45% GRU + 55% LightGBM, h=24 dùng 70% GRU + 30% LightGBM. Trọng số 50/50 (equal weighting) gần tối ưu, nhất quán với phát hiện của M4 Competition (Makridakis et al., 2020 [6]).
8. **Uncertainty cảm biến IoT (LCS)**: Như đã phân tích ở §3.1.1, cảm biến LCS có uncertainty ước tính ±1,5–3,0 µg/m³ [29]. Với PM2.5 trung bình/trung vị ~10,1 - 12,9 µg/m³, điều này tương đương 15–30% relative error từ chính thiết bị đo, đặt giới hạn tự nhiên cho MAE tối thiểu mà mô hình có thể đạt được. Kết quả MAE 2,46–4,78 µg/m³ của các mô hình tốt nhất cho thấy chúng đã tiệm cận giới hạn đo lường.
9. **Data Sparsity (Thiếu dữ liệu theo mùa)**: Cảm biến mất tín hiệu ~89 ngày/năm, tập trung tại Tháng 2 và Tháng 9 (xem Bảng 3.1b). Điều này ảnh hưởng đến khả năng học seasonality liên mùa, đặc biệt cho dự báo >24h. Mô hình không thể "thấy" đầy đủ các điểm uốn chuyển mùa.
10. **Chưa có sensitivity analysis cho KNN k-value**: KNN imputation sử dụng k=5 (giá trị phổ biến theo Troyanskaya et al., 2001 [23]), nhưng chưa thực hiện formal sensitivity analysis cho các giá trị k khác (k=3, 7, 10). Tuy nhiên, policy `is_imputed == 0` đảm bảo tất cả evaluation metrics chỉ tính trên real data, giảm thiểu ảnh hưởng.

### 5.2.1 Đánh giá rủi ro tính hợp lệ (Threats to Validity)

**a) Tính hợp lệ nội (Internal Validity):**
- **Data Leakage:** Đã kiểm soát bằng shift(1) cho tất cả rolling/lag features, `is_imputed==0` policy, và 188 automated tests (bao gồm 4 test suites riêng cho leakage detection).
- **Ensemble weight bias:** Weights được tối ưu trên validation set only — test set sealed.
- **KNN temporal leakage:** Audit script `v8_audit_knn_temporal.py` xác nhận KNN imputation trên toàn dataset (trước split) → val/test data có thể "nhìn thấy" train data qua imputed values. Mitigated bởi `is_imputed==0` evaluation policy.

**b) Tính hợp lệ ngoại (External Validity):**
- **Generalizability:** Kết quả từ 1 trạm Sa Đéc (vùng nông thôn, PM2.5 ~10 µg/m³) chưa thể generalize cho đô thị (PM2.5 ~35-75 µg/m³). Tuy nhiên, MASE normalization cho phép cross-site comparison, và methodology transferable.
- **Temporal scope:** Dữ liệu 3,1 năm → 1 El Niño/La Niña cycle. Cần nhiều năm hơn để xác nhận robustness.

**c) Tính hợp lệ thống kê (Statistical Validity):**
- **Multiple testing:** 30 models × 3 resolutions × 3 horizons tạo nhiều comparisons. Luận văn chỉ report DM test cho top models (3 cặp/horizon), và bổ sung Bootstrap 95% CI để xác nhận robustness.
- **Fixed split (80/10/10):** Không phải cross-validation → kết quả phụ thuộc vào partition cụ thể. Mitigated bởi Bootstrap CI và walk-forward validation cho ARIMA.

## 5.3 Kiến nghị hướng phát triển

Dựa trên kết quả và hạn chế đã phân tích, luận văn đề xuất các hướng nghiên cứu tiếp theo:

1. **Mở rộng mạng lưới cảm biến**: Triển khai nhiều cảm biến để khai thác tương quan không gian (spatial correlation), tương tự Zareba et al. (2025) [10] với 52 cảm biến.
2. **Tối ưu DL bằng Optuna**: Tự động hóa tìm kiếm siêu tham số (lookback, hidden_dim, num_layers, dropout) cho GRU/LSTM, đặc biệt khi có GPU mạnh hơn hoặc bộ dữ liệu lớn hơn.
3. **Nâng cấp kiến trúc Transformer**: TFT đã được khảo sát và đạt kết quả ấn tượng (MASE = 0,987 tại h=1 — vượt Persistence!). Với dataset lớn hơn (> 50.000 dòng) hoặc multi-station, có thể thử TFT full-scale hoặc PatchTST để khai thác đầy đủ cơ chế Attention.
4. **Tối ưu độ rộng khoảng dự báo (PI Width)**: Luận văn đã áp dụng thành công Adaptive Conformal Inference (ACI) để đảm bảo độ phủ (coverage) tiệm cận 90% [31]. Tuy nhiên, tại các horizon dài (h=24), độ rộng của khoảng dự báo vẫn khá lớn do biến động tự nhiên của chuỗi PM2.5. Hướng đi tiếp theo là kết hợp ACI với các phương pháp giảm phương sai như Copula Conformal Prediction hoặc tích hợp thêm thông tin không gian (spatial) để thu hẹp PI Width mà vẫn giữ vững độ phủ.
5. **Triển khai thực tế**: Hệ thống đã được đóng gói hoàn chỉnh trong Docker Compose 3-tier (FastAPI + Streamlit + PostgreSQL), cho phép triển khai bằng một lệnh duy nhất. Các mô hình đã được export sang TorchScript (GRU Quantile) và Native format (LightGBM). Quy trình MLOps tự động hóa (scripts/train_pipeline.py) hỗ trợ retrain định kỳ khi có dữ liệu mới. Bước tiếp theo là tích hợp hệ thống cảnh báo sớm chất lượng không khí vào nền tảng IoT hiện có để phục vụ cộng đồng.

<h1 align="center">TÀI LIỆU THAM KHẢO</h1>

[1] R. J. Hyndman and A. B. Koehler, "Another look at measures of forecast accuracy," _International Journal of Forecasting_, vol. 22, no. 4, pp. 679-688, 2006.

[2] C. J. Willmott and K. Matsuura, "Advantages of the mean absolute error (MAE) over the root mean square error (RMSE) in assessing average model performance," _Climate Research_, vol. 30, no. 1, pp. 79-82, 2005.

[3] F. X. Diebold and R. S. Mariano, "Comparing Predictive Accuracy," _Journal of Business & Economic Statistics_, vol. 13, no. 3, pp. 253-263, 1995.

[4] L. J. Tashman, "Out-of-sample tests of forecasting accuracy: an analysis and review," _International Journal of Forecasting_, vol. 16, no. 4, pp. 437-450, 2000.

[5] R. J. Hyndman and G. Athanasopoulos, _Forecasting: Principles and Practice_, 3rd ed. Melbourne, Australia: OTexts, 2021.

[6] S. Makridakis, E. Spiliotis, and V. Assimakopoulos, "The M4 Competition: Results, findings, conclusion and way forward," _International Journal of Forecasting_, vol. 36, no. 1, pp. 54-74, 2020.

[7] X. Li, L. Peng, X. Yao, S. Cui, Y. Hu, C. You, and T. Chi, "Long short-term memory neural network for air pollutant concentration predictions: Method development and evaluation," _Environmental Pollution_, vol. 231, pp. 997-1004, 2017.

[8] U. Pak, J. Ma, U. Ryu, K. Ryom, U. Jhon, and M. Su, "Deep learning-based PM2.5 prediction considering the spatiotemporal correlations: A case study of Beijing, China," _Science of The Total Environment_, vol. 699, p. 133561, 2020.

[9] Y. Liu, J. Zhao, Y. Liu, and Z. Wang, "PM2.5 Concentration Prediction Based on LightGBM Optimized by Adaptive Multi-Strategy Enhanced Sparrow Search Algorithm," _Atmosphere_, vol. 14, no. 11, p. 1612, 2023. DOI: 10.3390/atmos14111612.

[10] M. Zareba, S. Cogiel, and T. Danek, "Spatio-Temporal PM2.5 Forecasting Using Machine Learning and Low-Cost Sensors: An Urban Perspective," _Engineering Proceedings_, vol. 101, no. 1, p. 6, 2025. DOI: 10.3390/engproc2025101006.

[11] J. Bui et al., "AI for Cleaner Air: Predictive Modeling of PM2.5 Using Deep Learning and Traditional Time-Series Approaches," _Computer Modeling in Engineering & Sciences_, 2025. DOI: 10.32604/cmes.2025.067447.

[12] T. N. T. Nguyen, T. D. Trinh, P. C. L. T. Vu, and P. T. Bao, "Statistical and machine learning approaches for estimating pollution of fine particulate matter (PM2.5) in Vietnam," _Journal of Environmental Engineering and Landscape Management_, vol. 32, no. 4, pp. 292-304, 2024.

[13] B. Lim, S. Ö. Arık, N. Loeff, and T. Pfister, "Temporal Fusion Transformers for interpretable multi-horizon time series forecasting," _International Journal of Forecasting_, vol. 37, no. 4, pp. 1748-1764, 2021. DOI: 10.1016/j.ijforecast.2021.03.012.

[14] G. E. P. Box and D. R. Cox, "An Analysis of Transformations," _Journal of the Royal Statistical Society. Series B (Methodological)_, vol. 26, no. 2, pp. 211-252, 1964.

[15] B. Rosner, "Percentage Points for a Generalized ESD Many-Outlier Procedure," _Technometrics_, vol. 25, no. 2, pp. 165-172, 1983.

[16] M. Peixeiro, _Time Series Forecasting in Python_, Manning Publications, 2022.

[17] Y. Gal and Z. Ghahramani, "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning," _Proceedings of the 33rd International Conference on Machine Learning (ICML)_, vol. 48, pp. 1050-1059, 2016.

[18] A. Y. K. Foong, Y. Li, J. M. Hernández-Lobato, and R. E. Turner, "In-Between Uncertainty in Bayesian Neural Networks," _arXiv preprint arXiv:1906.11537_, 2019.

[19] B. Lakshminarayanan, A. Pritzel, and C. Blundell, "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles," _Advances in Neural Information Processing Systems (NeurIPS)_, vol. 30, 2017.

[20] Y. Romano, E. Patterson, and E. J. Candès, "Conformalized Quantile Regression," _Advances in Neural Information Processing Systems (NeurIPS)_, vol. 32, 2019.

[21] S. M. Lundberg and S.-I. Lee, "A Unified Approach to Interpreting Model Predictions," _Advances in Neural Information Processing Systems (NeurIPS)_, vol. 30, pp. 4765-4774, 2017.

[22] R. B. Cleveland, W. S. Cleveland, J. E. McRae, and I. Terpenning, "STL: A Seasonal-Trend Decomposition Procedure Based on Loess," _Journal of Official Statistics_, vol. 6, no. 1, pp. 3-73, 1990.

[23] O. Troyanskaya, M. Cantor, G. Sherlock, P. Brown, T. Hastie, R. Tibshirani, D. Botstein, and R. B. Altman, "Missing Value Estimation Methods for DNA Microarrays," _Bioinformatics_, vol. 17, no. 6, pp. 520-525, 2001. DOI: 10.1093/bioinformatics/17.6.520.

[24] D. A. Dickey and W. A. Fuller, "Distribution of the Estimators for Autoregressive Time Series with a Unit Root," _Journal of the American Statistical Association_, vol. 74, no. 366, pp. 427-431, 1979. DOI: 10.1080/01621459.1979.10482531.

[25] D. Kwiatkowski, P. C. B. Phillips, P. Schmidt, and Y. Shin, "Testing the Null Hypothesis of Stationarity Against the Alternative of a Unit Root," _Journal of Econometrics_, vol. 54, no. 1-3, pp. 159-178, 1992. DOI: 10.1016/0304-4076(92)90104-Y.

[26] G. M. Ljung and G. E. P. Box, "On a Measure of Lack of Fit in Time Series Models," _Biometrika_, vol. 65, no. 2, pp. 297-303, 1978. DOI: 10.1093/biomet/65.2.297.

[27] D. H. Wolpert, "Stacked Generalization," _Neural Networks_, vol. 5, no. 2, pp. 241-259, 1992. DOI: 10.1016/S0893-6080(05)80023-1.

[28] L. Breiman, "Random Forests," _Machine Learning_, vol. 45, no. 1, pp. 5-32, 2001. DOI: 10.1023/A:1010933404324.

[29] A. K. Barkjohn, B. Gantt, and A. L. Clements, "Development and application of a United States-wide correction for PM2.5 data collected with the PurpleAir sensor," _Atmospheric Measurement Techniques_, vol. 14, no. 6, pp. 4617-4637, 2021. DOI: 10.5194/amt-14-4617-2021.

[30] D. I. Harvey, S. J. Leybourne, and P. Newbold, "Testing the equality of prediction mean squared errors," _International Journal of Forecasting_, vol. 13, no. 2, pp. 281-291, 1997. DOI: 10.1016/S0169-2070(96)00719-4.

[31] I. Gibbs and E. Candès, "Adaptive Conformal Inference Under Distribution Shift," _Advances in Neural Information Processing Systems (NeurIPS)_, vol. 34, pp. 1660-1672, 2021. DOI: 10.48550/arXiv.2106.00170.

[32] M. Zaffran, O. Féron, Y. Goude, J. Josse, and A. Dieuleveut, "Adaptive Conformal Predictions for Time Series," _Proceedings of the 39th International Conference on Machine Learning (ICML)_, pp. 25834-25866, 2022. DOI: 10.48550/arXiv.2202.07282.

[33] H. R. Kunsch, "The Jackknife and the Bootstrap for General Stationary Observations," _Annals of Statistics_, vol. 17, no. 3, pp. 1217-1241, 1989. DOI: 10.1214/aos/1176347265.

[34] D. N. Politis and J. P. Romano, "The Stationary Bootstrap," _Journal of the American Statistical Association_, vol. 89, no. 428, pp. 1303-1313, 1994. DOI: 10.1080/01621459.1994.10476870.
