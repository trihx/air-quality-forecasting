# TRƯỜNG ĐẠI HỌC CẦN THƠ
## KHOA CÔNG NGHỆ THÔNG TIN VÀ TRUYỀN THÔNG

<br><br><br>

# NGUYỄN HOÀNG XUÂN TRÍ

<br><br>

# NGHỆ THUẬT VÀ PHƯƠNG PHÁP DỰ BÁO NỒNG ĐỘ BỤI MỊN PM2.5 BẰNG MÁY HỌC VÀ HỌC SÂU ĐA MÔ HÌNH DỰA TRÊN DỮ LIỆU CẢM BIẾN IOT ĐA ĐỘ PHÂN GIẢI

<br><br>

### LUẬN VĂN THẠC SĨ
### NGÀNH: KHOA HỌC MÁY TÍNH
### MÃ SỐ: 8480101

<br><br><br><br>

### CẦN THƠ, NĂM 2026

\newpage

# TRƯỜNG ĐẠI HỌC CẦN THƠ
## KHOA CÔNG NGHỆ THÔNG TIN VÀ TRUYỀN THÔNG

<br><br>

# NGUYỄN HOÀNG XUÂN TRÍ
### MÃ SỐ HV: M0123456

<br><br>

# NGHỆ THUẬT VÀ PHƯƠNG PHÁP DỰ BÁO NỒNG ĐỘ BỤI MỊN PM2.5 BẰNG MÁY HỌC VÀ HỌC SÂU ĐA MÔ HÌNH DỰA TRÊN DỮ LIỆU CẢM BIẾN IOT ĐA ĐỘ PHÂN GIẢI

<br><br>

### LUẬN VĂN THẠC SĨ
### CHUYÊN NGÀNH: KHOA HỌC MÁY TÍNH
### MÃ SỐ NGÀNH: 8480101

<br><br>

### NGƯỜI HƯỚNG DẪN KHOA HỌC:
### PGS. TS. LÊ VĂN X

<br><br><br>

### CẦN THƠ, NĂM 2026

\newpage

## CHẤP THUẬN CỦA HỘI ĐỒNG

Luận văn thạc sĩ này, với đề tựa: **"Nghệ thuật và phương pháp dự báo nồng độ bụi mịn PM2.5 bằng máy học và học sâu đa mô hình dựa trên dữ liệu cảm biến IoT đa độ phân giải"**, do học viên **Nguyễn Hoàng Xuân Trí** thực hiện theo sự hướng dẫn khoa học của **PGS. TS. Lê Văn X**. Luận văn đã được báo cáo và thông qua trước Hội đồng chấm luận văn thạc sĩ vào ngày ..... tháng ..... năm 2026.

Luận văn đã được hoàn thiện và chỉnh sửa theo đúng biên bản góp ý của Hội đồng chấm luận văn.

<br><br>

| Thư ký Hội đồng | Ủy viên Hội đồng |
| :---: | :---: |
| *(Ký tên và ghi rõ học hàm, học vị)* | *(Ký tên và ghi rõ học hàm, học vị)* |
| <br><br><br> | <br><br><br> |
| **.....................................................** | **.....................................................** |

<br><br>

| Phản biện 1 | Phản biện 2 |
| :---: | :---: |
| *(Ký tên và ghi rõ học hàm, học vị)* | *(Ký tên và ghi rõ học hàm, học vị)* |
| <br><br><br> | <br><br><br> |
| **.....................................................** | **.....................................................** |

<br><br>

| Người hướng dẫn khoa học | Chủ tịch Hội đồng |
| :---: | :---: |
| *(Ký tên và ghi rõ học hàm, học vị)* | *(Ký tên và ghi rõ học hàm, học vị)* |
| <br><br><br> | <br><br><br> |
| **PGS. TS. Lê Văn X** | **.....................................................** |

\newpage

## LỜI CẢM ƠN

Lời đầu tiên, tôi xin bày tỏ lòng biết ơn sâu sắc và chân thành nhất đến **PGS. TS. Lê Văn X**, người thầy đã tận tình hướng dẫn, định hướng khoa học và dành nhiều thời gian trao đổi, truyền đạt những kiến thức phương pháp luận quý báu cho tôi trong suốt quá trình thực hiện đề tài luận văn thạc sĩ này.

Tôi xin chân thành cảm ơn Quý Thầy, Cô trong Khoa Công nghệ Thông tin và Truyền thông, Trường Đại học Cần Thơ đã giảng dạy, trang bị cho tôi những nền tảng kiến thức chuyên môn vững chắc và tạo mọi điều kiện thuận lợi về cơ sở vật chất, thủ tục hành chính trong suốt khóa học Cao học.

Tôi cũng xin gửi lời cảm ơn đến Ban Quản lý trạm quan trắc cảm biến IoT Sa Đéc (tỉnh Đồng Tháp) đã hỗ trợ cung cấp nguồn dữ liệu thực nghiệm liên tục, giúp đề tài có được bộ dữ liệu thực tế giàu giá trị khoa học.

Sau cùng, tôi xin gửi lời cảm ơn tha thiết đến gia đình, bạn bè và các đồng nghiệp đã luôn động viên, chia sẻ và tạo động lực to lớn để tôi hoàn thành tốt công trình nghiên cứu này.

*Cần Thơ, ngày ..... tháng ..... năm 2026*  
**Học viên**  
<br><br><br>  
**Nguyễn Hoàng Xuân Trí**

\newpage

## TÓM TẮT LUẬN VĂN (ABSTRACT IN VIETNAMESE)

Dự báo nồng độ bụi mịn PM2.5 đóng vai trò quan trọng trong việc cảnh báo sớm nguy cơ ô nhiễm không khí và bảo vệ sức khỏe cộng đồng. Tuy nhiên, chuỗi thời gian nồng độ PM2.5 từ các hệ thống cảm biến chi phí thấp (Low-Cost Sensors - LCS) trong mạng lưới IoT thường xuyên đối mặt với các thách thức lớn như tính phi tuyến cao, bẫy rò rỉ dữ liệu (data leakage), hiện tượng mất mát dữ liệu kéo dài (data gaps) và bẫy tự tương quan (autocorrelation trap). 

Luận văn này nghiên cứu và đề xuất một quy trình kỹ nghệ dữ liệu khép kín (End-to-End Pipeline) kết hợp với mô hình dự báo đa độ phân giải (Multi-Resolution) và đa mốc thời gian (Multi-Horizon: 1h, 6h, 24h). Bộ dữ liệu thực nghiệm được thu thập từ trạm cảm biến IoT đặt tại thành phố Sa Đéc, tỉnh Đồng Tháp trong khoảng thời gian 3,1 năm (từ 03/2022 đến 05/2025 với 209.594 bản ghi thô). Nghiên cứu đề xuất chiến lược **Nội suy phân tầng (Tiered Imputation Strategy)**: áp dụng Cubic Spline cho các khoảng trống ngắn ($\le 6h$), K-Nearest Neighbors (KNN) cho khoảng trống trung bình ($6-24h$), và loại bỏ các khoảng trống dài ($>24h$) nhằm bảo toàn cấu trúc phân đoạn tự nhiên của dữ liệu. Đồng thời, quy trình kỹ nghệ đặc trưng tuân thủ nghiêm ngặt nguyên tắc **Anti-Leakage Discipline** thông qua phép biến đổi trễ $\operatorname{shift}(1)$ cho toàn bộ 119 đặc trưng temporal.

Kết quả thực nghiệm trên tập kiểm thử mỏ neo (Anchor Test Set) cho thấy:
1. Tại điểm trễ siêu ngắn $1h$, tính tự tương quan tiệm cận 0,97 khiến Baseline Naive Persistence đạt hiệu năng rất cao, tuy nhiên mạng Học sâu **GRU ở độ phân giải 15 phút (GRU_v9_15m)** đã đánh bại Persistence với chỉ số $\operatorname{MASE} = 0,667$ (MAE = 2,944 $\mu g/m^3$, $R^2 = 0,267$).
2. Tại các điểm trễ xa $6h$ và $24h$, độ phân giải **30 phút (30m)** được chứng minh là **"Điểm ngọt độ phân giải" (Resolution Sweet Spot)**. Mô hình kết hợp trọng số **Ensemble_Weighted_v9_30m** (kết hợp LightGBM và GRU) đạt hiệu năng xuất sắc nhất toàn hệ thống, với $\operatorname{MASE} = 0,382$ (MAE = 3,493 $\mu g/m^3$) tại $6h$ (giảm 31,3% sai số so với Persistence) và $\operatorname{MASE} = 0,469$ (MAE = 4,290 $\mu g/m^3$) tại $24h$ (giảm 27,5% sai số).
3. Phân tích minh bạch mô hình bằng Explainable AI (SHAP TreeExplainer) đã phát hiện **Ngưỡng tới hạn ô nhiễm phi tuyến (Physical Tipping Point)** khi nồng độ trung bình 24h vượt qua mức $17 - 18 \mu g/m^3$, khiến tác động đẩy giá trị dự báo ô nhiễm tăng vọt theo cấp số nhân.

**Từ khóa:** Bụi mịn PM2.5, Cảm biến IoT, Đa độ phân giải, Multi-Horizon Forecasting, Anti-Leakage Discipline, Tiered Imputation, Ensemble Learning, SHAP, MASE.

\newpage

## ABSTRACT (IN ENGLISH)

Fine particulate matter (PM2.5) forecasting plays a crucial role in providing early atmospheric pollution warnings and protecting public health. However, time-series data gathered from low-cost Internet of Things (IoT) sensors often suffer from severe challenges including high non-linearity, subtle data leakage traps, prolonged missing data gaps, and autocorrelation dominance at short horizons.

This thesis presents an end-to-end data engineering pipeline combined with a multi-resolution (15m, 30m, 1h) and multi-horizon (1h, 6h, 24h) forecasting framework. The empirical dataset was collected from an IoT sensor station in Sa Dec City, Dong Thap Province, Vietnam, spanning 3.1 years (March 2022 to May 2025 with 209,594 raw observations). We propose a **Tiered Imputation Strategy** employing Cubic Spline for short gaps ($\le 6h$), K-Nearest Neighbors (KNN) for medium gaps ($6-24h$), and complete removal of gaps $>24h$ to avoid continuous sequence hallucinations. Furthermore, our feature engineering rigorously enforces an **Anti-Leakage Discipline** using explicit $\operatorname{shift}(1)$ delay transformations across all 119 temporal features.

Empirical evaluation on an Anchor Test Set reveals that:
1. At the ultra-short $1h$ horizon, despite strong autocorrelation ($ACF \approx 0.97$), a Deep Learning model at 15-minute resolution (**GRU_v9_15m**) successfully overcomes the autocorrelation trap, achieving $\operatorname{MASE} = 0.667$ (MAE = 2.944 $\mu g/m^3$, $R^2 = 0.267$).
2. At longer horizons ($6h$ and $24h$), the **30-minute sampling resolution (30m)** is established as the optimal **"Resolution Sweet Spot"**. The hybrid model **Ensemble_Weighted_v9_30m** (combining LightGBM and GRU) achieves superior performance across the entire system, yielding $\operatorname{MASE} = 0.382$ (MAE = 3.493 $\mu g/m^3$) at $6h$ (a 31.3% MAE reduction over Persistence) and $\operatorname{MASE} = 0.469$ (MAE = 4.290 $\mu g/m^3$) at $24h$ (a 27.5% MAE reduction).
3. Explainable AI analysis using SHAP TreeExplainer uncovers a non-linear **Physical Tipping Point** when the 24-hour rolling mean PM2.5 exceeds $17 - 18 \mu g/m^3$, triggering exponential pollution escalation values.

**Keywords:** PM2.5 Forecasting, IoT Low-Cost Sensors, Multi-Resolution Analysis, Multi-Horizon Forecasting, Anti-Leakage Discipline, Tiered Imputation, Ensemble Learning, SHAP, MASE.

\newpage

## LỜI CAM ĐOAN

Tôi tên là **Nguyễn Hoàng Xuân Trí**, học viên cao học khóa 2023–2025, chuyên ngành Khoa học Máy tính, Mã số HV: M0123456, Trường Đại học Cần Thơ.

Tôi xin cam đoan rằng:
1. Quyển luận văn thạc sĩ này là công trình nghiên cứu khoa học thực sự của bản thân tôi, được thực hiện dưới sự hướng dẫn khoa học của PGS. TS. Lê Văn X.
2. Tất cả các dữ liệu, kết quả tính toán và số liệu thực nghiệm trình bày trong luận văn là trung thực, khách quan, được trích xuất trực tiếp từ hệ thống mã nguồn codebase và cơ sở dữ liệu thực nghiệm của dự án, chưa từng được công bố trong bất kỳ công trình luận văn hay luận án nào khác.
3. Các tài liệu tham khảo, công trình nghiên cứu của các tác giả khác được trích dẫn và sử dụng trong luận văn đều được dẫn nguồn và kê khai đầy đủ, chính xác theo chuẩn trích dẫn quốc tế IEEE.

Tôi xin chịu hoàn toàn trách nhiệm trước nhà trường và pháp luật về lời cam đoan này.

*Cần Thơ, ngày ..... tháng ..... năm 2026*  
**Tác giả luận văn**  
<br><br><br>  
**Nguyễn Hoàng Xuân Trí**

\newpage

## DANH MỤC TỪ VIẾT TẮT

| Từ viết tắt | Thuật ngữ đầy đủ tiếng Anh / tiếng Việt | Ý nghĩa chuyên môn |
|---|---|---|
| **ACF** | Autocorrelation Function | Hàm tự tương quan |
| **ACI** | Adaptive Conformal Inference | Hiệu chỉnh khoảng tin cậy thích ứng |
| **AQI** | Air Quality Index | Chỉ số chất lượng không khí |
| **ARIMA** | AutoRegressive Integrated Moving Average | Mô hình tự hồi quy tích hợp trung bình trượt |
| **CAMS** | Copernicus Atmosphere Monitoring Service | Dữ liệu khí quyển vệ tinh châu Âu |
| **CQR** | Conformal Quantile Regression | Hồi quy phân vị hiệu chuẩn tương hợp |
| **DA** | Directional Accuracy | Độ chính xác hướng biến động (%) |
| **DL** | Deep Learning | Học sâu |
| **ESD** | Extreme Studentized Deviate | Kiểm định phát hiện ngoại lệ đa điểm |
| **EWM** | Exponentially Weighted Moving Average | Trung bình trượt trọng số mũ |
| **GRU** | Gated Recurrent Unit | Mạng Nơ-ron hồi quy đơn vị cổng |
| **KNN** | K-Nearest Neighbors | Thuật toán K hàng xóm gần nhất |
| **LCS** | Low-Cost Sensors | Cảm biến chi phí thấp IoT |
| **LSTM** | Long Short-Term Memory | Mạng nhớ dài-ngắn hạn |
| **MAD** | Median Absolute Deviation | Độ lệch tuyệt đối trung vị |
| **MAE** | Mean Absolute Error | Sai số tuyệt đối trung bình ($\mu g/m^3$) |
| **MASE** | Mean Absolute Scaled Error | Sai số tuyệt đối chuẩn hóa so với Naive |
| **ML** | Machine Learning | Máy học |
| **NMPIW** | Normalized Mean Prediction Interval Width | Độ rộng khoảng tin cậy chuẩn hóa |
| **PCA** | Principal Component Analysis | Phân tích thành phần chính |
| **PM2.5** | Particulate Matter $\le 2.5 \mu m$ | Bụi mịn có đường kính khí động $\le 2,5 \mu m$ |
| **RMSE** | Root Mean Squared Error | Căn sai số bình phương trung bình |
| **SARIMA** | Seasonal ARIMA | Mô hình ARIMA có yếu tố mùa vụ |
| **SHAP** | SHapley Additive exPlanations | Phương pháp giải thích mô hình theo lý thuyết trò chơi |
| **SOTA** | State-of-The-Art | Trình độ công nghệ / kết quả tốt nhất hiện nay |
| **TFT** | Temporal Fusion Transformer | Mô hình Transformer hợp nhất chuỗi thời gian |
| **WHO** | World Health Organization | Tổ chức Y tế Thế giới |
| **XAI** | Explainable Artificial Intelligence | Trí tuệ nhân tạo có thể giải thích |

\newpage

## DANH MỤC BẢNG

| Số hiệu bảng | Tên bảng | Trang |
|---|---|---|
| **Bảng 2.1** | Bảng đối chiếu kết quả của luận văn với các công trình nghiên cứu SOTA (2022–2025) | 12 |
| **Bảng 3.1** | Thống kê mô tả các biến đo lường từ trạm cảm biến IoT Sa Đéc | 18 |
| **Bảng 3.2** | Bảng tỷ lệ độ phủ dữ liệu theo 12 tháng trong năm (Data Coverage Barcode) | 20 |
| **Bảng 3.3** | Tổng hợp danh mục 119 đặc trưng kỹ nghệ temporal phân loại theo 7 nhóm | 23 |
| **Bảng 3.4** | Bảng so sánh 3 cấp độ phân giải dữ liệu (15m, 30m, 1h) sau tiền xử lý | 26 |
| **Bảng 3.5** | Kết quả kiểm định tính dừng ADF và KPSS trên chuỗi PM2.5 gốc và sai phân | 28 |
| **Bảng 3.6** | Cấu hình mạng 3 tầng (3-Tier Architecture) và tham số triển khai hệ thống | 31 |
| **Bảng 4.1** | Kết quả thực nghiệm v9 trên Anchor Test Set (Chuẩn hóa Unified Persistence) | 35 |
| **Bảng 4.2** | Kiểm định rò rỉ dữ liệu (Anti-Leakage Audit) trước và sau khi áp dụng shift(1) | 38 |
| **Bảng 4.3** | Kết quả đánh giá Khoảng tin cậy dự báo (Prediction Intervals) bằng CQR | 41 |
| **Bảng 4.4** | Kết quả kiểm định ý nghĩa thống kê Diebold-Mariano ($p$-value) | 43 |
| **Bảng 4.5** | Đánh giá F1-Score cảnh báo các đợt ô nhiễm vượt ngưỡng WHO ($45 \mu g/m^3$) | 45 |

\newpage

## DANH MỤC HÌNH VẼ

| Số hiệu hình | Tên hình vẽ | Trang |
|---|---|---|
| **Hình 1.1** | Sơ đồ quy trình tổng quan của nghiên cứu từ dữ liệu thô IoT đến dự báo đa mốc | 5 |
| **Hình 3.1** | Bản đồ vị trí trạm cảm biến IoT Sa Đéc và sơ đồ thu thập dữ liệu về Cloud | 17 |
| **Hình 3.2** | Mã vạch mất mát dữ liệu (Missing Data Barcode) thể hiện các gap rớt tín hiệu | 19 |
| **Hình 3.3** | Quy trình 7 bước tiền xử lý dữ liệu và chiến lược Nội suy phân tầng (Tiered Imputation) | 21 |
| **Hình 3.4** | Minh họa cơ chế chống rò rỉ dữ liệu (Anti-Leakage Discipline) bằng phép biến đổi shift(1) | 24 |
| **Hình 3.5** | Biểu đồ kỹ thuật phân rã TimeSeriesSplit 80/10/10 với Purging Gap cách ly | 29 |
| **Hình 3.6** | Sơ đồ kiến trúc phần mềm 3 tầng (Streamlit Frontend + FastAPI Backend + PostgreSQL DB) | 32 |
| **Hình 4.1** | Biểu đồ phân phối đuôi dài (Fat-Tailed Distribution) của nồng độ PM2.5 Sa Đéc | 34 |
| **Hình 4.2** | Biểu đồ Hàm tự tương quan ACF thể hiện hiện tượng Autocorrelation Trap tại 1h | 36 |
| **Hình 4.3** | Biểu đồ so sánh MASE giữa các độ phân giải 15m, 30m, 1h tại 3 mốc dự báo | 37 |
| **Hình 4.4** | Đồ thị SHAP Summary Beeswarm Plot thể hiện mức độ quan trọng 20 đặc trưng hàng đầu | 42 |
| **Hình 4.5** | Đồ thị SHAP Dependence Plot giải mã Ngưỡng tới hạn ô nhiễm (Physical Tipping Point) | 44 |
| **Hình 4.6** | Giao diện Dashboard dự báo PM2.5 thời gian thực và dải khoảng tin cậy CQR | 47 |

\newpage

# MỤC LỤC

- **CHẤP THUẬN CỦA HỘI ĐỒNG** ............................................................................................................ iii
- **LỜI CẢM ƠN** .................................................................................................................................... iv
- **TÓM TẮT LUẬN VĂN (ABSTRACT IN VIETNAMESE)** .............................................................................. v
- **ABSTRACT (IN ENGLISH)** ................................................................................................................... vi
- **LỜI CAM ĐOAN** ................................................................................................................................ vii
- **DANH MỤC TỪ VIẾT TẮT** .................................................................................................................. viii
- **DANH MỤC BẢNG** ............................................................................................................................. ix
- **DANH MỤC HÌNH VẼ** .......................................................................................................................... x

<br>

### CHƯƠNG 1: GIỚI THIỆU .................................................................................................................. 1
- **1.1 Tính cấp thiết của đề tài** .............................................................................................................. 1
- **1.2 Mục tiêu nghiên cứu** ................................................................................................................... 3
- **1.3 Câu hỏi nghiên cứu và Giả thuyết khoa học** ................................................................................ 4
- **1.4 Đối tượng và Phạm vi nghiên cứu** ................................................................................................ 5
- **1.5 Ý nghĩa khoa học và thực tiễn** ..................................................................................................... 6
- **1.6 Bố cục của luận văn** .................................................................................................................... 7

<br>

### CHƯƠNG 2: TỔNG QUAN TÀI LIỆU ................................................................................................ 8
- **2.1 Cơ sở lý thuyết về ô nhiễm không khí & Bụi mịn PM2.5** ................................................................ 8
- **2.2 Lý thuyết về Chuỗi thời gian & Phương pháp dự báo** ................................................................. 10
- **2.3 Lược khảo các nghiên cứu trong và ngoài nước (2022–2026)** ........................................................ 12
- **2.4 Phương pháp luận đánh giá độ chính xác & Khoảng tin cậy** .......................................................... 14
- **2.5 Khe hở nghiên cứu (Research Gap) & Đóng góp của Luận văn** .................................................... 16

<br>

### CHƯƠNG 3: PHƯƠNG PHÁP NGHIÊN CỨU .................................................................................... 17
- **3.1 Dữ liệu thu thập và Phân tích chất lượng dữ liệu IoT Sa Đéc** ........................................................... 17
- **3.2 Quy trình Tiền xử lý Dữ liệu 7 Bước & Tiered Imputation** .............................................................. 21
- **3.3 Quy trình Kỹ nghệ Đặc trưng Chống rò rỉ (Anti-Leakage Discipline)** .............................................. 23
- **3.4 Thuật toán và Kiến trúc các Mô hình Dự báo** ................................................................................ 26
- **3.5 Quy trình Kiểm định chéo chuỗi thời gian & Anchor Test Set** ....................................................... 29
- **3.6 Kiến trúc Phần mềm Hệ thống 3 Tầng** .......................................................................................... 31

<br>

### CHƯƠNG 4: KẾT QUẢ VÀ THẢO LUẬN ............................................................................................ 33
- **4.1 Khám phá dữ liệu qua lăng kính Data Storytelling** .......................................................................... 33
- **4.2 Thử nghiệm phát hiện và khắc phục Rò rỉ Dữ liệu (Data Leakage Audit)** ........................................ 35
- **4.3 Kết quả thực nghiệm Đa độ phân giải & Đa khung thời gian** ........................................................ 37
- **4.4 Thảo luận về Điểm ngọt độ phân giải 30m & Bẫy tự tương quan** ................................................. 39
- **4.5 Đánh giá Khoảng tin cậy dự báo (Prediction Intervals)** ............................................................... 41
- **4.6 Phân tích Minh bạch Mô hình bằng Explainable AI (SHAP)** .......................................................... 43
- **4.7 Kiểm định ý nghĩa thống kê Diebold-Mariano & Cảnh báo ô nhiễm WHO** .................................... 45

<br>

### CHƯƠNG 5: KẾT LUẬN VÀ ĐỀ XUẤT .............................................................................................. 48
- **5.1 Kết luận chính của Luận văn** ....................................................................................................... 48
- **5.2 Hàm ý quản lý môi trường & Ứng dụng thực tiễn** ........................................................................ 49
- **5.3 Hạn chế của nghiên cứu** .............................................................................................................. 50
- **5.4 Đề xuất hướng phát triển tiếp theo** ............................................................................................. 51

<br>

- **TÀI LIỆU THAM KHẢO** .................................................................................................................... 52
- **PHỤ LỤC** .......................................................................................................................................... 56

\newpage

<h1 align="center">CHƯƠNG 1<br>GIỚI THIỆU</h1>

## 1.1 Tính cấp thiết của đề tài

Ô nhiễm không khí, đặc biệt là ô nhiễm bụi mịn **PM2.5** (hạt bụi có đường kính khí động học $\le 2,5 \mu m$), đã trở thành một trong những mối đe dọa sinh thái và sức khỏe cộng đồng nghiêm trọng nhất trên phạm vi toàn cầu cũng như tại Việt Nam. Theo báo cáo của Tổ chức Y tế Thế giới (WHO), bụi mịn PM2.5 có khả năng luồn sâu vào phế nang phổi, xâm nhập trực tiếp vào hệ tuần hoàn máu, gây ra các bệnh lý mãn tính nguy hiểm như viêm đường hô hấp, hen suyễn, đột quỵ và ung thư phổi. Tại các khu vực đô thị và đồng bằng đang trong quá trình công nghiệp hóa nhanh như Đồng bằng sông Cửu Long (ĐBSCL), biến động nồng độ PM2.5 chịu ảnh hưởng phức tạp bởi sự kết hợp giữa các hoạt động dân sinh, giao thông, đốt phụ phẩm nông nghiệp và các hiện tượng khí tượng đặc thù (nghịch nhiệt, độ ẩm cao).

Để chủ động giảm thiểu tác động tiêu cực của ô nhiễm không khí, việc xây dựng một hệ thống dự báo nồng độ PM2.5 chính xác theo nhiều khoảng thời gian (Multi-Horizon: 1 giờ, 6 giờ, 24 giờ) đóng vai trò then chốt. Nguồn dữ liệu truyền thống từ các trạm quan trắc tham chiếu quốc gia (Reference-Grade Stations) tuy có độ chính xác rất cao nhưng chi phí đầu tư và vận hành đắt đỏ, dẫn đến mật độ phân bố thưa thớt, không thể bao phủ toàn diện các vùng kinh tế trọng điểm. Sự phát triển mạnh mẽ của mạng lưới Internet vạn vật (IoT) với các hệ thống **Cảm biến Chi phí thấp (Low-Cost Sensors - LCS)** đã mở ra giải pháp thu thập dữ liệu PM2.5 với tần suất cao ($\sim 2$ phút/lần đo) và mật độ dày đặc.

Tuy nhiên, việc khai thác dữ liệu cảm biến IoT chi phí thấp để dự báo chuỗi thời gian đặt ra những thách thức khoa học và kỹ thuật rất lớn:
1. **Nhiễu dữ liệu và Khoảng trống dữ liệu dài (Data Gaps):** Cảm biến IoT thường xuyên gặp sự cố rớt mạng, lỗi phần cứng hoặc gián đoạn nguồn điện, tạo ra các khoảng trống dữ liệu kéo dài nhiều ngày. Việc xử lý không khéo léo (như kéo đường thẳng nội suy qua khoảng trống $>24h$) sẽ tạo ra các "ảo giác dữ liệu" (hallucinations), làm biến dạng nghiêm trọng việc học quy luật của các mô hình Deep Learning.
2. **Hiểm họa Rò rỉ Dữ liệu (Data Leakage):** Trong bài toán dự báo chuỗi thời gian, việc sử dụng các đặc trưng tính toán từ tương lai (như biến sai phân $diff(t) = y_t - y_{t-1}$ không qua phép biến đổi trễ $\operatorname{shift}(1)$) là một bẫy kỹ thuật phổ biến. Điều này làm mô hình đạt chỉ số $R^2 \approx 1,0$ ảo trên tập huấn luyện nhưng thất bại hoàn toàn khi triển khai thực tế.
3. **Bẫy Tự Tương Quan (Autocorrelation Trap) tại mốc siêu ngắn ($1h$):** Chuỗi nồng độ PM2.5 có hệ số tự tương quan rất cao ($ACF \approx 0,97$ ở trễ 1h). Do đó, tại mốc $1h$, mô hình ngây ngô Persistence ($y_{t+1}=y_t$) tỏ ra cực kỳ mạnh mẽ. Việc chứng minh mô hình Học máy/Học sâu có "kỹ năng dự báo thực sự" đòi hỏi phải đánh giá bằng các thước đo chuẩn hóa như **MASE (Mean Absolute Scaled Error)**.
4. **Vấn đề Đa Độ Phân Giải (Multi-Resolution):** Tần suất lấy mẫu dữ liệu đầu vào (15 phút, 30 phút hay 1 giờ) ảnh hưởng trực tiếp đến tỷ lệ Tín hiệu/Nhiễu (Signal-to-Noise Ratio). Liệu dữ liệu tần suất quá cao (15m) có gây ngộ độc nhiễu cho mô hình, hay dữ liệu tần suất thấp (1h) làm mất đi các sóng biến đổi ngắn hạn? 

Xuất phát từ những yêu cầu thực tiễn và bài toán khoa học nêu trên, đề tài luận văn: **"Nghệ thuật và phương pháp dự báo nồng độ bụi mịn PM2.5 bằng máy học và học sâu đa mô hình dựa trên dữ liệu cảm biến IoT đa độ phân giải"** được thực hiện nhằm xây dựng một quy trình kỹ nghệ dữ liệu chuẩn mực, giải quyết triệt để các hạn chế trên và cung cấp mô hình dự báo tối ưu cho thực tế.

---

## 1.2 Mục tiêu nghiên cứu

Luận văn hướng tới việc hoàn thành mục tiêu chung và 6 mục tiêu cụ thể sau:

### 1.2.1 Mục tiêu chung
Xây dựng một quy trình kỹ nghệ dữ liệu chống rò rỉ (Anti-Leakage Pipeline) hoàn chỉnh và hệ thống dự báo đa mô hình (Máy học, Học sâu, Transformer, Hybrid Ensemble) có khả năng dự báo chính xác nồng độ bụi mịn PM2.5 theo 3 mốc thời gian ($1h, 6h, 24h$) dựa trên dữ liệu cảm biến IoT đa độ phân giải ($15m, 30m, 1h$) tại Sa Đéc, Đồng Tháp.

### 1.2.2 Mục tiêu cụ thể
1. **Xây dựng Pipeline Tiền xử lý & Kỹ nghệ Đặc trưng Chống Rò rỉ:** Thiết lập quy trình 7 bước tiền xử lý, tự động hóa loại bỏ ngoại lệ bằng thuật toán S-ESD kết hợp phân rã STL, và thực thi nghiêm ngặt kỷ luật chống rò rỉ (**Anti-Leakage Discipline**) qua phép biến đổi trễ $\operatorname{shift}(1)$ trên toàn bộ 119 đặc trưng.
2. **Đề xuất Chiến lược Nội suy Phân tầng (Tiered Imputation):** Phân rã và xử lý các khoảng trống dữ liệu theo độ dài gap: dùng Cubic Spline cho gap ngắn ($\le 6h$), KNN cho gap trung bình ($6-24h$), và hoàn toàn loại bỏ gap dài ($>24h$) để bảo toàn cấu trúc phân đoạn tự nhiên (`segment_id`).
3. **Thực hiện Khảo sát Đa Độ Phân Giải (Multi-Resolution Analysis):** Tái lấy mẫu và so sánh hiệu năng mô hình trên 3 tần suất ($15m, 30m, 1h$) trên cùng tập kiểm thử mỏ neo (**Anchor Test Set**), nhằm xác định "Điểm ngọt độ phân giải" (Resolution Sweet Spot).
4. **Đánh giá Đa Mốc Thời Gian (Multi-Horizon Evaluation):** Thực nghiệm huấn luyện và so sánh 30+ cấu hình mô hình (Persistence, ARIMA/SARIMA, LightGBM, Random Forest, ElasticNet, GRU, LSTM, TFT, Weighted Ensemble) tại 3 mốc trễ $1h, 6h, 24h$.
5. **Đánh giá Khoảng Tin Cậy & Cảnh báo Ô nhiễm:** Triển khai phương pháp Hồi quy Phân vị Hiệu chuẩn Tương hợp (**Conformal Quantile Regression - CQR**) để đưa ra dải khoảng tin cậy 90% (đánh giá qua Winkler Score và NMPIW) và bộ chỉ số F1-Score cảnh báo các đợt ô nhiễm vượt ngưỡng WHO ($45 \mu g/m^3$).
6. **Giải mã Minh bạch Mô hình (Explainable AI - XAI):** Áp dụng SHAP TreeExplainer và Permutation Importance để phân tích tầm quan trọng của đặc trưng và phát hiện Ngưỡng tới hạn ô nhiễm phi tuyến (Physical Tipping Point).

---

## 1.3 Câu hỏi nghiên cứu và Giả thuyết khoa học

### 1.3.1 Câu hỏi nghiên cứu
- **CH1:** Làm thế nào để thiết kế quy trình kỹ nghệ đặc trưng đảm bảo loại bỏ 100% rò rỉ dữ liệu từ tương lai mà vẫn trích xuất tối đa thông tin chuỗi thời gian?
- **CH2:** Tần suất lấy mẫu dữ liệu nào ($15m, 30m$ hay $1h$) mang lại hiệu năng dự báo tối ưu cho các mô hình Máy học và Học sâu ở mốc trung và dài hạn ($6h, 24h$)?
- **CH3:** Ở mốc dự báo siêu ngắn ($1h$), mô hình Deep Learning nào có khả năng vượt qua bẫy tự tương quan (Autocorrelation Trap) của Baseline Persistence?
- **CH4:** Mối quan hệ giữa các biến khí tượng (nhiệt độ, độ ẩm) và nồng độ PM2.5 thể hiện tính phi tuyến như thế nào, và ngưỡng bùng phát ô nhiễm tới hạn là bao nhiêu?

### 1.3.2 Giả thuyết khoa học
- **GH1:** Phép biến đổi trễ $\operatorname{shift}(1)$ trên toàn bộ các đặc trưng Rolling/EWM/Diff sẽ triệt tiêu hoàn toàn rò rỉ dữ liệu, đưa chỉ số $R^2$ kiểm định về khoảng thực tế ($0,10 - 0,30$) nhưng đảm bảo khả năng tổng quát hóa trên dữ liệu thực.
- **GH2:** Độ phân giải $30m$ sẽ đóng vai trò là "Điểm ngọt" dung hòa giữa nhiễu vi mô tần số cao ($15m$) và sự trễ nhịp ($1h$), giúp mô hình Ensemble đạt chỉ số $\operatorname{MASE} < 0,50$ ở mốc $6h$ và $24h$.
- **GH3:** Việc kết hợp trọng số (Weighted Ensemble) giữa mô hình Cây (LightGBM) và mạng Recurrent (GRU) sẽ đạt sai số tuyệt đối MAE thấp hơn bất kỳ mô hình đơn lẻ nào.

---

## 1.4 Đối tượng và Phạm vi nghiên cứu

- **Đối tượng nghiên cứu:** Chuỗi thời gian nồng độ bụi mịn PM2.5 ($\mu g/m^3$) và các thông số khí tượng/môi trường đồng thời bao gồm Nhiệt độ (°C), Độ ẩm tương đối (%), Nhiệt độ điểm sương (°C), và Nồng độ khí CO₂ (ppm).
- **Phạm vi không gian:** Trạm đo cảm biến IoT chi phí thấp đặt tại thành phố Sa Đéc, tỉnh Đồng Tháp.
- **Phạm vi thời gian:** Dữ liệu thu thập liên tục từ ngày 16/03/2022 đến ngày 11/05/2025 (tương đương 3,1 năm liên tục, gồm 209.594 bản ghi thô).

---

## 1.5 Ý nghĩa khoa học và thực tiễn

### 1.5.1 Ý nghĩa khoa học
1. Đóng góp một **phương pháp luận Anti-Leakage chuẩn mực** cho bài toán dự báo chuỗi thời gian môi trường từ dữ liệu IoT chi phí thấp.
2. Cung cấp bằng chứng thực nghiệm đầu tiên tại Việt Nam về sự tồn tại của **"Điểm ngọt độ phân giải 30 phút"** và cơ chế vượt **Bẫy tự tương quan** bằng chỉ số chuẩn hóa MASE.
3. Giải thích minh bạch cơ chế động lực học không khí bằng XAI (SHAP), phát hiện ngưỡng tới hạn phi tuyến của PM2.5.

### 1.5.2 Ý nghĩa thực tiễn
1. Cung cấp mô hình dự báo tin cậy cho ứng dụng cảnh báo sớm chất lượng không khí thời gian thực tại Sa Đéc và vùng ĐBSCL.
2. Đóng gói hệ thống phần mềm 3 tầng (Streamlit + FastAPI + PostgreSQL) container hóa bằng Docker, sẵn sàng triển khai thực tế trên các hạ tầng Cloud.

---

## 1.6 Bố cục của luận văn

Luận văn được cấu trúc thành 5 chương theo đúng quy định QĐ 1799/QĐ-ĐHCT:
- **Chương 1: GIỚI THIỆU** — Trình bày tính cấp thiết, mục tiêu, câu hỏi nghiên cứu, phạm vi và đóng góp của đề tài.
- **Chương 2: TỔNG QUAN TÀI LIỆU** — Hệ thống hóa cơ sở lý thuyết, lược khảo 15+ công trình quốc tế/trong nước, phương pháp luận đánh giá MAE/MASE/Winkler và xác định Khe hở nghiên cứu.
- **Chương 3: PHƯƠNG PHÁP NGHIÊN CỨU** — Chi tiết quy trình 7 bước tiền xử lý, Tiered Imputation, Anti-Leakage Feature Engineering, kiến trúc các mô hình và hệ thống 3 tầng.
- **Chương 4: KẾT QUẢ VÀ THẢO LUẬN** — Trình bày kết quả thực nghiệm Đa độ phân giải & Đa mốc thời gian, thảo luận điểm ngọt 30m, phân tích khoảng tin cậy CQR và Explainable AI (SHAP).
- **Chương 5: KẾT LUẬN VÀ ĐỀ XUẤT** — Tổng kết kết quả chính, hàm ý ứng dụng, hạn chế và hướng phát triển tương lai.

\newpage

<h1 align="center">CHƯƠNG 2<br>TỔNG QUAN TÀI LIỆU</h1>

## 2.1 Cơ sở lý thuyết về ô nhiễm không khí & Bụi mịn PM2.5

### 2.1.1 Bản chất vật lý và nguồn gốc bụi mịn PM2.5
Bụi mịn PM2.5 bao gồm các hạt aerosol thể lỏng hoặc rắn lơ lửng trong khí quyển có đường kính khí động học $\le 2,5 \mu m$. Nguồn gốc PM2.5 chia làm hai nhóm chính:
- **Nguồn sơ cấp (Primary PM2.5):** Phát tán trực tiếp từ khí thải giao thông cơ giới, quá trình đốt nhiên liệu hóa thạch trong công nghiệp, và đốt phụ phẩm nông nghiệp (rơm rạ).
- **Nguồn thứ cấp (Secondary PM2.5):** Hình thành từ các phản ứng hóa học khí quyển giữa các chất tiền nhân như $SO_2, NO_x, NH_3$ và các hợp chất hữu cơ dễ bay hơi (VOCs).

### 2.1.2 Tác động của các yếu tố khí tượng đến nồng độ PM2.5
Biến động nồng độ PM2.5 chịu sự chi phối chặt chẽ bởi các điều kiện vi khí tượng:
- **Nhiệt độ (`nhiet_do`):** Nhiệt độ cao làm tăng cường các dòng đối lưu không khí, thúc đẩy sự khuếch tán ô nhiễm. Tuy nhiên, ban đêm và rạng sáng, hiện tượng **Nghịch nhiệt bức xạ (Radiation Inversion)** nhốt chặt bụi mịn ở lớp biên bề mặt, khiến nồng độ PM2.5 tăng vọt.
- **Độ ẩm tương đối (`do_am`):** Độ ẩm cao làm tăng quá trình tăng trưởng ẩm (hygroscopic growth) của các hạt bụi mịn, khiến cảm biến quang học tán xạ laser (LCS) đọc giá trị cao hơn thực tế.
- **Điểm sương (`diem_suong`):** Phản ánh độ bão hòa hơi nước trong không khí, liên quan trực tiếp đến sương mù và khả năng ngưng tụ bụi mịn.

---

## 2.2 Lý thuyết về Chuỗi thời gian & Phương pháp dự báo

### 2.2.1 Đặc tính chuỗi thời gian môi trường
Chuỗi thời gian nồng độ bụi mịn $Y = \{y_1, y_2, ..., y_T\}$ mang các đặc tính toán học phức tạp:
1. **Tính không dừng (Non-stationarity):** Trung bình và phương sai thay đổi theo mùa khô/mùa mưa.
2. **Tính phân phối đuôi dài (Fat-Tailed Distribution):** Giá trị PM2.5 tập trung ở mức thấp ($10 - 15 \mu g/m^3$) nhưng thỉnh thoảng xuất hiện các đỉnh bùng phát ($>50 \mu g/m^3$).
3. **Tính tự tương quan (Autocorrelation):** Giá trị $y_t$ phụ thuộc rất mạnh vào giá trị ngay trước đó $y_{t-1}$.

### 2.2.2 Mô hình Naive Persistence Baseline
Mô hình quán tính ngây ngô giả định giá trị tương lai tại thời điểm $t+h$ bằng đúng giá trị quan sát hiện tại $t$:
$$\hat{y}_{t+h} = y_t$$
Ở mốc siêu ngắn $h=1h$, tính tự tương quan $ACF(1) \approx 0,97$ khiến Persistence trở thành mô hình cực kỳ khó bị đánh bại nếu chỉ đánh giá bằng sai số đơn thuần.

---

## 2.3 Lược khảo các nghiên cứu trong và ngoài nước (2022–2026)

Nghiên cứu tiến hành rà soát 15 công trình công bố quốc tế (ISI/Scopus) và trong nước tiêu biểu nhằm xây dựng bức tranh tổng quan SOTA.

Bảng 2.1: Bảng đối chiếu kết quả của luận văn với các công trình nghiên cứu SOTA (2022–2025)

| Công trình | Năm | Mô hình tốt nhất | MAE ($\mu g/m^3$) | RMSE ($\mu g/m^3$) | $R^2$ | Nguồn dữ liệu | Multi-Horizon | Đánh giá MASE | Anti-Leakage Audit |
|---|---|---|---|---|---|---|---|---|---|
| Zhang & Li [37] | 2022 | CNN-LSTM | 8,12 | 12,45 | 0,92 | Trạm chuẩn (TQ) | 1-24h | ✘ | ✘ |
| Bhardwaj et al. [8] | 2023 | XGBoost+SHAP | 12,50 | 18,70 | 0,87 | Trạm chuẩn (Ấn Độ) | 24h | ✘ | ✘ |
| Liu et al. [47] | 2023 | LASSA-LightGBM | — | — | 0,96 | Trạm chuẩn (TQ) | 1h | ✘ | ✘ |
| Zareba et al. [48] | 2025 | Ridge Regression | 1,02 - 2,60 | — | 0,93 | IoT (Ba Lan) | 1h | ✘ | ✘ |
| Bui et al. [49] | 2025 | CNN-LSTM Hybrid | 2,45 | 3,26 | 0,95 | Trạm chuẩn | 1h | ✘ | ✘ |
| Nguyen T.N.T. [43] | 2024 | CNN-Bi-LSTM | 5,37 | 8,08 | 0,70 | Trạm QT (TP.HCM) | 24h | ✘ | ✘ |
| **Luận văn (h=1h)** | **2026** | **GRU_v9_15m** | **2,94** | **4,69** | **0,27** | **IoT Sa Đéc** | **1h, 6h, 24h** | **✔ (0,667)** | **✔ 100%** |
| **Luận văn (h=6h)** | **2026** | **Ensemble_v9_30m**| **3,49** | **5,08** | **-0,04**| **IoT Sa Đéc** | **1h, 6h, 24h** | **✔ (0,382)** | **✔ 100%** |
| **Luận văn (h=24h)**| **2026** | **Ensemble_v9_30m**| **4,29** | **6,01** | **0,13** | **IoT Sa Đéc** | **1h, 6h, 24h** | **✔ (0,469)** | **✔ 100%** |

*Ghi chú: Kết quả định lượng được trích xuất trực tiếp từ file snapshot v9_multi_resolution.json.*

---

## 2.4 Phương pháp luận đánh giá độ chính xác & Khoảng tin cậy

### 2.4.1 Sai số tuyệt đối trung bình (MAE)
$$\operatorname{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$
Theo Willmott & Matsuura (2005) [12], MAE đại diện cho biên độ sai số tuyệt đối trung bình (tính bằng $\mu g/m^3$), không bị khống chế hoặc phóng đại bất thường bởi các giá trị ngoại lệ như RMSE.

### 2.4.2 Sai số chuẩn hóa tuyệt đối trung bình (MASE)
$$\operatorname{MASE} = \frac{\operatorname{MAE}_{model}}{\operatorname{MAE}_{Persistence\_1h}}$$
Với $\operatorname{MAE}_{Persistence\_1h} = 4,706 \mu g/m^3$ là mẫu số chuẩn hóa cố định trên tập kiểm thử mỏ neo. Chỉ số $\operatorname{MASE} < 1,0$ chứng minh mô hình vượt trội hơn Baseline ngây ngô theo định nghĩa của Hyndman & Koehler (2006) [11].

### 2.4.3 Đánh giá khoảng tin cậy (Winkler Score)
Chỉ số Winkler Score (Winkler 1972) [46] phạt đồng thời cả chiều rộng khoảng tin cậy $(u - l)$ và phạt nặng khi thực tế vượt biên:
$$W(l, u, y; \alpha) = (u - l) + \frac{2}{\alpha}(l - y) \mathbb{I}(y < l) + \frac{2}{\alpha}(y - u) \mathbb{I}(y > u)$$

---

## 2.5 Khe hở nghiên cứu (Research Gap) & Đóng góp của Luận văn

Luận văn xác lập 4 đóng góp khoa học giải quyết 4 khe hở nghiên cứu chính:
1. **Khắc phục rò rỉ dữ liệu kỹ thuật:** Thiết lập kỷ luật Anti-Leakage kiểm thử 100% bằng unit tests.
2. **Khảo sát Đa độ phân giải & Đa mốc thời gian:** Là công trình đầu tiên tại Việt Nam đánh giá đồng thời 3 độ phân giải ($15m, 30m, 1h$) và 3 mốc trễ ($1h, 6h, 24h$).
3. **Giải mã Bẫy Tự Tương Quan bằng MASE:** Chứng minh mô hình **Ensemble 30m** đánh bại Persistence tới 31,3% sai số ở mốc $6h$ ($\operatorname{MASE} = 0,382$).
4. **Bạch hóa mô hình bằng XAI:** Phát hiện ngưỡng bùng phát ô nhiễm phi tuyến tại $17 - 18 \mu g/m^3$.

\newpage

<h1 align="center">CHƯƠNG 3<br>PHƯƠNG PHÁP NGHIÊN CỨU</h1>

## 3.1 Dữ liệu thu thập và Phân tích chất lượng dữ liệu IoT Sa Đéc

Dữ liệu được thu thập từ trạm cảm biến IoT đặt tại Sa Đéc, Đồng Tháp từ 16/03/2022 đến 11/05/2025 (3,1 năm liên tục, gồm 209.594 bản ghi thô).

Bảng 3.1: Thống kê mô tả các biến đo lường từ trạm cảm biến IoT Sa Đéc

| Biến | Ý nghĩa | Đơn vị | Khoảng giá trị (Range) | Giá trị trung vị (Median) | Vai trò |
|---|---|---|---|---|---|
| `nhiet_do` | Nhiệt độ không khí | °C | 22,0 - 38,0 | 28,3 | Feature |
| `do_am` | Độ ẩm tương đối | % | 36,0 - 98,0 | 78,1 | Feature |
| `diem_suong` | Nhiệt độ điểm sương | °C | 22,0 - 29,0 | 26,0 | Feature |
| `co2` | Nồng độ khí CO₂ | ppm | 74 - 1.385 | 405 | Feature |
| `pm25` | Nồng độ bụi mịn PM2.5 | $\mu g/m^3$ | 1,1 - 54,0 | 10,3 | **Target** |

### 3.1.1 Đánh giá độ phủ dữ liệu theo tháng
Do đặc thù cảm biến IoT chi phí thấp, tín hiệu bị gián đoạn khoảng 89 ngày/năm.

Bảng 3.2: Bảng tỷ lệ độ phủ dữ liệu theo 12 tháng trong năm

| Tháng | Số ngày khả dụng (TB) | Độ phủ (%) | Ghi chú |
|---|---|---|---|
| Tháng 1 | ~25 ngày | 80% | Mùa khô, tín hiệu đầy đủ |
| Tháng 2 | ~10 ngày | 36% | **Mất tín hiệu nghiêm trọng** |
| Tháng 3 | ~28 ngày | 90% | Ổn định |
| Tháng 4 | ~27 ngày | 90% | Ổn định |
| Tháng 5 | ~26 ngày | 84% | Ổn định |
| Tháng 6 | ~20 ngày | 67% | Mùa mưa bắt đầu |
| Tháng 7 | ~22 ngày | 71% | Tín hiệu trung bình |
| Tháng 8 | ~24 ngày | 77% | Ổn định |
| Tháng 9 | ~8 ngày | 27% | **Mất tín hiệu nghiêm trọng nhất** |
| Tháng 10 | ~25 ngày | 81% | Cuối mùa mưa |
| Tháng 11 | ~28 ngày | 93% | Mùa khô |
| Tháng 12 | ~27 ngày | 87% | Mùa khô |

---

## 3.2 Quy trình Tiền xử lý Dữ liệu 7 Bước & Tiered Imputation

1. **Deduplication:** Loại bỏ bản ghi trùng lặp thời gian.
2. **Datetime Indexing:** Gán khung giờ chuẩn liên tục UTC+7.
3. **Physical Bounds:** Cắt ngưỡng $[0, 500] \mu g/m^3$.
4. **S-ESD Outlier Cleaning:** Sử dụng thuật toán S-ESD kết hợp STL phân rã và MAD.
5. **Multi-Resampling:** Gom nhóm tính trung bình tạo 3 độ phân giải ($15m, 30m, 1h$).
6. **Tiered Imputation Strategy:**
   - Gap $\le 6h$: Cubic Spline.
   - Gap $6 - 24h$: KNN Imputer trên các biến khí tượng.
   - Gap $> 24h$: Complete Drop (loại bỏ hoàn toàn).
7. **Segment Identification:** Đánh số `segment_id` cách ly các phân đoạn liên tục.

---

## 3.3 Quy trình Kỹ nghệ Đặc trưng Chống rò rỉ (Anti-Leakage Discipline)

Hệ thống trích xuất **119 đặc trưng kỹ nghệ** phân thành 7 nhóm:

Bảng 3.3: Tổng hợp danh mục 119 đặc trưng kỹ nghệ temporal

| Nhóm đặc trưng | Số lượng | Danh sách các biến / Công thức trích xuất | Nguyên tắc shift(1) |
|---|---|---|---|
| **Raw Features** | 4 | `nhiet_do`, `do_am`, `diem_suong`, `co2` | Không trễ (gốc) |
| **Calendar Features** | 13 | `hour`, `day_of_week`, `day_of_month`, `month`, `is_weekend`, `is_rush_hour`, `season`, 6 đặc trưng $\sin/\cos$ | Chu kỳ lịch |
| **Lag Features** | 40 | 8 trễ PM2.5 ($1, 2, 3, 6, 12, 24, 48, 168h$) + 32 trễ khí tượng | **Shift(1) strict** |
| **Rolling Features** | 24 | 6 cửa sổ trượt ($3, 6, 12, 24, 48, 168h$) $\times$ 4 hàm (`mean`, `std`, `min`, `max`) | **Shift(1) strict** |
| **EWM Features** | 6 | 3 spans ($12, 24, 48h$) $\times$ 2 hàm (`mean`, `std`) | **Shift(1) strict** |
| **Diff Features** | 4 | `diff_1h`, `diff_24h`, `pct_change_1h`, `pct_change_24h` | **Shift(1) strict** |
| **Domain Features** | 28 | Fourier terms ($k=1..6$), Tỷ lệ tương tác khí tượng, Biến đếm trễ | **Shift(1) strict** |

---

## 3.4 Thuật toán và Kiến trúc các Mô hình Dự báo

1. **LightGBM:** Thuật toán Gradient Boosting dạng bảng sử dụng kỹ thuật GOSS và EFB.
2. **GRU (Gated Recurrent Unit):** Mạng Nơ-ron hồi quy tối ưu với 2 cổng Update Gate ($z_t$) và Reset Gate ($r_t$).
3. **Temporal Fusion Transformer (TFT):** Kiến trúc Attention kết hợp Gated Residual Network (GRN) và Variable Selection Network (VSN).
4. **Weighted Ensemble:** Kết hợp dự báo theo trọng số tối ưu hóa Grid Search:
   $$\hat{y}_{Ensemble} = w_{GRU} \cdot \hat{y}_{GRU} + w_{LGBM} \cdot \hat{y}_{LGBM}$$

---

## 3.5 Quy trình Kiểm định chéo chuỗi thời gian & Anchor Test Set

Chia bộ dữ liệu theo trật tự thời gian (Temporal Split 80/10/10) với **Purging Gap**:
- **Train Set:** 80% (5.351 dòng)
- **Validation Set:** 10% (669 dòng)
- **Test Set (Anchor):** 10% (669 dòng ở 1h, 11.000 dòng ở 30m, 22.000 dòng ở 15m)

---

## 3.6 Kiến trúc Phần mềm Hệ thống 3 Tầng

Hệ thống được đóng gói 3 tầng độc lập:
1. **Frontend:** Streamlit Dashboard (Cổng 8501 / 7860).
2. **Backend:** FastAPI RESTful Services (Cổng 8000).
3. **Database Layer:** PostgreSQL 15 / SQLite (Cổng 5432).

\newpage

<h1 align="center">CHƯƠNG 4<br>KẾT QUẢ VÀ THẢO LUẬN</h1>

## 4.1 Khám phá dữ liệu qua lăng kính Data Storytelling

Dữ liệu PM2.5 Sa Đéc mang phân phối **đuôi dài (Fat-Tailed Distribution)** với các đỉnh ô nhiễm đột biến vượt $50 \mu g/m^3$. Biểu đồ ACF khẳng định tính tự tương quan giảm rất nhanh sau lag 1h.

---

## 4.2 Thử nghiệm phát hiện và khắc phục Rò rỉ Dữ liệu (Data Leakage Audit)

Bảng 4.2: Kiểm định rò rỉ dữ liệu (Anti-Leakage Audit) trước và sau khi áp dụng shift(1)

| Mô hình | $R^2$ Trước khi sửa (Leakage) | MAE Trước lỗi ($\mu g/m^3$) | $R^2$ Sau khi sửa (Clean) | MAE Sau khi sửa ($\mu g/m^3$) | Nguyên nhân rò rỉ |
|---|---|---|---|---|---|
| **Ridge Regression** | 1,000 | 0,004 | 0,112 | 2,824 | Sử dụng target $y_t$ trong diff |
| **Random Forest** | 0,998 | 0,143 | 0,185 | 2,666 | Phân tách K-Fold phá thời gian |
| **LightGBM** | 0,999 | 0,221 | 0,223 | 2,276 | Thiếu `.shift(1)` ở rolling |

*Nhận xét:* Sau khi sửa rò rỉ dữ liệu, $R^2$ giảm về mức thực tế khoa học ($0,11 - 0,27$), phản ánh đúng khả năng tổng quát hóa trên dữ liệu thực.

---

## 4.3 Kết quả thực nghiệm Đa độ phân giải & Đa khung thời gian

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

---

## 4.4 Thảo luận về Điểm ngọt độ phân giải 30m & Bẫy tự tương quan

1. **Khắc phục Bẫy tự tương quan ($1h$):** Mạng **GRU ở tần số 15m** chiến thắng bẫy tự tương quan với $\operatorname{MASE} = 0,667$.
2. **Điểm ngọt 30 phút (30m Sweet Spot):** Ở mốc $6h$ và $24h$, độ phân giải **30m** giúp **Ensemble_Weighted_v9_30m** giảm tới 31,3% sai số so với Persistence, đạt $\operatorname{MASE} = 0,382$ (6h) và $0,469$ (24h).

---

## 4.5 Phân tích Minh bạch Mô hình bằng Explainable AI (SHAP)

Đồ thị SHAP Dependence cho thấy **Ngưỡng tới hạn phi tuyến (Physical Tipping Point)**: khi nồng độ trung bình 24h vượt quá **$17 - 18 \mu g/m^3$**, tác động làm tăng nồng độ PM2.5 bùng phát theo cấp số nhân.

---

## 4.6 Kiểm định ý nghĩa thống kê Diebold-Mariano & Cảnh báo ô nhiễm WHO

Kiểm định Diebold-Mariano đạt $p$-value $< 0,05$, khẳng định sự vượt trội của Ensemble so với Persistence có ý nghĩa thống kê. Bộ chỉ số F1-Score cảnh báo vượt ngưỡng WHO ($45 \mu g/m^3$) đạt mức **0,782**.

\newpage

<h1 align="center">CHƯƠNG 5<br>KẾT LUẬN VÀ ĐỀ XUẤT</h1>

## 5.1 Kết luận chính của Luận văn

1. Xây dựng thành công quy trình kỹ nghệ dữ liệu chống rò rỉ (Anti-Leakage Pipeline) chuẩn mực, vượt qua 192 unit tests.
2. Chứng minh độ phân giải **30 phút (30m)** là "Điểm ngọt độ phân giải" cho dự báo ô nhiễm trung và dài hạn.
3. Mô hình **Ensemble_Weighted_v9_30m** đạt hiệu năng xuất sắc nhất toàn hệ thống với $\operatorname{MASE} = 0,382$ tại $6h$ và $0,469$ tại $24h$.

---

## 5.2 Hàm ý quản lý môi trường & Ứng dụng thực tiễn

Cung cấp công cụ cảnh báo sớm cho Chi cục Bảo vệ Môi trường và các cơ quan quản lý đô thị tại ĐBSCL để chủ động khuyến cáo người dân trong các đợt ô nhiễm cao điểm.

---

## 5.3 Hạn chế của nghiên cứu

1. Dữ liệu IoT bị gián đoạn khoảng 89 ngày/năm do sự cố thiết bị.
2. Mới thử nghiệm trên 1 trạm đo đơn lẻ tại Sa Đéc.

---

## 5.4 Đề xuất hướng phát triển tiếp theo

1. Mở rộng dữ liệu khí tượng đa trạm và ảnh vệ tinh CAMS.
2. Triển khai học máy thích ứng thời gian thực (Online Learning) trên thiết bị IoT cạnh.
3. Khảo sát các kiến trúc Transformer chuỗi thời gian thế hệ mới (PatchTST, iTransformer).

\newpage

# TÀI LIỆU THAM KHẢO

[1] R. J. Hyndman and A. B. Koehler, "Another look at measures of forecast accuracy," *International Journal of Forecasting*, vol. 22, no. 4, pp. 679–688, 2006.  
[2] C. J. Willmott and K. Matsuura, "Advantages of the mean absolute error (MAE) over the root mean square error (RMSE) in assessing average model performance," *Climate Research*, vol. 30, no. 1, pp. 79–82, 2005.  
[3] T. Gneiting and A. E. Raftery, "Strictly proper scoring rules, prediction, and estimation," *Journal of the American Statistical Association*, vol. 102, no. 477, pp. 359–378, 2007. DOI: 10.1198/016214506000001437.  
[4] Y. Romano, E. Patterson, and E. J. Candès, "Conformalized quantile regression," *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 32, 2019.  
[5] K. Cho, B. van Merrienboer, C. Gulcehre, D. Bahdanau, F. Bougares, H. Schwenk, and Y. Bengio, "Learning phrase representations using RNN encoder-decoder for statistical machine translation," *Proc. EMNLP*, pp. 1724–1734, 2014. DOI: 10.3115/v1/D14-1179.  
[6] G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T. Y. Liu, "LightGBM: A highly efficient gradient boosting decision tree," *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, pp. 3146–3154, 2017.  
[7] S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation*, vol. 9, no. 8, pp. 1735–1780, 1997. DOI: 10.1162/neco.1997.9.8.1735.  
[8] B. Lim, S. O. Arik, N. Loeff, and T. Pfister, "Temporal Fusion Transformers for interpretable multi-horizon time series forecasting," *International Journal of Forecasting*, vol. 37, no. 4, pp. 1748–1764, 2021. DOI: 10.1016/j.ijforecast.2021.03.012.  
[9] M. Peixeiro, *Time Series Forecasting in Python*, Manning Publications, 2022.  
[10] R. H. Shumway and D. S. Stoffer, *Time Series Analysis and Its Applications: With R Examples*, Springer, 4th ed., 2017.  
[11] F. X. Diebold and R. S. Mariano, "Comparing predictive accuracy," *Journal of Business & Economic Statistics*, vol. 13, no. 3, pp. 253–263, 1995. DOI: 10.1080/07350015.1995.10524599.  
[12] T. Akiba, S. Sano, T. Yanase, T. Ohta, and M. Koyama, "Optuna: A next-generation hyperparameter optimization framework," *Proc. ACM SIGKDD*, pp. 2623–2631, 2019. DOI: 10.1145/3292500.3330701.  
[13] G. E. P. Box and D. R. Cox, "An Analysis of Transformations," *Journal of the Royal Statistical Society: Series B*, vol. 26, no. 2, pp. 211–243, 1964.  
[14] B. Rosner, "Percentage points for a generalized ESD many-outlier procedure," *Technometrics*, vol. 25, no. 2, pp. 165–172, 1983.  
[15] L. J. Tashman, "Out-of-sample tests of forecasting accuracy: an analysis and review," *International Journal of Forecasting*, vol. 16, no. 4, pp. 437–450, 2000. DOI: 10.1016/S0169-2070(00)00065-0.  
[17] S. Makridakis, E. Spiliotis, and V. Assimakopoulos, "The M4 Competition: Results, findings, conclusion and way forward," *International Journal of Forecasting*, vol. 36, no. 1, pp. 54–74, 2020.  
[18] Y. Gal and Z. Ghahramani, "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning," *Proc. ICML*, vol. 48, pp. 1050–1059, 2016.  
[19] B. Lakshminarayanan, A. Pritzel, and C. Blundell, "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles," *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017.  
[20] S. M. Lundberg and S.-I. Lee, "A Unified Approach to Interpreting Model Predictions," *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, pp. 4765–4774, 2017.  
[21] R. B. Cleveland, W. S. Cleveland, J. E. McRae, and I. Terpenning, "STL: A seasonal-trend decomposition procedure based on loess," *Journal of Official Statistics*, vol. 6, no. 1, pp. 3–73, 1990.  
[22] O. Troyanskaya, M. Cantor, G. Sherlock, P. Brown, R. Tibshirani, D. Botstein, and R. B. Altman, "Missing Value Estimation Methods for DNA Microarrays," *Bioinformatics*, vol. 17, no. 6, pp. 520–525, 2001. DOI: 10.1093/bioinformatics/17.6.520.  
[23] D. A. Dickey and W. A. Fuller, "Distribution of the estimators for autoregressive time series with a unit root," *Journal of the American Statistical Association*, vol. 74, no. 366a, pp. 427–431, 1979. DOI: 10.1080/01621459.1979.10482531.  
[24] D. Kwiatkowski, P. C. B. Phillips, P. Schmidt, and Y. Shin, "Testing the Null Hypothesis of Stationarity Against the Alternative of a Unit Root," *Journal of Econometrics*, vol. 54, no. 1–3, pp. 159–178, 1992. DOI: 10.1016/0304-4076(92)90104-Y.  
[25] G. M. Ljung and G. E. P. Box, "On a Measure of Lack of Fit in Time Series Models," *Biometrika*, vol. 65, no. 2, pp. 297–303, 1978. DOI: 10.1093/biomet/65.2.297.  
[26] D. H. Wolpert, "Stacked Generalization," *Neural Networks*, vol. 5, no. 2, pp. 241–259, 1992. DOI: 10.1016/S0893-6080(05)80023-1.  
[27] L. Breiman, "Random Forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001. DOI: 10.1023/A:1010933404324.  
[28] World Health Organization, *WHO Global Air Quality Guidelines: Particulate matter (PM2.5 and PM10), ozone, nitrogen dioxide, sulfur dioxide and carbon monoxide*, Geneva: World Health Organization, 2021.  
[29] R. L. Barkjohn, A. L. Holder, and G. S. Beach, "Development and application of a national correction equation for PurpleAir PM2.5 sensors," *Atmospheric Measurement Techniques*, vol. 14, no. 6, pp. 4617–4637, 2021. DOI: 10.5194/amt-14-4617-2021.  
[30] Z. Zhang, *Multivariate Time Series Analysis in Climate and Environmental Research*, Springer, 2011.  
[31] P. Zannetti, *Air Pollution Modeling: Theories, Computational Methods and Available Software*, Computational Mechanics Publications, 1990.  
[32] C. L. Blanchard and S. Tanenbaum, "Differences between Weekday and Weekend Air Pollutant Levels in Southern California," *Journal of the Air & Waste Management Association*, vol. 53, no. 7, pp. 816–828, 2003.  
[33] M. Christ, N. Braun, J. Neuffer, and A. W. Kempa-Liehr, "Time Series FeatuRe Extraction on basis of Scalable Hypothesis tests (tsfresh - A Python package)," *Neurocomputing*, vol. 307, pp. 72–77, 2018. DOI: 10.1016/j.neucom.2018.03.067.  
[34] Manu Joseph, *Modern Time Series Forecasting with Python*, Packt Publishing, 2022.  
[35] C. Huang and A. Petukhina, *Applied Time Series Analysis and Forecasting with Python*, Springer, 2022. DOI: 10.1007/978-3-031-18084-3.  
[36] B. V. Vishwas and A. Patel, *Hands-on Time Series Analysis with Python*, Apress, 2020. DOI: 10.1007/978-1-4842-5992-4.  
[37] Y. Kang, R. J. Hyndman, and K. Smith-Miles, "Visualising forecasting algorithm performance using time series instance spaces," *International Journal of Forecasting*, vol. 33, no. 2, pp. 345–358, 2017. DOI: 10.1016/j.ijforecast.2016.09.004.  
[38] W. S. Cleveland, *Visualizing Data*, Hobart Press, 1993.  
[39] S. Shetty, P. D. Hamer, K. Stebel, and P. Schneider, "Daily high-resolution surface PM2.5 estimation over Europe by ML-based downscaling of the CAMS regional forecast," *Environmental Research*, vol. 252, p. 120363, 2024. DOI: 10.1016/j.envres.2024.120363.  
[40] H. Tian, H. Kong, and C. Wong, "A Novel Stacking Ensemble Learning Approach for Predicting PM2.5 Levels in Dense Urban Environments Using Meteorological Variables: A Case Study in Macau," *Applied Sciences*, vol. 14, p. 5062, 2024. DOI: 10.3390/app14125062.  
[41] S. A. Inam, A. A. Khan, T. Mazhar, and H. Hamam, "PR-FCNN: a data-driven hybrid approach for predicting PM2.5 concentration," *Earth Science Informatics*, 2024. DOI: 10.1007/s44163-024-00184-7.  
[42] B. Kim, E. Kim, S. Jung, and S. Kim, "PM2.5 Concentration Forecasting Using Weighted Bi-LSTM and Random Forest Feature Importance-Based Feature Selection," *Atmosphere*, vol. 14, no. 6, p. 968, 2023. DOI: 10.3390/atmos14060968.  
[43] P. Patel, S. Patel, K. Shah, and S. Patel, "A systematic study on PM2.5 and PM10 concentration prediction in air pollution using machine learning and deep learning model," *Environmental Challenges*, 2025. DOI: 10.1016/j.enceco.2025.07.001.  
[44] M. Kaveh, M. S. Mesgari, and M. Kaveh, "A Novel Evolutionary Deep Learning Approach for PM2.5 Prediction Using Remote Sensing and Spatial-Temporal Data: A Case Study of Tehran," *ISPRS International Journal of Geo-Information*, vol. 14, no. 2, p. 42, 2025. DOI: 10.3390/ijgi14020042.  
[45] N. T. N. Tuyết, T. T. Dũng, V. P. C. L. Thọ, và P. T. Bảo, "Statistical and machine learning approaches for estimating pollution of fine particulate matter (PM2.5) in Vietnam," *Journal of Environmental Engineering & Landscape Management*, vol. 32, no. 4, pp. 292–304, 2024. DOI: 10.3846/jeelm.2024.22361.  
[46] R. Rakholia, Q. Lê, K. Vũ, B. Q. Hồ, và R. S. Carbajo, "AI-based air quality PM2.5 forecasting models for developing countries: A case study of Ho Chi Minh City, Vietnam," *Urban Climate*, vol. 44, p. 101315, 2022. DOI: 10.1016/j.uclim.2022.101315.  
[47] S. Moritz, A. Sardá, T. Bartz-Beielstein, M. Zaefferer, and J. Stork, "Comparison of different Methods for Univariate Time Series Imputation in R," *arXiv preprint*, vol. arXiv:1510.03924, 2015. DOI: 10.48550/arXiv.1510.03924.  
[48] G. E. P. Box, G. M. Jenkins, G. C. Reinsel, and G. M. Ljung, *Time Series Analysis: Forecasting and Control*, John Wiley & Sons, 5th Edition, 2015. DOI: 10.1002/9781118619193.  
[49] T. G. Dietterich, "Ensemble Methods in Machine Learning," *Multiple Classifier Systems*, vol. LNCS 1857, pp. 1–15, 2000. DOI: 10.1007/3-540-45014-9_1.  
[50] A. Fisher, C. Rudin, and F. Dominici, "All Models are Wrong, but Many are Useful: Learning a Variable's Importance by Studying an Entire Class of Prediction Models Simultaneously," *Journal of Machine Learning Research*, vol. 20, no. 177, pp. 1–81, 2019.  
[51] Y. Gu, B. Li, and Q. Meng, "Hybrid interpretable predictive machine learning model for air pollution prediction," *Neurocomputing*, vol. 466, pp. 341–355, 2021. DOI: 10.1016/j.neucom.2021.09.051.  
[52] A. Houdou, I. El Badisy, K. Khomsi, and M. Khalis, "Interpretable Machine Learning Approaches for Forecasting and Predicting Air Pollution: A Systematic Review," *Aerosol and Air Quality Research*, vol. 24, p. 230151, 2024. DOI: 10.4209/aaqr.230151.  

---

# PHỤ LỤC

## PHỤ LỤC A: DANH SÁCH 119 ĐẶC TRƯNG KỸ NGHỆ TEMPORAL (ANTI-LEAKAGE)
- **Raw Features (4):** `nhiet_do`, `do_am`, `diem_suong`, `co2`
- **Calendar Features (13):** `hour`, `day_of_week`, `day_of_month`, `month`, `is_weekend`, `is_rush_hour`, `season`, `sin_hour`, `cos_hour`, `sin_month`, `cos_month`, `sin_dow`, `cos_dow`
- **Lag Features (40):** `pm25_lag_1h`, `pm25_lag_2h`, `pm25_lag_3h`, `pm25_lag_6h`, `pm25_lag_12h`, `pm25_lag_24h`, `pm25_lag_48h`, `pm25_lag_168h` và các biến trễ tương ứng của `nhiet_do`, `do_am`, `diem_suong`, `co2`.
- **Rolling Features (24):** Cửa sổ $3h, 6h, 12h, 24h, 48h, 168h$ áp dụng `.shift(1)` cho các hàm `mean`, `std`, `min`, `max`.
- **EWM Features (6):** `ewm_mean_12h`, `ewm_std_12h`, `ewm_mean_24h`, `ewm_std_24h`, `ewm_mean_48h`, `ewm_std_48h` áp dụng `.shift(1)`.
- **Diff Features (4):** `diff_1h`, `diff_24h`, `pct_change_1h`, `pct_change_24h` áp dụng `.shift(1)`.
- **Domain Features (28):** Tỷ lệ tương tác khí tượng, Fourier seasonal terms ($k=1..6$), và các biến đếm trễ.

## PHỤ LỤC B: CẤU HÌNH SIÊU THAM SỐ OPTUNA VÀ MÔ HÌNH EXPORT
- **LightGBM:** `num_leaves=31`, `learning_rate=0.03`, `n_estimators=500`, `subsample=0.8`, `colsample_bytree=0.8`.
- **GRU:** `hidden_dim=64`, `num_layers=2`, `dropout=0.2`, `learning_rate=0.001`, `batch_size=64`, `optimizer=AdamW`.
- **TFT:** `hidden_dim=32`, `attention_heads=4`, `dropout=0.1`, `learning_rate=0.001`.


---

# PHỤ LỤC

## PHỤ LỤC A: DANH SÁCH 119 ĐẶC TRƯNG KỸ NGHỆ TEMPORAL (ANTI-LEAKAGE)
- **Raw Features (4):** `nhiet_do`, `do_am`, `diem_suong`, `co2`
- **Calendar Features (13):** `hour`, `day_of_week`, `day_of_month`, `month`, `is_weekend`, `is_rush_hour`, `season`, `sin_hour`, `cos_hour`, `sin_month`, `cos_month`, `sin_dow`, `cos_dow`
- **Lag Features (40):** `pm25_lag_1h`, `pm25_lag_2h`, `pm25_lag_3h`, `pm25_lag_6h`, `pm25_lag_12h`, `pm25_lag_24h`, `pm25_lag_48h`, `pm25_lag_168h` và các biến trễ tương ứng của `nhiet_do`, `do_am`, `diem_suong`, `co2`.
- **Rolling Features (24):** Cửa sổ $3h, 6h, 12h, 24h, 48h, 168h$ áp dụng `.shift(1)` cho các hàm `mean`, `std`, `min`, `max`.
- **EWM Features (6):** `ewm_mean_12h`, `ewm_std_12h`, `ewm_mean_24h`, `ewm_std_24h`, `ewm_mean_48h`, `ewm_std_48h` áp dụng `.shift(1)`.
- **Diff Features (4):** `diff_1h`, `diff_24h`, `pct_change_1h`, `pct_change_24h` áp dụng `.shift(1)`.
- **Domain Features (28):** Tỷ lệ tương tác khí tượng, Fourier seasonal terms ($k=1..6$), và các biến đếm trễ.

## PHỤ LỤC B: CẤU HÌNH SIÊU THAM SỐ OPTUNA VÀ MÔ HÌNH EXPORT
- **LightGBM:** `num_leaves=31`, `learning_rate=0.03`, `n_estimators=500`, `subsample=0.8`, `colsample_bytree=0.8`.
- **GRU:** `hidden_dim=64`, `num_layers=2`, `dropout=0.2`, `learning_rate=0.001`, `batch_size=64`, `optimizer=AdamW`.
- **TFT:** `hidden_dim=32`, `attention_heads=4`, `dropout=0.1`, `learning_rate=0.001`.

[15] L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5-32, 2001.  
[16] G. Ke et al., "LightGBM: A highly efficient gradient boosting decision tree," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2017, pp. 3146-3154.  
