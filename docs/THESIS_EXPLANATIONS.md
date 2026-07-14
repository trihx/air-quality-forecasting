# 📚 Cơ Sở Khoa Học & Giải Thích Phương Pháp Luận (Thesis Explanations)

> **Mục đích:** Tài liệu này tổng hợp các giải thích chuyên sâu, insight từ biểu đồ và các quyết định kỹ thuật của dự án. Người dùng có thể sao chép trực tiếp các lập luận này vào Báo cáo Luận văn để củng cố độ tin cậy khoa học của nghiên cứu.

---

## Chương 1: Phương pháp luận Explainable AI (XAI)

### 1.1. Sự phân luồng giải thích: SHAP và Permutation Importance
Trong nghiên cứu dự báo chuỗi thời gian, việc biến các mô hình Machine Learning từ "hộp đen" (Black-box) thành "hộp trắng" (White-box) là yêu cầu bắt buộc để đảm bảo tính minh bạch. Hệ thống của chúng ta áp dụng phân luồng phương pháp giải thích dựa trên bản chất thuật toán:
- **Đối với mô hình Tree-based (LightGBM):** Sử dụng **SHAP (SHapley Additive exPlanations)** thông qua thuật toán `TreeExplainer`. Đây là giải pháp SOTA (State-of-the-Art) cho các mô hình Cây, giúp tính toán chính xác 100% giá trị đóng góp biên (marginal contribution) của từng biến mà tốc độ cực kỳ nhanh.
- **Đối với mô hình Deep Learning (GRU/LSTM/TFT):** Thay vì dùng SHAP (có thể mất hàng giờ chạy `KernelExplainer` và chỉ đưa ra kết quả xấp xỉ), nghiên cứu ưu tiên **Permutation Importance**. Phương pháp này xáo trộn dữ liệu của từng biến để đo lường mức độ suy giảm hiệu suất, qua đó xác định độ quan trọng một cách trực quan và ít tốn kém tài nguyên máy tính.

### 1.2. LightGBM với vai trò Mô hình Đại diện (Surrogate Model)
Trong quá trình trích xuất SHAP values, **LightGBM** được lựa chọn làm "Mô hình giải thích đại diện". Lý do:
1. Độ chính xác thuộc nhóm top đầu trong mảng dữ liệu dạng bảng (Tabular Time-Series), chỉ thua kém đôi chút so với các Ensemble Models.
2. Nắm bắt xuất sắc các mối quan hệ phi tuyến tính (non-linear).
3. Hỗ trợ thuật toán `TreeExplainer` cực nhanh, cho phép chúng ta khám phá động lực học môi trường (environmental dynamics) của toàn bộ tập dữ liệu, từ đó suy luận cách các mô hình phức tạp (Ensemble) tiếp cận dữ liệu.

### 1.3. Cách đọc và Diễn giải SHAP Beeswarm Plot
Biểu đồ Beeswarm là công cụ biểu diễn đa chiều:
- **Trục Y (Mức độ quan trọng):** Các biến có tầm ảnh hưởng lớn nhất nằm ở trên cùng.
- **Trục X (SHAP Value):** Điểm nằm bên phải (số dương) đẩy dự báo tăng lên (ô nhiễm hơn); nằm bên trái (số âm) kéo dự báo giảm xuống (trong lành hơn).
- **Màu sắc (Giá trị thực của biến):** Đỏ thể hiện giá trị cao (High), Xanh thể hiện giá trị thấp (Low).
- **Độ phân tán:** Thể hiện sự tập trung của dữ liệu; các vệt kéo dài thể hiện những tác động đột biến (outliers).

### 1.4. Cơ sở lý thuyết và Tài liệu tham khảo (Theoretical Foundations)
Sự lựa chọn phương pháp Explainable AI (XAI) trong nghiên cứu này không chỉ dựa trên khả năng lập trình mà còn được củng cố bởi nền tảng lý thuyết vững chắc từ các nghiên cứu SOTA (State-of-the-Art):
- **Cơ sở cho SHAP & TreeExplainer:** Nghiên cứu áp dụng SHAP dựa trên lý thuyết trò chơi hợp tác của Lundberg & Lee (2017), cho phép phân bổ công bằng (fair allocation) mức độ đóng góp của từng đặc trưng. Việc chọn thuật toán `TreeExplainer` (Lundberg et al., 2020) giúp giải quyết triệt để rào cản về thời gian tính toán cho mô hình LightGBM, đảm bảo tính nhất quán (consistency) và độ chính xác cục bộ (local accuracy) mà KernelExplainer truyền thống không thể đạt được.
- **Cơ sở cho Permutation Importance:** Do kiến trúc phức tạp của Neural Networks (GRU/LSTM/TFT), việc sử dụng SHAP đòi hỏi tài nguyên tính toán khổng lồ và thường chỉ mang tính xấp xỉ. Luận văn sử dụng phương pháp thay thế là Permutation Importance — một kỹ thuật "model-agnostic" dựa trên nguyên lý xáo trộn biến (Breiman, 2001; Fisher et al., 2019). Thuật toán này đo lường trực tiếp độ sụt giảm hiệu năng (gia tăng MAE) khi cấu trúc của một biến bị phá vỡ, cung cấp một góc nhìn trực quan và đáng tin cậy về tác động của đặc trưng lên mô hình Học sâu.
- **Tương quan với các nghiên cứu Ứng dụng:** Hướng tiếp cận kết hợp các mô hình Machine Learning/Deep Learning với SHAP và Permutation Importance đang là tiêu chuẩn thực hành tốt nhất hiện nay. Nghiên cứu tổng quan của Houdou et al. (2024) chỉ ra rằng SHAP chiếm tới 46.4% trong các phương pháp giải thích mô hình ô nhiễm không khí. Tương tự, Gu et al. (2021) đã chứng minh hiệu quả vượt trội của việc kết hợp Interpretable Machine Learning (như LightGBM/XGBoost với SHAP) để bóc tách các yếu tố tác động đến nồng độ ô nhiễm.

---

## Chương 2: Các Insight Khoa Học Rút Ra Từ Biểu Đồ

### 2.1. Tính Phi tuyến tính (Non-linearity)
Khi phân tích các biến mang tính chu kỳ (ví dụ `fourier_daily_cos_2`), màu sắc phân bổ đan xen thay vì tách bạch hai bên trái-phải. Điều này chứng minh tác động của biến là phi tuyến (phụ thuộc vào sự tương tác chéo với các biến khác như nhiệt độ, độ ẩm). Đây là lợi thế tuyệt đối của Machine Learning so với các mô hình kinh điển như ARIMA (ốn định dạng tuyến tính).

### 2.2. Hiệu ứng "Đuôi dài" (Heavy-tail/Asymmetry) của Ô nhiễm
Biến `pm25_roll_24h_mean` cho thấy hiện tượng vệt màu Đỏ (nồng độ PM2.5 nền cao) kéo dài đột biến về phía bên phải so với cụm màu Xanh. Insight: Mô hình nhận thức được sự bất đối xứng của không khí — khi trời trong lành, mức độ dao động nhỏ; nhưng khi xảy ra nghịch nhiệt hoặc phát thải đột biến, mức độ ô nhiễm sẽ vọt lên một cách cực đoan (không tuân theo phân phối chuẩn phân bổ đều).

### 2.3. Sự Chuyển dịch Trọng tâm (Horizon Shift)
Tầm quan trọng của các biến thay đổi theo khung thời gian dự báo:
- Ở khung ngắn hạn (1h): Mô hình phụ thuộc cực lớn vào các độ trễ ngắn (`lag_1h`), mang tính "quán tính" (Inertial).
- Ở khung trung/dài hạn (6h, 24h): Mô hình gạt bỏ quán tính và tập trung vào các đặc trưng phản ánh xu hướng nền (`pm25_roll_24h_mean`) và các chu kỳ trong ngày (`hour_sin`). Khả năng tự thích ứng này chứng minh hệ thống không bị mắc bẫy Naive/Persistence.

---

## Chương 3: Tiêu chuẩn Đánh giá (Model Evaluation Metrics)

### 3.1. Ý nghĩa hệ thống Metric đa chiều
Nghiên cứu sử dụng kết hợp nhiều chỉ số để tránh góc nhìn phiến diện:
- **MASE (Mean Absolute Scaled Error):** Tiêu chuẩn vàng để đo lường "Kỹ năng thực sự" (Skill Score) của AI. Nếu MASE < 1.0, mô hình thực sự học được quy luật; nếu MASE >= 1.0, mô hình chỉ đơn giản là đang copy dữ liệu ngày hôm trước (Naive approach).
- **RMSE (Root Mean Square Error):** So với MAE, RMSE nhạy cảm và trừng phạt các sai số lớn. Điều này tối quan trọng trong dự báo chất lượng không khí, vì việc dự báo sai một đợt đỉnh điểm ô nhiễm nguy hiểm cho sức khỏe sẽ bị phạt rất nặng.
- **R² (Hệ số xác định):** Đo lường tỷ lệ phương sai mà mô hình giải thích được.
- **DA (Directional Accuracy):** Đo lường "Tỷ lệ đoán đúng xu hướng (Tăng/Giảm)". Đây là chỉ số then chốt giúp các cơ quan quản lý đưa ra quyết định thực tiễn (Ví dụ: "Ngày mai chất lượng không khí diễn biến xấu đi hay tốt lên?").

### 3.2. Lưu trữ Mô hình Lịch sử (Legacy Models)
Một số mô hình ở phiên bản cũ (v1-v5) thiếu các chỉ số R² và DA do khác biệt về pipeline thời kỳ đầu. Để giữ tính toàn vẹn và trung thực của tiến trình nghiên cứu khoa học, các số liệu thiếu được hiển thị là ký hiệu (`—`) thay vì cố tình huấn luyện lại hoặc mô phỏng số liệu. Điều này phản ánh rõ nét sự tiến hóa (evolution) của bộ tiêu chuẩn đánh giá qua từng giai đoạn dự án.

---

## Chương 4: Tính Toàn Vẹn Của Dữ Liệu & Tài Liệu Tham Khảo

### 4.1. Cơ chế Anti-Leakage (Chống rò rỉ dữ liệu)
Rò rỉ dữ liệu (Data Leakage) là lỗi nghiêm trọng nhất có thể làm phá sản một nghiên cứu Time Series. Trong việc tạo đặc trưng (Feature Engineering):
- Tất cả các phép biến đổi như `diff()` (đạo hàm bậc 1) hay `rolling()` đều bắt buộc phải kết hợp với hàm `shift(1)`.
- Ví dụ: Việc tính mức chênh lệch bụi hiện tại trừ đi 1 giờ trước `diff(1)` sẽ chứa thông tin của `Y_target` tại thời điểm `t`. Nếu không lùi `shift(1)` trước khi tính, mô hình sẽ "nhìn trộm" được kết quả tương lai, dẫn đến việc trên biểu đồ Test, đường dự báo (Pred) trùng khít vô lý với đường thực tế (Actual).

### 4.2. Quản trị Tài liệu Tham khảo (Literature Review Integrity)
Trong kỷ nguyên AI, rủi ro lớn nhất của phần Tổng quan Tài liệu là "Ảo giác AI" (Hallucinated Citations - AI tự bịa ra tiêu đề và mã DOI không có thật). Để đảm bảo uy tín học thuật:
- Mọi tài liệu tham khảo trong nghiên cứu phải được đối soát (Cross-verify) thông qua các cơ sở dữ liệu thật (Crossref API/Semantic Scholar).
- Tiêu chí lựa chọn SOTA (State-of-the-Art): Ưu tiên các bài báo thuộc chu kỳ 2022-2025 từ các nguồn uy tín (IEEE, Elsevier, Nature, MDPI) và loại trừ ngay lập tức các bài báo không tra cứu được mã DOI trên hệ thống định danh quốc tế.

---

## Chương 5: Phân tích SHAP Dependence Plots

### 5.1. Lý do tập trung phân tích ở Horizon 6h
Trong các báo cáo phân tích độ tin cậy của mô hình (Explainability), Horizon 6h thường được lựa chọn làm điểm rơi tối ưu (Sweet spot) vì:
- **Horizon 1h bị chi phối bởi quán tính (Inertia):** Ở độ trễ 1 giờ, nồng độ PM2.5 chưa có sự thay đổi rõ rệt, mô hình chủ yếu dựa vào thuật toán Naive (lấy giá trị giờ trước áp cho giờ sau). Biểu đồ SHAP ở 1h thường là tuyến tính tẻ nhạt, không phản ánh được năng lực học sâu của Machine Learning.
- **Horizon 24h có độ bất định cao:** Chứa quá nhiều nhiễu khí tượng chưa thể nắm bắt.
- **Horizon 6h đòi hỏi kỹ năng thực sự (True Skill):** Sau 6 giờ, quán tính ô nhiễm đã bị triệt tiêu. Để dự báo chính xác, mô hình bắt buộc phải học được các quy luật phức tạp như: chu kỳ ngày đêm, lớp nghịch nhiệt nhiệt độ, và tốc độ tích tụ/phân tán ô nhiễm. Đây là lúc sức mạnh nắm bắt tương tác phi tuyến của LightGBM được thể hiện rõ nét nhất.

### 5.2. Các Insight Khoa học từ SHAP Dependence

**A. Hiệu ứng Chu kỳ ngày đêm (Diurnal Cycle - `hour_sin`)**
Biểu đồ phụ thuộc của biến `hour_sin` thể hiện một hình thái phân bổ đối nghịch. Tại các thời điểm có biến thiên dương (ví dụ: ban ngày có bức xạ mặt trời phá vỡ nghịch nhiệt, tạo luồng đối lưu), SHAP value giảm sâu (mô hình chủ động hạ mức dự báo PM2.5). Ngược lại, ở các thời điểm đêm/rạng sáng (thời tiết ổn định, tích tụ bụi), SHAP value tăng vọt. Mối quan hệ tương tác với `pm25_roll_6h_min` (thể hiện qua màu sắc) còn chỉ ra rằng: Bất kể nền bụi trước đó là cao hay thấp, cứ rơi vào "khoảng giờ vàng phát tán", mô hình sẽ tự động kéo giảm dự báo.

**B. Hiệu ứng Ngưỡng bùng phát ô nhiễm (Threshold Effect - `pm25_roll_24h_mean`)**
Sự phụ thuộc của PM2.5 vào trung bình 24h hoàn toàn **không mang tính tuyến tính**. Trên biểu đồ, SHAP value duy trì ở mức âm khi trung bình 24h dưới 15 µg/m³. Tuy nhiên, ngay khi nồng độ này vượt qua một ngưỡng tới hạn (tipping point) khoảng 17-18 µg/m³, SHAP value vọt thẳng đứng lên mức dương cực đại. Điều này chứng minh mô hình đã tự học được giới hạn tự làm sạch (self-cleansing capacity) của tầng khí quyển khu vực; khi nồng độ chất ô nhiễm vượt quá ngưỡng này, sự phân tán giảm mạnh và ô nhiễm sẽ bùng phát theo cấp số nhân trong 6 giờ tới.

**C. Hiệu ứng Tích lũy Nền (Baseline Accumulation - `pm25_roll_24h_min`)**
Bụi PM2.5 không sinh ra và mất đi ngay lập tức. Biểu đồ Dependence của `pm25_roll_24h_min` đi lên theo dạng bậc thang. Nếu mức độ ô nhiễm "sạch nhất" trong ngày (giá trị min) vẫn nằm ở mức cao, kèm theo mức trung bình 48 giờ (`pm25_ewm_48h_mean`) cũng cao (chấm màu đỏ), hệ thống nhận diện đây là một "Đợt ô nhiễm kéo dài" (Prolonged Pollution Episode). Khi đó, mô hình sẽ neo chặt dự báo ở mức cao, bỏ qua các yếu tố nhiễu làm giảm ô nhiễm cục bộ.


### Đánh giá khả năng dự báo với Dữ liệu khuyết thiếu (Data Sparsity)
**1. Dự báo ngắn hạn (Short-term: 1h - 24h):** Khả năng thành công **RẤT CAO**. Dữ liệu IoT dù khuyết nguyên cả tháng, nhưng ở các mảng có dữ liệu, độ phủ theo giờ là 24/24h. Mô hình học rất tốt chu kỳ ngày (Diurnal cycle) và độ tự tương quan (Autocorrelation). Các mô hình GRU/Ensemble của dự án đã chứng minh đạt MASE rất tốt ở horizon ngắn.
**2. Dự báo dài hạn & Chu kỳ mùa:** Khả năng **TRUNG BÌNH/KÉM**. Do thiếu hụt 89 ngày/năm (mù tịt tháng 2 và tháng 9), mô hình không thể học được điểm uốn chuyển mùa. Việc cố ép mô hình học Fourier_Yearly có thể sinh ra sai số nội suy.

### Khuyến nghị xử lý (Imputation vs External Data)
**1. Tuyệt đối KHÔNG DÙNG Machine Learning để Impute gap quá dài (Nguyên tắc dự án):**
Theo quy tắc Tiered Imputation: Spline (gap ngắn) -> KNN (gap trung bình) -> DROP (gap dài >1 tuần). Việc cố dùng ML để bịa ra dữ liệu nguyên 1 tháng sẽ tạo ra tín hiệu giả (hallucination) và gây Data Leakage nghiêm trọng, làm sai lệch mô hình khi test trên dữ liệu thực.
**2. Phương án dùng Dữ liệu Ngoại lai (External Data):**
- **Trạm Lãnh sự quán Mỹ (AirNow):** Tại TP.HCM, dữ liệu đo đạc liên tục, ít bị ngắt quãng. Có thể ghép vào để lấp đầy khoảng mù của chu kỳ mùa.
- **Open-Meteo Historical Air Quality:** Cung cấp API miễn phí để lấy lại dải dữ liệu PM2.5 theo toạ độ địa lý chính xác (2022-2025). Có thể dùng để nội suy chéo (Cross-imputation).


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
- **Ablation Study (Outlier Impact):** Thí nghiệm v10 chứng minh IQR loại bỏ sai 66 đỉnh ô nhiễm >54 µg/m³. Điều này làm mô hình đạt MAE thấp hơn một cách ảo tưởng (False Sense of Accuracy), mất khả năng cảnh báo. Vì vậy, Pipeline chính thức (v9) phải dùng Domain Bounds (0-500) để đảm bảo an toàn y tế.
- **SHAP Analysis:** Horizon shift effect, Threshold bùng phát ô nhiễm ở ~17-18 ug/m3

### Chương 5: Bàn luận & Kết luận
- **Đóng góp khoa học:** Multi-Resolution x Multi-Horizon methodology cho IoT PM2.5
- **Đóng góp kỹ thuật:** Anti-Leakage 4 tầng + Tiered Imputation + Test-on-Real-Only
- **Hạn chế:** Data Sparsity (89 ngày/năm mù), đơn trạm, 4 biến phụ
- **BÀI HỌC OUTLIER TRAP:** Đoạn biện luận quan trọng (xem Chương 7.5)
- **Hướng phát triển:** FD-1 -> FD-6 (xem Dashboard Conclusion page)
- **External Data Policy:** Không merge vào pipeline chính (bias hệ thống PM2.5: IoT ~13.7 vs CAMS ~22.2)
