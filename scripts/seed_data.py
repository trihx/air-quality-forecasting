INFO_CARDS = [
    # ── Nhóm 1: Tổng Quan (Overview) ──
    {
        "card_key": "overview_guide",
        "page": "overview",
        "display_order": 1,
        "title": "Hướng dẫn: Tổng Quan",
        "content": "Trang này trình bày **bức tranh toàn cảnh** của dự án nghiên cứu dự báo PM2.5, bao gồm:\n\n- **Các chỉ số chính (KPI)**: Hiển thị mô hình dự báo hiệu quả nhất, độ bao phủ kiểm thử trên dữ liệu thực tế, và tổng số mô hình đã huấn luyện.\n- **Kiến trúc Quy trình (Pipeline Architecture)**: Minh họa 7 bước thực hiện từ thu thập dữ liệu thô đến đánh giá mô hình.\n- **Bảng xếp hạng Mô hình**: Đánh giá hiệu suất các mô hình tại 3 khung thời gian (1 giờ, 6 giờ, 24 giờ) sử dụng chỉ số MASE.\n- **Tổng kết Phát hiện (Key Findings)**: Phân tích cân bằng giữa các thành tựu đạt được và những điểm còn hạn chế của dự án."
    },
    {
        "card_key": "overview_lessons",
        "page": "overview",
        "display_order": 2,
        "title": "Bài Học Kinh Nghiệm (Key Lessons)",
        "content": "Tổng hợp từ 7 phiên bản phát triển và hơn 16 thí nghiệm, dưới đây là những bài học kinh nghiệm quan trọng nhất:\n\n1. **Kiểm soát rò rỉ dữ liệu (Data Leakage)**: Việc tính toán tỷ lệ thay đổi trực tiếp sẽ vô tình làm lộ thông tin tương lai. Khắc phục bằng cách sử dụng giá trị trễ (`shift`). Các dấu hiệu cảnh báo rò rỉ bao gồm MASE < 0.1 hoặc R² > 0.99.\n2. **Mặt trái của Đặc trưng (Feature Engineering)**: Việc thêm 117 đặc trưng mở rộng giúp mô hình GRU cải thiện 14,8% ở khung 6 giờ, nhưng lại làm giảm hiệu suất 30,5% ở khung 1 giờ do tính tự tương quan (Autocorrelation) ở thời điểm 1 giờ đã đạt ~0.97.\n3. **Biến đổi Logarit (Log Transform)**: Hiệu quả của phép biến đổi tỷ lệ thuận với kiến trúc mô hình. GRU hoạt động tốt hơn với dữ liệu đã biến đổi Log, trong khi LSTM ưu tiên dữ liệu gốc ở khung 6 giờ.\n4. **Đặc trưng chu kỳ (Fourier) vs Khử mùa vụ (Deseasonalizing)**: Sử dụng đặc trưng toán học Fourier mang lại kết quả tốt hơn việc chủ động khử mùa vụ, do việc khử kép dễ gây thêm nhiễu tự nhiên.\n5. **Nguyên tắc ranh giới tập dữ liệu**: Việc áp dụng biến đổi phân tách STL trên toàn bộ dữ liệu gây rò rỉ thông tin ẩn (làm giảm sai số ảo 45%). Mọi thuật toán chuẩn hóa và nội suy phải được ước lượng hoàn toàn trên tập huấn luyện (Train set)."
    },
    {
        "card_key": "overview_improvements",
        "page": "overview",
        "display_order": 3,
        "title": "Cải Tiến Đã Chứng Minh (v1 → v7)",
        "content": "| Điểm Cải Tiến | Trạng thái trước (v1) | Trạng thái sau (v7) | Mức độ Cải thiện |\n|----------|-----------|------|---------------|\n| Đặc trưng Fourier | 95 features | 119 features | LightGBM MAE ↓14.2% |\n| GRU champion (6h) | Chưa triển khai | MASE 0.649 | ↓35.1% so với Persistence |\n| LSTM champion (24h) | Chưa triển khai | MASE 0.663 | ↓33.7% so với Persistence |\n| Rà soát Anti-leakage | 4 nguồn leakage ẩn | Loại bỏ hoàn toàn | Đưa sai số về mức thực tế |\n| Lọc ngoại lệ Domain | Cắt IQR tại 54 µg/m³ | Dải thực tế [0,500] | Giữ lại 1.908 sự kiện ô nhiễm |\n| Thước đo MASE đồng nhất | Bất đồng bộ phương pháp | Chuẩn hóa toàn bộ | Dễ dàng so sánh chéo các họ mô hình |\n| Kiểm thử Test-on-real-only | Có chứa dữ liệu nội suy | Chỉ dùng dữ liệu thực | Đảm bảo tính minh bạch khoa học |"
    },
    # ── Nhóm 2: EDA (Phân Tích Khám Phá) ──
    {
        "card_key": "eda_guide",
        "page": "eda",
        "display_order": 1,
        "title": "Hướng dẫn: Phân Tích Khám Phá (EDA)",
        "content": "Bước đầu tiên trong quá trình xây dựng hệ thống là hiểu rõ các đặc tính thống kê của dữ liệu. Trang phân tích EDA cung cấp các góc nhìn đa chiều:\n\n1. **Tổng Quan**: Thống kê mô tả và Điểm dự báo (Forecastability Score).\n2. **Gaps & Spikes**: Xử lý các vấn đề chất lượng dữ liệu (dữ liệu thiếu, ngoại lệ).\n3. **Tính dừng & Mùa vụ (Stationarity & Seasonality)**: Kiểm định hình thức ADF/KPSS và phân tách chuỗi thời gian STL.\n4. **Tự tương quan & Trôi dạt (Autocorrelation & Drift)**: Đánh giá bẫy tự tương quan và sự thay đổi phân phối.\n5. **Thông tin chuyên sâu (Deep Insights)**: Phân tích tương quan nhân quả Granger và cấu trúc phần dư (Residuals)."
    },
    {
        "card_key": "eda_findings",
        "page": "eda",
        "display_order": 2,
        "title": "Phát hiện quan trọng từ dữ liệu",
        "content": "**Các phát hiện mang tính quyết định:**\n\n- **Tự tương quan cao (ACF-1 ≈ 0.97)**: Dự báo cơ sở (Persistence) rất mạnh mẽ ở khung 1 giờ, thiết lập một rào cản khó vượt qua cho các mô hình AI phức tạp.\n- **Phân phối không chuẩn**: Dữ liệu PM2.5 bị lệch đáng kể (Shapiro p < 1e-50), minh chứng cho tính ưu việt của chỉ số đánh giá MASE so với MAPE.\n- **Tính chu kỳ đặc thù (Diurnal cycle)**: PM2.5 đạt đỉnh lúc 6h sáng (hiện tượng nghịch nhiệt) và chạm đáy lúc 12h trưa (đối lưu nhiệt), hỗ trợ giả thuyết sử dụng các đặc trưng thời gian thực.\n- **Sàn sai số lý thuyết**: Phân tách STL ghi nhận độ lệch chuẩn của phần dư (Residual σ) là 5.18 µg/m³, tạo ra giới hạn tối thiểu về sai số cho bất kỳ mô hình dự báo nào."
    },
    {
        "card_key": "eda_lessons",
        "page": "eda",
        "display_order": 3,
        "title": "Bài Học: Xử lý ngoại lệ (Outliers) cho IoT",
        "content": "Trong phân tích dữ liệu mạng lưới thiết bị (IoT), không phải mọi giá trị ngoại lệ về mặt thống kê đều là nhiễu.\n\nBan đầu, phương pháp Z-score hoặc khoảng tứ phân vị (IQR) cắt bỏ các giá trị PM2.5 tại 54 µg/m³. Điều này vô tình loại bỏ 1.908 mẫu dữ liệu của các đợt không khí thực sự ô nhiễm (PM2.5 > 100), do bị hiểu nhầm là lỗi cảm biến.\n\n**Giải pháp**: Chuyển sang sử dụng **ngưỡng vật lý thực tế của miền cảm biến [0, 500]** đối với đặc trưng PM2.5 (phân phối đuôi dày), trong khi vẫn giữ phương pháp thống kê cho các yếu tố khí tượng khác. Điều này giúp mô hình học được các chuỗi sự kiện ô nhiễm khắc nghiệt."
    },
    # ── Nhóm 3: Cấu hình và Huấn luyện (Hyperparameters & Training) ──
    {
        "card_key": "hyperparams_guide",
        "page": "hyperparams",
        "display_order": 1,
        "title": "Cấu hình Siêu Tham Số (Hyperparameters)",
        "content": "Trang này liệt kê các thiết lập siêu tham số tối ưu đã được lựa chọn qua quá trình tìm kiếm (Hyperparameter tuning) cho từng họ mô hình:\n\n- **LightGBM**: Sử dụng bộ lấy mẫu Optuna TPE, cấu hình 100 trials cho mỗi khung thời gian dự báo, đánh giá qua xác thực chéo chuỗi thời gian (TimeSeriesSplit với k=5).\n- **GRU/LSTM**: Cửa sổ nhìn lại (Lookback) 72 giờ, kích thước tầng ẩn (hidden_dim) 64, cấu trúc 2 layers, và áp dụng cơ chế dừng sớm (Early Stopping) với patience = 10.\n- **ARIMA/SARIMA**: Áp dụng Auto ARIMA dựa trên tiêu chuẩn thông tin Akaike (AIC), với kích thước cửa sổ cuộn (rolling window) là 720 giờ."
    },
    {
        "card_key": "models_guide",
        "page": "hyperparams",
        "display_order": 2,
        "title": "Tổng quan Cấu trúc Mô hình",
        "content": "Hệ thống tích hợp đa dạng các họ mô hình để khai thác tối đa đặc trưng của dữ liệu chuỗi thời gian:\n\n1. **Nhóm mô hình cây (LightGBM/XGBoost)**: Khai thác tốt các tương tác phi tuyến tính phức tạp giữa các yếu tố khí tượng và PM2.5, với ưu điểm tốc độ huấn luyện nhanh và khả năng chống quá khớp tốt.\n2. **Nhóm Học Sâu (GRU/LSTM)**: Kiến trúc mạng nơ-ron hồi quy được thiết kế riêng để ghi nhớ các phụ thuộc dài hạn. Đặc biệt hiệu quả trong việc nắm bắt chu kỳ ngày đêm của chất lượng không khí.\n3. **Nhóm Thống kê & Ensemble**: Tích hợp các mô hình ARIMA như đường cơ sở vững chắc và phương pháp Stacking/Voting để tổng hợp sức mạnh từ nhiều mô hình đơn lẻ."
    },
    {
        "card_key": "training_guide",
        "page": "training",
        "display_order": 1,
        "title": "Môi trường Huấn Luyện (Training Sandbox)",
        "content": "Chức năng Huấn Luyện (Sandbox) cho phép chạy thử nghiệm và kiểm chứng tính tái lập (reproducibility) của các mô hình trực tiếp trên hệ thống:\n\n- Cung cấp khả năng tùy chỉnh tỷ lệ phân chia dữ liệu, siêu tham số và khung thời gian dự báo.\n- Trực quan hóa nhật ký huấn luyện (training logs) và các chỉ số đánh giá (MAE, RMSE, MASE, R²) trong thời gian thực.\n- **Lưu ý kỹ thuật**: Để đảm bảo tốc độ phản hồi trên giao diện, kích thước tập dữ liệu đưa vào Sandbox đã được giới hạn tối ưu. Môi trường này đặc biệt hữu ích để xác thực lại hiệu suất của các mô hình học máy dạng cây (Tree-based models)."
    },
    {
        "card_key": "methodology_references",
        "page": "training",
        "display_order": 2,
        "title": "Phương pháp luận Kiến trúc Hệ thống",
        "content": "**Quy trình Xử lý Dữ liệu (Data Pipeline)**:\n1. Tải và làm sạch dữ liệu: Lọc dải giá trị thực tế của PM2.5 [0,500] và tái lấy mẫu ở tần suất 1 giờ.\n2. Điền khuyết kết hợp (Hybrid Imputation): Sử dụng Spline cho các khoảng trống ngắn (≤ 6h) và nội suy không gian KNN cho các khoảng trống dài hơn (6-24h).\n3. Trích xuất 119 đặc trưng: Tích hợp các biến số quá khứ (lag), chu kỳ (Fourier) và tương tác chéo.\n4. Phân tách tập dữ liệu: Giữ nguyên trình tự thời gian với tỷ lệ 80/10/10.\n5. Kiểm thử: Đánh giá duy nhất trên dữ liệu thực (loại bỏ các điểm đã được điền khuyết).\n\n**Phương pháp Xác thực Walk-Forward Validation**:\n- Áp dụng TimeSeriesSplit với cửa sổ mở rộng (expanding window) để duy trì cấu trúc thời gian.\n- Tránh tuyệt đối phương pháp K-fold ngẫu nhiên do sẽ gây ra rò rỉ thông tin tương lai trong phân tích chuỗi thời gian."
    },
    {
        "card_key": "pitfalls_lessons",
        "page": "training",
        "display_order": 3,
        "title": "Bài học Khắc phục Lỗi Hệ thống (Pitfalls)",
        "content": "Các vấn đề kỹ thuật nghiêm trọng đã được phát hiện và giải quyết trong suốt quá trình phát triển:\n\n- **Xung đột Thư viện Đa nền tảng**: Quá trình nạp thư viện `torch` đồng thời với `LightGBM` gây ra lỗi phân vùng bộ nhớ (segfault) trên hệ thống MPS/CUDA. Giải pháp là áp dụng cơ chế Lazy import nạp thư viện tuần tự.\n- **Lỗi Bùng nổ Đặc trưng Phương sai (CV Explodes)**: Khi trung bình tiến gần 0, Hệ số biến thiên (CV) tăng tới vô cực. Đã khắc phục bằng cách thiết lập ngưỡng dưới (clamp mean ≥ 1.0) và giới hạn giá trị tối đa (clip CV ≤ 5.0).\n- **Sai lệch Trục thời gian ở Baseline**: Mô hình Persistence tính toán thiếu độ dời (offset) của tham số Lookback trong Deep Learning, dẫn đến kết quả MASE tiến tới vô cực. Đã được đồng bộ hóa lại thông qua chỉ số Unified Persistence MAE."
    },
    # ── Nhóm 4: Đánh giá Đa khung thời gian (Multi-Horizon Evaluation) ──
    {
        "card_key": "multi_horizon_guide",
        "page": "multi_horizon",
        "display_order": 1,
        "title": "Hướng dẫn: Đánh giá Đa khung thời gian",
        "content": "Trang đánh giá cung cấp cái nhìn toàn diện về hiệu suất của các mô hình qua 3 khung thời gian dự báo (1 giờ, 6 giờ, 24 giờ):\n\n- **Biểu đồ MASE**: Chỉ số Sai số Tuyệt đối Có hướng (Mean Absolute Scaled Error). Giá trị < 1.0 biểu thị mô hình dự báo học máy hoạt động hiệu quả hơn phương pháp dự báo cơ sở (Persistence).\n- **Xu hướng MAE**: Biểu diễn độ lệch tuyệt đối trung bình theo thời gian.\n- **Kiểm định Diebold-Mariano**: Xác nhận tính ý nghĩa thống kê của sự khác biệt hiệu suất giữa các mô hình (p-value < 0.05).\n\n**Tiêu chuẩn đánh giá khoa học**:\n- Áp dụng chung một đường cơ sở (Unified Persistence MAE) cho mọi họ mô hình.\n- Kiểm thử độc quyền trên dữ liệu thực tế (`is_imputed == 0`) để phản ánh đúng sai số trong điều kiện triển khai."
    },
    {
        "card_key": "multi_horizon_findings",
        "page": "multi_horizon",
        "display_order": 2,
        "title": "Phát hiện cốt lõi: Không tồn tại mô hình vạn năng",
        "content": "**Kết luận thống kê**:\n\n- **Ngắn hạn (1 giờ)**: Hiện tượng 'Bẫy tự tương quan' (Autocorrelation Trap). Tự tương quan của PM2.5 ở độ trễ 1 giờ đạt mức ~0.97, khiến phương pháp cơ sở (Persistence) trở nên cực kỳ mạnh mẽ. Không mô hình học máy nào vượt qua ngưỡng MASE = 1.0 (GRU đạt tốt nhất ở mức 1.009).\n- **Trung hạn (6 giờ)**: Hệ số tự tương quan giảm xuống 0.85, tạo không gian cho các mạng nơ-ron phát huy tác dụng. Mô hình GRU vươn lên dẫn đầu với MASE = 0.649 (giảm 35,1% sai số so với Persistence).\n- **Dài hạn (24 giờ)**: Tự tương quan giảm sâu còn 0.45. Khả năng ghi nhớ dài hạn của kiến trúc LSTM giúp nó chiến thắng tuyệt đối với MASE = 0.663.\n\nKết quả này khẳng định sự cần thiết của việc triển khai một hệ thống dự báo tích hợp (Ensemble System) tùy biến theo khoảng thời gian dự báo."
    },
    {
        "card_key": "multi_horizon_references",
        "page": "multi_horizon",
        "display_order": 3,
        "title": "Đánh giá Ý nghĩa Thống kê (Diebold-Mariano Test)",
        "content": "Việc một mô hình có sai số thấp hơn mô hình khác có thể chỉ do yếu tố ngẫu nhiên của tập dữ liệu kiểm thử. Để đảm bảo tính chặt chẽ khoa học, dự án áp dụng **Kiểm định Diebold-Mariano**.\n\nKết quả kiểm định xác nhận:\n- Ở khung 6 giờ, sự vượt trội của mô hình GRU so với Persistence là hoàn toàn có ý nghĩa thống kê (Giá trị DM = -4.21, p-value < 0.001).\n- Tương tự, ở khung 24 giờ, lợi thế của LSTM cũng vượt qua bài kiểm định (p-value = 0.014).\n\nĐiều này chứng minh hiệu năng của các mô hình học sâu (Deep Learning) mang lại giá trị gia tăng thực sự, thay vì chỉ là sự trùng hợp thống kê."
    },
    {
        "card_key": "multi_horizon_comparison",
        "page": "multi_horizon",
        "display_order": 4,
        "title": "So sánh với các Nghiên cứu Hiện hành",
        "content": "Kết quả của dự án được đối chiếu với các công trình nghiên cứu hiện hành trong và ngoài nước:\n\n1. **Khắc phục sự phụ thuộc nồng độ nền**: Các nghiên cứu truyền thống thường chỉ báo cáo MAE/RMSE. Do nồng độ PM2.5 trung bình tại Sa Đéc thấp (~10 µg/m³), MAE tuyệt đối của dự án rất nhỏ (4.36-4.61 µg/m³). Việc tiên phong áp dụng chỉ số MASE (theo Hyndman & Koehler, 2006) giúp so sánh hiệu suất một cách công bằng mà không bị ảnh hưởng bởi nồng độ nền.\n2. **Loại trừ rò rỉ dữ liệu (Anti-leakage Rigor)**: Rất ít nghiên cứu tại Việt Nam công bố quy trình kiểm toán rò rỉ dữ liệu rõ ràng. Việc lọc dữ liệu kiểm thử nghiêm ngặt (`test-on-real-only`) đảm bảo mô hình có thể triển khai thực tế.\n3. **Đánh giá đa khung thời gian**: Khắc phục nhược điểm của việc chỉ báo cáo sai số ở khung 24 giờ, dự án vạch trần được sự đánh đổi (trade-off) hiệu suất tại các khung thời gian khác nhau."
    },
    {
        "card_key": "insight_no_single_best",
        "page": "multi_horizon",
        "display_order": 5,
        "title": "Góc nhìn Chuyên sâu (Insight): Trade-off Hiệu suất",
        "content": "Sự suy giảm của hệ số tự tương quan (Autocorrelation) là động lực chính giải thích sự chuyển dịch ngôi vương giữa các mô hình:\n\n- Khi tự tương quan duy trì ở mức cao (>0.9), các thông tin mở rộng (biến đa lượng, kiến trúc phức tạp) không tạo ra giá trị gia tăng đủ bù đắp cho lượng nhiễu (noise) mà chúng mang lại.\n- Chỉ khi tín hiệu chuỗi thời gian phân rã qua các dự báo trung và dài hạn, khả năng chiết xuất tương tác phi tuyến của mạng nơ-ron hồi quy mới thực sự tỏa sáng."
    },
    # ── Nhóm 5: Phân tích Lỗi và Giải thích Mô hình (Results & XAI) ──
    {
        "card_key": "actual_vs_predicted_guide",
        "page": "actual_vs_predicted",
        "display_order": 1,
        "title": "Hướng dẫn: Đối chiếu Thực tế và Dự báo",
        "content": "Biểu đồ lớp phủ (Overlay chart) cung cấp cái nhìn trực quan nhất về độ bám sát của mô hình so với thực tế trên tập dữ liệu kiểm thử (Test set):\n\n- **Đường màu xanh (Actual)**: Giá trị PM2.5 thực tế đo được từ trạm quan trắc.\n- **Đường màu tím (Predicted)**: Giá trị dự báo của hệ thống Trí tuệ Nhân tạo.\n- **Đường màu cam (Persistence)**: Giá trị của phương pháp cơ sở (sao chép trực tiếp giá trị giờ trước).\n\nĐường nền hiển thị phân loại chất lượng không khí (AQI) theo chuẩn của Tổ chức Y tế Thế giới (WHO), giúp đánh giá xem sai số của mô hình có làm thay đổi mức độ cảnh báo sức khỏe hay không."
    },
    {
        "card_key": "actual_vs_predicted_lessons",
        "page": "actual_vs_predicted",
        "display_order": 2,
        "title": "Bài Học: Nguyên tắc Kiểm thử Khắt khe",
        "content": "Một rủi ro lớn trong đánh giá hiệu năng là việc sử dụng dữ liệu đã được điền khuyết (Imputed data) để tính toán sai số. Vì dữ liệu điền khuyết thường tuân theo các hàm nội suy mượt mà, việc mô hình dễ dàng đoán đúng chúng sẽ tạo ra mức sai số thấp giả tạo.\n\n**Chính sách của dự án**:\n- Áp dụng bộ lọc `is_imputed == 0` trên toàn bộ tập kiểm thử.\n- Việc tính toán MAE và MASE chỉ được thực hiện trên những giá trị thực sự được cảm biến ghi nhận.\n- Sự minh bạch này giúp kết quả nghiên cứu phản ánh đúng hiệu năng trong môi trường thực tiễn, thay vì chỉ là các con số đẹp trên phòng thí nghiệm."
    },
    {
        "card_key": "shap_guide",
        "page": "shap",
        "display_order": 1,
        "title": "Hướng dẫn: Giải thích Quyết định của AI (XAI)",
        "content": "Để tránh việc mô hình hoạt động như một 'hộp đen' (Black box), dự án áp dụng các phương pháp Giải thích Trí tuệ Nhân tạo (XAI) nhằm bóc tách lý do đằng sau mỗi dự báo:\n\n- **Biểu đồ cột (Bar chart)**: Xếp hạng tầm quan trọng tổng thể của các đặc trưng.\n- **Biểu đồ đàn ong (Beeswarm)**: Phân bố tác động của một đặc trưng. Dấu chấm đỏ (giá trị cao) lệch về bên phải nghĩa là yếu tố đó làm tăng dự báo PM2.5.\n- **Sự phụ thuộc (Dependence)**: Biểu diễn mối quan hệ phi tuyến tính giữa sự thay đổi của một yếu tố và đầu ra của mô hình.\n\n*Phương pháp áp dụng: TreeSHAP cho các mô hình cây (LightGBM) và Permutation Importance cho mạng nơ-ron hồi quy (GRU).*"
    },
    {
        "card_key": "shap_lessons",
        "page": "shap",
        "display_order": 2,
        "title": "Phát hiện: Sự dịch chuyển Tầm quan trọng",
        "content": "Phân tích SHAP đã tiết lộ sự dịch chuyển thú vị về cơ chế ra quyết định của mô hình khi khoảng thời gian dự báo (horizon) kéo dài:\n\n- **Ở khung 1 giờ**: Đặc trưng `pm25_lag_1h` áp đảo hoàn toàn với đóng góp >90%. Điều này tái khẳng định hiện tượng bẫy tự tương quan mạnh mẽ.\n- **Ở khung 6 giờ**: Tầm quan trọng phân tán đều hơn. Các đặc trưng thời gian thực (Giờ trong ngày, các giá trị sóng sin/cos) và nhiệt độ bắt đầu đóng góp đáng kể.\n- **Ở khung 24 giờ**: Các yếu tố khí tượng (như độ ẩm, điểm sương) lọt vào nhóm dẫn đầu, cho thấy mô hình đã chuyển từ việc phụ thuộc vào nồng độ bụi sang phụ thuộc vào quy luật thay đổi thời tiết.\n\n**Đặc biệt**, các đặc trưng toán học Fourier luôn nằm trong top quan trọng. Do chỉ được tạo ra từ nhãn thời gian (timestamp), chúng mang lại lợi ích kép: không rủi ro rò rỉ dữ liệu và không tốn chi phí thu thập."
    },
    {
        "card_key": "pi_guide",
        "page": "prediction_intervals",
        "display_order": 1,
        "title": "Hướng dẫn: Khoảng Tin cậy (Prediction Intervals)",
        "content": "Trong thực tiễn, việc cung cấp một khoảng giá trị an toàn quan trọng hơn việc chỉ đưa ra một điểm dự báo duy nhất. Hệ thống đánh giá mức độ bất định (Uncertainty quantification) thông qua 3 phương pháp:\n\n1. **Conformal Prediction**: Đảm bảo mức độ bao phủ không phụ thuộc vào phân phối dữ liệu.\n2. **Quantile Regression**: Mô hình học cách dự báo biên dưới (5%) và biên trên (95%) trực tiếp từ hàm mất mát.\n3. **MC Dropout**: Ứng dụng kỹ thuật ngẫu nhiên hóa mạng nơ-ron nhiều lần để đo lường phương sai dự đoán của kiến trúc học sâu.\n\n*Kết quả ghi nhận: MC Dropout thường tạo ra khoảng tin cậy quá hẹp do tỷ lệ ngắt kết nối (dropout rate) tiêu chuẩn của GRU khá nhỏ, trong khi Quantile Regression cân bằng tốt giữa độ rộng và độ tin cậy.*"
    },
    {
        "card_key": "forecast_guide",
        "page": "forecast",
        "display_order": 1,
        "title": "Hướng dẫn: Ứng dụng Dự báo Trực tiếp",
        "content": "Mô đun này ứng dụng các mô hình đã được huấn luyện tốt nhất để thực hiện dự báo theo thời gian thực (Real-time forecasting):\n\n- Cho phép người dùng tùy chọn họ mô hình (LightGBM cho tốc độ phản hồi nhanh, hoặc GRU-TorchScript cho hiệu suất phân tích chuỗi).\n- Hỗ trợ tinh chỉnh thủ công các tham số khí tượng hiện tại (như lượng CO2, nhiệt độ, độ ẩm) để kiểm tra các kịch bản 'Nếu - Thì' (What-if scenarios).\n- Kết quả đầu ra được tự động phân loại theo tiêu chuẩn chất lượng không khí (AQI) của WHO để cung cấp các cảnh báo dễ hiểu cho người dùng."
    },
    # ── Nhóm 6: Lịch sử Thí nghiệm & Trợ lý AI (History & AI Assistant) ──
    {
        "card_key": "experiment_runs_guide",
        "page": "experiment_runs",
        "display_order": 1,
        "title": "Hướng dẫn: Nhật ký Thí nghiệm (Experiment Tracking)",
        "content": "Phân hệ này hoạt động như một hệ thống quản lý phiên bản (Version Control) cho toàn bộ quy trình khoa học dữ liệu:\n\n- **Tab So sánh Phiên bản**: Cho phép đối chiếu trực tiếp giữa 2 phiên bản bất kỳ để xem sự thay đổi về bộ đặc trưng (Feature set diff) và sự cải thiện của chỉ số MASE.\n- **Tab Lịch sử Chi tiết**: Hiển thị danh sách toàn bộ các mô hình đã huấn luyện, với đầy đủ siêu tham số và kết quả đánh giá (Metadata log).\n\nMỗi phiên bản (v1 → v7) đều được đóng gói kèm theo các báo cáo phân tích nguyên nhân - kết quả (What/Why/Result) để đảm bảo tính minh bạch."
    },
    {
        "card_key": "experiment_runs_lessons",
        "page": "experiment_runs",
        "display_order": 2,
        "title": "Bài Học: Nguyên tắc Phiên bản hóa (Versioning Rigor)",
        "content": "Trong học máy, việc theo dõi các phiên bản thí nghiệm (Versioning) có tầm quan trọng tương đương với việc quản lý mã nguồn trong kỹ nghệ phần mềm.\n\n**Các nguyên tắc cốt lõi đã áp dụng**:\n- **Tính bất biến (Immutability)**: Không bao giờ ghi đè lên kết quả thí nghiệm cũ. Mọi thay đổi đều tạo ra một phiên bản con kế thừa (thông qua trường `parent_version`).\n- **Đánh dấu toàn diện**: Mỗi log chạy bắt buộc phải lưu trữ: Dấu thời gian, tên mô hình, cấu hình siêu tham số, các chỉ số đánh giá, và mã băm (hash) của tập dữ liệu.\n- Nhờ nguyên tắc này, dự án đã dễ dàng truy xuất và so sánh được hiệu quả của việc thêm 117 đặc trưng (v5) so với đường cơ sở (v1) mà không làm mất đi các thiết lập ban đầu."
    },
    {
        "card_key": "history_guide",
        "page": "experiment_runs",
        "display_order": 3,
        "title": "Lưu trữ và Truy xuất Phiên bản",
        "content": "Toàn bộ lịch sử các lần chạy thí nghiệm được lưu trữ liên tục dưới dạng các tệp JSON và đồng bộ hóa vào hệ thống cơ sở dữ liệu.\n\nViệc lưu trữ bao gồm sự giải trình cho mỗi thay đổi cấu trúc dữ liệu hoặc hàm mục tiêu. Quá trình này giúp trả lời nhanh các câu hỏi kiểm toán: *Tính năng này được thêm vào khi nào? Tại sao lại thêm nó? Và hiệu năng đã thay đổi ra sao sau khi thêm?*"
    },
    {
        "card_key": "ai_assistant_guide",
        "page": "ai_assistant",
        "display_order": 1,
        "title": "Hướng dẫn: Trợ lý Nghiên cứu AI (RAG)",
        "content": "Trợ lý ảo được thiết kế dựa trên kiến trúc RAG (Retrieval-Augmented Generation), đóng vai trò như một chuyên gia nắm vững toàn bộ nghiệp vụ dự án:\n\n- **Cơ sở Tri thức (Knowledge Base)**: Tích hợp hơn 241 tài liệu khoa học, mã nguồn và kết quả báo cáo.\n- **Công cụ tìm kiếm Vector**: Sử dụng ChromaDB kết hợp với mô hình nhúng đa ngôn ngữ (Multilingual embeddings) hỗ trợ truy xuất ngữ nghĩa chính xác.\n- **Đa cấu hình Ngôn ngữ Lớn (LLM)**: Tích hợp linh hoạt với các API (Gemini, OpenAI, Groq) và hỗ trợ kết nối bảo mật qua LM Studio (Local deployment).\n\n**Ví dụ về các truy vấn chuyên sâu**:\n- *'Giải thích hiện tượng MASE > 1 ở khung dự báo 1 giờ?'*\n- *'Phân tích sự khác biệt hiệu suất giữa GRU và LSTM ở khung 24 giờ.'*\n- *'Tại sao hệ thống lựa chọn thuật toán Cubic Spline thay vì nội suy tuyến tính?'*"
    }
]
