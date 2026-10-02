---
title: "Tiêu chuẩn vàng MASE thay thế RMSE & Mẫu số chuẩn hóa đồng nhất"
tags: [evaluation, metrics, mase, rmse, mae, hyndman-2006, benchmark-standard]
aliases: ["Tiêu chuẩn MASE", "MASE vs RMSE", "Chuẩn hóa Hyndman"]
domain: evaluation
created: "2026-10-02"
status: completed
---

# 📏 Tiêu Chuẩn Vàng MASE Thay Thế RMSE & Mẫu Số Chuẩn Hóa Đồng Nhất

## 1. Hạn Chế Trầm Trọng Của RMSE, MAE và $R^2$ Trong Chuỗi Thời Gian

### Hạn chế của MAE và RMSE (Scale-dependent):
- MAE và RMSE phụ thuộc chặt chẽ vào thang đo dữ liệu (scale).
- Nồng độ bụi mịn tại Sa Đéc có mức nền trung bình khoảng **$10 - 15\,\mu g/m^3$**. Sai số $MAE = 3.5\,\mu g/m^3$ ở Sa Đéc tương đương với sai số $MAE \approx 45\,\mu g/m^3$ ở những thành phố ô nhiễm nặng như New Delhi ($150\,\mu g/m^3$).
- Do đó, không thể so sánh năng lực dự báo giữa các địa bàn hay các mùa khác nhau chỉ bằng MAE hay RMSE.

### Hạn chế của $R^2$ (Hệ số xác định):
- Trong chuỗi thời gian có tính tự tương quan cao, mẫu số của $R^2$ sử dụng giá trị trung bình $\bar{y}$ làm baseline.
- Việc so sánh với đường trung bình nằm ngang $\bar{y}$ là phi lý vì ngay cả một mô hình ngây thơ copy hôm qua cũng vượt trội hơn $\bar{y}$ rất nhiều. Khi đưa sang tập kiểm định ngoài mẫu (out-of-sample), $R^2$ hoàn toàn có thể nhận giá trị âm nếu mô hình bị lệch pha.

---

## 2. Tiêu Chuẩn Vàng MASE (Hyndman & Koehler, 2006)
Đề án áp dụng **MASE (Mean Absolute Scaled Error)** làm thước đo trung tâm của toàn bộ hệ thống:

$$MASE = \f\frac{MAE_{\text{model}}}{\f\frac{1}{N-1} \sum_{t=2}^{N} |y_t - y_{t-1}|}$$

Trong đó:
- Tử số: Sai số tuyệt đối trung bình của mô hình trên tập kiểm định.
- Mẫu số: Sai số tuyệt đối trung bình của mô hình ngây thơ **In-sample Naive Persistence** tính trên tập Huấn luyện.

### Mẫu số chuẩn hóa đồng nhất trong đề án:
Để bảo đảm tính so sánh công bằng và nhất quán giữa 11 mô hình:
$$MAE_{\text{Persistence\_1h}} = \mathbf{1.821}\,\mu g/m^3$$
Mọi mô hình qua tất cả các tầm dự báo đều được chia cho mẫu số chuẩn hóa cố định này.

---

## 3. Ý Nghĩa Ngưỡng Phân Định MASE = 1.0 (True Predictive Skill)
- **$MASE < 1.0$ (Khu vực Kỹ năng Thật - True Skill):** Mô hình thực sự học được động lực học môi trường, mang lại giá trị gia tăng rõ rệt so với việc đoán mò ngây thơ.
- **$MASE = 1.0$:** Mô hình chỉ ngang bằng với phép dự báo sao chép giá trị gần nhất.
- **$MASE > 1.0$ (Vùng Vô Giá Trị):** Mô hình hoàn toàn vô dụng trong thực tiễn. Việc đầu tư hạ tầng máy tính phức tạp nhưng cho kết quả kém hơn sao chép giá trị trước là một sự lãng phí công nghệ.

Toàn bộ các mô hình Champion của đề án (Weighted Ensemble, GRU 15m) đều đạt $MASE$ trong khoảng **$0.382 - 0.712$**, khẳng định năng lực dự báo thực chất vượt trội.

---
*Liên kết liên quan: [[01_Pipeline_7_Steps]] | [[04_Multi_Resolution_Framing]] | [[02_Diebold_Mariano_Hypothesis_Testing]] | [[01_Master_Defense_QnA_Hoi_Dong]]*\n