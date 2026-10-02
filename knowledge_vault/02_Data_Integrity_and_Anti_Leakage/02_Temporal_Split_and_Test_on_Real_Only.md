---
title: "Phân tách thời gian 80:10:10 & Kỷ luật đánh giá trên dữ liệu thực (Test on Real Only)"
tags: [time-series, temporal-split, train-val-test, test-on-real-only, anchor-split]
aliases: ["Anchor Split 80:10:10", "Test on Real Only", "Chia tập thời gian"]
domain: data-engineering
created: "2026-10-02"
status: completed
---

# ⏳ Phân Tách Thời Gian 80:10:10 & Kỷ Luật "Test on Real Only"

## 1. Tại Sao Cấm Tuyệt Đối K-Fold Cross Validation Ngẫu Nhiên?
Trong dữ liệu chuỗi thời gian (Time-Series Data), các quan sát có sự phụ thuộc chặt chẽ theo chiều thời gian (Temporal Dependence).
- Nếu sử dụng K-Fold CV ngẫu nhiên, các điểm đo ở tương lai sẽ lọt vào tập Train để dự báo quá khứ trong tập Test.
- Điều này vi phạm nghiêm trọng **nguyên lý nhân quả (Principle of Causality)** và dẫn đến việc đánh giá quá mức hiệu năng thực tế.

---

## 2. Thiết Kế Mỏ Neo (Anchor Split) 80:10:10

Toàn bộ dữ liệu sạch sau khi tiền xử lý được phân tách theo trật tự thời gian tuyến tính nghiêm ngặt:

```
[────────────────── Tập TRAIN (80%) ──────────────────][── VAL (10%) ──][── TEST (10%) ──]
 16/03/2022                                   30/09/2024 01/10/2024 31/12/2024 01/01/2025 11/05/2025
 (Học quy luật, xu hướng và chu kỳ mùa)                 (Optuna TPE)   (Kiểm định mỏ neo)
```

### Thông số kỹ thuật các tập:
1. **Tập Huấn Luyện (Train Set - 80%):**
   - Khoảng thời gian: 16/03/2022 đến 30/09/2024 (~30 tháng).
   - Mục đích: Huấn luyện tham số cho 11 mô hình; học các chu kỳ ngày đêm và biến động mùa mưa - mùa khô miền Tây.
2. **Tập Xác Thực (Validation Set - 10%):**
   - Khoảng thời gian: 01/10/2024 đến 31/12/2024 (~3 tháng).
   - Mục đích: Tinh chỉnh siêu tham số bằng Optuna TPE qua 50 trials; thiết lập cơ chế Early Stopping (patience=10) ngăn overfit.
3. **Tập Kiểm Định Mỏ Neo (Anchor Test Set - 10%):**
   - Khoảng thời gian: 01/01/2025 đến 11/05/2025 (~4.5 tháng).
   - Quy mô mẫu: **669 giờ** ở độ phân giải 1h, **863 mẫu** ở 30m, **1.836 mẫu** ở 15m.
   - Mục đích: Đánh giá mù (Out-of-sample Blind Test) hiệu năng mô hình; tuyệt đối không dùng tập này để chọn tham số hay hiệu chuẩn trọng số.

---

## 3. Kỷ Luật "Test on Real Only" (Chỉ Chấm Điểm Trên Dữ Liệu Đo Thật)
Để phục hồi dữ liệu trong quá trình huấn luyện, các kỹ thuật Spline và KNN đã được sử dụng (xem [[02_Tiered_Imputation_and_Data_Sparsity]]). Tuy nhiên:
> **TRÊN TẬP TEST, TẤT CẢ CÁC METRICS ĐÁNH GIÁ (MASE, RMSE, MAE, R²) BẮT BUỘC CHỈ ĐƯỢC TÍNH TOÁN TRÊN NHỮNG ĐIỂM THỜI GIAN CÓ GIÁ TRỊ ĐO THẬT TỪ CẢM BIẾN (`is_imputed == 0`).**

### Ý nghĩa khoa học:
- Nếu đánh giá mô hình trên dữ liệu đã qua nội suy, ta đang chấm điểm xem mô hình ML có bắt chước được thuật toán Spline/KNN hay không, chứ không phải đánh giá khả năng phản ánh tự nhiên.
- Bộ lọc `is_imputed == 0` bảo vệ độ liêm chính học thuật 100%, bảo đảm rằng mọi con số trong Báo cáo Luận văn đều là kết quả đối chứng trực tiếp với thế giới thực.

---
*Liên kết liên quan: [[01_Pipeline_7_Steps]] | [[01_Anti_Leakage_Shift1_Discipline]] | [[01_Metrics_Standard_MASE_over_RMSE]] | [[01_Master_Defense_QnA_Hoi_Dong]]*\n