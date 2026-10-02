---
title: "Khung phân tích đa độ phân giải (15m, 30m, 1h) & Điểm ngọt Pareto 30 phút"
tags: [time-series, multi-resolution, 15m, 30m, 1h, pareto-sweet-spot, sweet-spot]
aliases: ["Điểm ngọt 30 phút", "Multi-Resolution Framing", "Pareto Sweet Spot"]
domain: data-engineering
created: "2026-10-02"
status: completed
---

# 📐 Khung Phân Tích Đa Độ Phân Giải & Điểm Ngọt Pareto 30 Phút

## 1. Động Lực Của Khung Đa Độ Phân Giải (Multi-Resolution Framing)
Hầu hết các nghiên cứu dự báo ô nhiễm truyền thống chỉ vận hành trên một độ phân giải duy nhất:
- **Chuỗi 24 giờ:** Quá thô, bỏ sót toàn bộ dao động trong ngày, không thể phục vụ điều hành giao thông hay cảnh báo khẩn cấp.
- **Chuỗi 1 giờ:** Mắc bẫy tự tương quan nặng ($r=0.86$), biến các mô hình Machine Learning phức tạp thành kẻ thua cuộc trước mô hình ngây thơ Persistence.
- **Chuỗi siêu ngắn (2-5 phút):** Quá nhiều nhiễu vi cơ học (turbulent noise), tỷ lệ nhiễu trên tín hiệu (SNR) quá thấp khiến mô hình dễ bị overfit.

Đề án thiết lập **Khung 3 Độ Phân Giải Đồng Thời**: **15 phút, 30 phút, và 1 giờ**.

---

## 2. Ma Trận Đánh Đổi Tín Hiệu - Nhiễu (Signal-to-Noise Trade-off)

```
Độ phân giải cao (15 phút) ─────────────── Điểm ngọt (30 phút) ─────────────── Độ phân giải thấp (1 giờ)
[+ Bắt trọn xung phát thải]                 [★ CÂN BẰNG TỐI ƯU]               [+ Tín hiệu mượt, xu hướng rõ]
[- Nhiễu vi cơ học lớn]                     [★ 10/15 top-5 ranks]              [- Mắc bẫy tự tương quan r=0.86]
[- Chi phí tính toán cao]                   [★ Hiệu năng cao nhất]             [- Mất mát động lực học ngắn hạn]
```

---

## 3. Điểm Ngọt Pareto 30 Phút (The 30-Minute Sweet Spot)
Phân tích thực nghiệm trên 11 kiến trúc mô hình qua các tầm dự báo ($h=1, 6, 24$) chứng minh:
- **Tần suất lọt top:** Độ phân giải **30 phút chiếm tới 10 trên tổng số 15 vị trí trong top-5 mô hình xuất sắc nhất** toàn đề án.
- **MAE & MASE tối ưu:** Tại độ phân giải 30 phút, mô hình **Weighted Ensemble** đạt:
  - Tầm dự báo 6h: $MASE = 0.382$ (giảm 25.5% sai số so với Persistence).
  - Tầm dự báo 24h: $MASE = 0.469$ (giảm 15.0% sai số so với Persistence).
- **Lý do khoa học:** Khoảng thời gian 30 phút vừa đủ dài để lọc sạch nhiễu cảm biến cục bộ, nhưng cũng vừa đủ ngắn để các hàm toán học (như biến trễ và trích xuất đặc trưng Fourier) bắt kịp đợt sóng tích tụ bụi mịn của giao thông đô thị.

---

## 4. Vai Trò Chiến Lược Của Độ Phân Giải 15 Phút
Mặc dù 30 phút là điểm ngọt toàn diện, độ phân giải **15 phút** đóng vai trò là "vũ khí phá bẫy":
- Tại độ phân giải 1h, mô hình Persistence là bất bại ở tầm $h=1$ ($MASE = 1.000$).
- Tuy nhiên, khi chuyển xuống dữ liệu 15 phút, mạng **GRU 15m** đạt được $MASE = 0.667$ tại tầm dự báo 1h (tương ứng 4 bước 15m), chứng minh khả năng vượt trội có ý nghĩa thực tế (xem [[02_Autocorrelation_Trap_and_GRU_15m]]).

---
*Liên kết liên quan: [[01_Pipeline_7_Steps]] | [[02_Autocorrelation_Trap_and_GRU_15m]] | [[03_Sweet_Spot_30m_Ensemble]] | [[01_Master_Defense_QnA_Hoi_Dong]]*\n