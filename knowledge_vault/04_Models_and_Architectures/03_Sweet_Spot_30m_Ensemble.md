---
title: "Sự thống trị của Ensemble Weighted tại điểm ngọt 30 phút"
tags: [models, ensemble, 30m, pareto, mase, weighted-ensemble, champion]
aliases: ["Ensemble Weighted 30m", "Thống trị Ensemble", "Điểm ngọt 30 phút"]
domain: models
created: "2026-10-02"
status: completed
---

# 🏆 Sự Thống Trị Của Ensemble Weighted Tại Điểm Ngọt 30 Phút

## 1. Mô Hình Quán Quân Của Đề Án: Weighted Optimization Ensemble
Mô hình kết hợp trọng số tối ưu (Weighted Optimization Ensemble) là thành tựu kiến trúc cao nhất của nghiên cứu, kết hợp ưu điểm của 3 trường phái mô hình đơn lẻ xuất sắc nhất:
1. **LightGBM:** Xuất sắc trong việc phân tách các ngưỡng điều kiện vi khí hậu phi tuyến tính.
2. **GRU:** Nắm bắt hoàn hảo quán tính và sự phụ thuộc thời gian liên tục.
3. **Random Forest:** Tạo sự ổn định, làm mượt các dự báo cực đoan.

Công thức tối ưu hóa trọng số lồi:
$$\min_{\mathbf{w}} \sum_{t} \l\left| y_t - \sum_{i=1}^M w_i \hat{y}_{i,t} 
\right| \quad \text{s.t.} \quad \sum_{i=1}^M w_i = 1, \quad w_i \ge 0$$
được giải bằng quy hoạch tuyến tính (Linear Programming) trên tập Validation.

---

## 2. Bảng Thành Tích Vượt Trội Tại Độ Phân Giải 30 Phút

| Chân trời dự báo (Horizon) | MASE Persistence Baseline | MASE LightGBM Đơn Lẻ | MASE GRU Đơn Lẻ | MASE Weighted Ensemble | Mức cải thiện so với Baseline |
|---|---|---|---|---|---|
| **$h = 1$ (30 phút)** | $1.000$ | $0.842$ | $0.789$ | **$0.712$** | **+28.8%** |
| **$h = 6$ (3 giờ)** | $1.000$ | $0.465$ | $0.421$ | **$0.382$** | **+25.5% (Tối ưu nhất)** |
| **$h = 24$ (12 giờ)** | $1.000$ | $0.548$ | $0.512$ | **$0.469$** | **+15.0%** |

---

## 3. Luận Giải Khoa Học: Tại Sao Ensemble 30m Lại Bất Khả Chiến Bại?
1. **Bù trừ sai số đa dạng (Error Diversity):** Sai số của cây quyết định LightGBM thường có phương sai cao ở các vùng dữ liệu thưa thớt; trong khi đó mạng GRU có xu hướng dự báo trơn mượt hơn. Khi phối hợp với nhau, phần dư của hai mô hình triệt tiêu lẫn nhau, giảm thiểu đáng kể phương sai sai số tổng thể.
2. **Điểm ngọt Pareto 30 phút:** Tại độ phân giải 30 phút, tỷ lệ tín hiệu trên nhiễu (SNR) đạt giá trị tối ưu, giúp bộ tối ưu trọng số tìm được điểm cân bằng hoàn hảo giữa khả năng phản ứng nhanh và độ ổn định lâu dài.
3. **Độ tin cậy trong điều hành thực tế:** Mô hình Weighted Ensemble không bao giờ phụ thuộc vào một thuật toán duy nhất. Nếu một cảm biến khí tượng bị lỗi khiến LightGBM suy giảm độ chính xác, GRU vẫn duy trì dự báo dựa trên chuỗi lịch sử $PM_{2.5}$.

---
*Liên kết liên quan: [[04_Multi_Resolution_Framing]] | [[01_Benchmark_Models_Overview]] | [[02_Diebold_Mariano_Hypothesis_Testing]] | [[02_Slides_Narrative_and_Key_Arguments]]*\n