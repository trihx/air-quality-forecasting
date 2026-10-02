---
title: "Kiểm định tính dừng (ADF & KPSS) & Bản chất phi chuẩn của chuỗi PM2.5"
tags: [time-series, stationarity, adf-test, kpss-test, shapiro-wilk, econometrics, fat-tailed]
aliases: ["Kiểm định tính dừng", "ADF và KPSS", "Stationarity Tests", "Bảng 3.5", "Bảng 4.1"]
domain: data-engineering
created: "2026-10-02"
status: completed
---

# 📈 Kiểm Định Tính Dừng (ADF & KPSS) & Bản Chất Phi Chuẩn Chuỗi PM2.5

## 1. Khung Ma Trận Kiểm Định Tính Dừng Kết Hợp (Bảng 3.5 & Bảng 4.1)
Trong kinh tế lượng chuỗi thời gian, việc chỉ dùng một kiểm định tính dừng duy nhất (như ADF) thường dẫn đến kết luận thiên lệch do độ mạnh kiểm định (power of test) hạn chế đối với chuỗi có độ nhớ dài (long-memory process). Đề án áp dụng **khung kiểm định kết hợp ADF và KPSS**:

| Phương pháp kiểm định | Giả thuyết Không ($H_0$) | Thống kê kiểm định | Giá trị $p$-value | Kết luận thống kê |
|---|---|---|---|---|
| **Augmented Dickey-Fuller (ADF)** | Chuỗi có nghiệm đơn vị (Không dừng - Non-stationary) | **$-8.124$** (Critical 1%: $-3.43$) | **$p < 0.001$** | **Bác bỏ $H_0 \implies$ Chuỗi dừng** |
| **KPSS (Kwiatkowski et al.)** | Chuỗi dừng quanh mức xu hướng (Trend-stationary) | **$0.892$** (Critical 5%: $0.463$) | **$p = 0.010 < 0.05$** | **Bác bỏ $H_0 \implies$ Không dừng** |

### Luận giải bản chất kinh tế lượng:
- Khi **ADF bác bỏ $H_0$** (chỉ ra chuỗi dừng) nhưng **KPSS cũng bác bỏ $H_0$** (chỉ ra chuỗi không dừng quanh mức tĩnh), chuỗi thời gian rơi vào trạng thái:
  $$	extbf{Dừng cục bộ & Biến thiên theo mùa (Locally Stationary with Strong Seasonality)}$$
- Chuỗi không có hiện tượng trôi dạt bùng nổ (no unit root / explosive random walk), nhưng phương sai và kỳ vọng thay đổi tuần hoàn theo chu kỳ ngày đêm và các đợt phát thải thời vụ (mùa đốt đồng, mùa mưa).
- Do đó, các mô hình hồi quy tuyến tính cổ điển (OLS) sẽ bị sai lệch nghiêm trọng; cần các mô hình phi tuyến tính mạnh như LightGBM, GRU hoặc mô hình chu kỳ lượng giác Fourier.

---

## 2. Kiểm Định Tính Chuẩn (Normality Tests)
Chuỗi quan trắc $PM_{2.5}$ tại Sa Đéc được kiểm tra phân phối chuẩn qua hai kiểm định:
1. **Shapiro-Wilk Test:** Thống kê $W = 0.781$, giá trị $p < 0.0001 \implies$ Bác bỏ hoàn toàn giả định phân phối Gauss.
2. **Jarque-Bera Test:** Thống kê $JB = 14.820$, $p < 0.0001$.

### Các thông số phân phối định lượng:
- **Độ lệch (Skewness):** $+2.0046$ (lệch phải cực kỳ đậm nét).
- **Độ nhọn (Kurtosis):** $+6.1458$ (đuôi dày, leptokurtic).

### Ý nghĩa thực tiễn:
- Phân phối hạt mịn $PM_{2.5}$ có đặc tính đuôi nặng (Fat-tailed / Heavy-tailed).
- Các phương pháp ước lượng khoảng tin cậy truyền thống dựa trên phân phối chuẩn Gauss ($\hat{y} \pm 1.96 \hat{\sigma}$) hoàn toàn thất bại trong việc bao phủ các đợt bùng phát ô nhiễm.
- Đây là cơ sở lý thuyết bắt buộc đề án phải phát triển module **Conformal Prediction (CQR + ACI)** để có bảo đảm toán học không phụ thuộc phân phối (Distribution-free uncertainty quantification).

---
*Liên kết liên quan: [[01_Pipeline_7_Steps]] | [[03_Outlier_Trap_and_Domain_Bounds]] | [[03_Uncertainty_Quantification_CQR_ACI]] | [[01_Master_Defense_QnA_Hoi_Dong]]*\n