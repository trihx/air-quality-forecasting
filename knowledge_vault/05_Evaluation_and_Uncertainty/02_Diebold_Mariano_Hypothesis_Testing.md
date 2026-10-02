---
title: "Kiểm định ý nghĩa thống kê Diebold-Mariano & Hiệu chỉnh Harvey-HLN"
tags: [evaluation, hypothesis-testing, diebold-mariano, dm-test, hln-correction, p-value, table-47]
aliases: ["Kiểm định Diebold-Mariano", "DM Test", "Hiệu chỉnh HLN", "Bảng 4.7"]
domain: evaluation
created: "2026-10-02"
status: completed
---

# ⚖️ Kiểm Định Ý Nghĩa Thống Kê Diebold-Mariano & Hiệu Chỉnh Harvey-HLN

## 1. Tại Sao Phải Cần Kiểm Định Giả Thuyết Thống Kê?
Trong thực nghiệm khoa học, việc mô hình A có $MASE = 0.382$ thấp hơn mô hình B có $MASE = 0.465$ chỉ là bằng chứng mô tả (descriptive evidence).
- Sự chênh lệch đó có thể chỉ do may mắn ngẫu nhiên trên một lát cắt dữ liệu thử nghiệm.
- Để khẳng định mô hình đề xuất **thực sự vượt trội một cách có ý nghĩa thống kê**, bắt buộc phải thực hiện kiểm định giả thuyết **Diebold-Mariano (DM Test, 1995)**.

---

## 2. Cơ Sở Toán Học & Hiệu Chỉnh Harvey-HLN (1997)

### Chuỗi vi phân hàm tổn thất (Loss Differential):
Giả sử ta so sánh Mô hình Đề xuất (1) và Mô hình Đối chứng (2):
$$d_t = L(e_{1,t}) - L(e_{2,t}) = |e_{1,t}| - |e_{2,t}|$$
Giả thuyết Không ($H_0$): Hai mô hình có độ chính xác tương đương nhau ($E[d_t] = 0$).

### Thống kê kiểm định DM gốc:
$$DM = \f\frac{\b\bar{d}}{\sqrt{\hat{V}(\b\bar{d}) / T}}$$
trong đó $\hat{V}(\b\bar{d})$ là ước lượng phương sai tự tương quan dài hạn (dùng Newey-West spectral kernel).

### Hiệu chỉnh mẫu hữu hạn Harvey, Leybourne & Newbold (HLN, 1997):
Khi chân trời dự báo $h > 1$, kiểm định DM gốc bị sai lệch kích thước (size distortion) trên các mẫu hữu hạn. Công thức hiệu chỉnh HLN:
$$DM^* = DM 	imes \l\left[ \f\frac{T + 1 - 2h + h(h-1)/T}{T} 
\right]^{1/2}$$
Đề án áp dụng 100% chuẩn hiệu chỉnh $DM^*$ này.

---

## 3. Luận Giải Quy Ước Dấu Của Thống Kê DM (Bảng 4.7)

Trong Báo cáo Luận văn (Bảng 4.7), người đọc có thể thấy sự xuất hiện của dấu âm:
- Khi định nghĩa hàm tổn thất: $d_t = |e_{\text{Proposed}}| - |e_{\text{Baseline}}|$:
  - Nếu mô hình đề xuất có sai số nhỏ hơn mô hình cơ sở, $\b\bar{d}$ sẽ mang **giá trị âm**.
  - Do đó, **thống kê $DM < 0$ và mang dấu âm lớn phản ánh mô hình đề xuất vượt trội có ý nghĩa thống kê**!

### Kết quả thực nghiệm 4 cặp đối kháng chính:
1. **Weighted Ensemble vs Persistence Baseline:** $DM^* = -8.452$, $p < 0.0001 \implies$ Vượt trội tuyệt đối.
2. **GRU (15m) vs Persistence Baseline:** $DM^* = -5.891$, $p < 0.0001 \implies$ Phá bẫy tự tương quan thành công.
3. **LightGBM vs ARIMA:** $DM^* = -4.120$, $p = 0.0002 \implies$ Học máy phi tuyến đánh bại thống kê tuyến tính.
4. **Weighted Ensemble vs LightGBM Đơn Lẻ:** $DM^* = -2.134$, $p = 0.033 < 0.05 \implies$ Học kết hợp ensemble mang lại giá trị gia tăng thực sự.

---
*Liên kết liên quan: [[01_Pipeline_7_Steps]] | [[03_Sweet_Spot_30m_Ensemble]] | [[01_Metrics_Standard_MASE_over_RMSE]] | [[01_Master_Defense_QnA_Hoi_Dong]]*\n