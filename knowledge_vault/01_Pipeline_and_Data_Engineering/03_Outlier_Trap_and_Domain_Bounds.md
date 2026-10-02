---
title: "Bẫy xóa ngoại lai (Outlier Trap) & Bảo tồn đỉnh ô nhiễm bằng Domain Bounds"
tags: [time-series, outliers, iqr-trap, domain-bounds, s-esd, fat-tailed, who-aqi]
aliases: ["Bẫy IQR 3.0", "Outlier Removal Trap", "Domain Bounds [0, 500]", "Fat-tailed"]
domain: data-engineering
created: "2026-10-02"
status: completed
---

# 🚨 Bẫy Xóa Ngoại Lai (Outlier Trap) & Bảo Tồn Đỉnh Ô Nhiễm Bằng Domain Bounds

## 1. Bẫy Thống Kê IQR 3.0 Cổ Điển (The Outlier Removal Trap)
Trong tiền xử lý dữ liệu chuẩn, phương pháp loại bỏ ngoại lai phổ biến nhất là quy tắc Tukey IQR:
$$\text{Upper Threshold} = Q_3 + c 	imes IQR$$
với $c = 1.5$ (ngoại lai nhẹ) hoặc $c = 3.0$ (ngoại lai cực đoan).

Trong các phiên bản thử nghiệm ban đầu (Ablation Study v10):
- Ngưỡng $Q_3 + 3.0 	imes IQR$ tương ứng với mức nồng độ xấp xỉ **$54\,\mu g/m^3$**.
- Thuật toán IQR đã tự động "gọt bỏ" (clip/drop) **66 đỉnh ô nhiễm thực tế** có nồng độ từ $55$ đến hơn $120\,\mu g/m^3$.

---

## 2. Ảo Tưởng Về Độ Chính Xác (False Sense of Accuracy)

Khi gọt bỏ 66 đỉnh ô nhiễm:
| Tiêu chí | Sử dụng IQR 3.0 (v10 Ablation) | Giữ nguyên bằng Domain Bounds (v9 Chuẩn) |
|---|---|---|
| **RMSE tập Validation** | Rất thấp ($2.12\,\mu g/m^3$) — Trông rất "đẹp" | Cao hơn ($3.84\,\mu g/m^3$) |
| **Độ bao phủ đỉnh ô nhiễm** | **0% (Mù hoàn toàn trước đỉnh ô nhiễm)** | **100% (Học được cơ chế tích tụ)** |
| **Cảnh báo sớm vượt ngưỡng WHO** | Thất bại nặng nề ($F_1 < 0.35$) | Thành công ($F_1 = 0.782$ tại ngưỡng $45\,\mu g/m^3$) |
| **Bản chất khoa học** | Tự lừa dối bằng dữ liệu đã bị gọt mượt | Phản ánh đúng thực tế khí quyển |

> **Bài học xương máu:** Việc xóa ngoại lai cơ học bằng IQR tạo ra **ảo tưởng về độ chính xác (False Sense of Accuracy)**. Mô hình chỉ học được trạng thái khí quyển bình yên và hoàn toàn bất lực khi xảy ra biến cố ô nhiễm khẩn cấp.

---

## 3. Phân Phối Đuôi Nặng (Fat-Tailed Distribution) Của PM2.5
Dữ liệu $PM_{2.5}$ không bao giờ tuân theo phân phối chuẩn Gauss (Gaussian Bell Curve):
- **Độ lệch (Skewness):** $2.0046$ (Lệch phải mạnh, nồng độ thấp xuất hiện phổ biến, nồng độ cao tạo vệt dài).
- **Độ nhọn (Kurtosis):** $6.1458$ (Phân phối đuôi dày nhọn - Leptokurtic).

Các đỉnh ô nhiễm cao chính là **tín hiệu quý giá nhất** phản ánh hiện tượng nghịch nhiệt tầng thấp, khói bụi mùa gặt hoặc kẹt xe đường thủy, hoàn toàn không phải lỗi thiết bị.

---

## 4. Giải Pháp: Domain Bounds Vật Lý [0, 500] µg/m³
Thay vì dùng ngưỡng thống kê thuần túy, đề án áp dụng **Domain Bounds** dựa trên cơ sở vật lý và quy chuẩn môi trường:
1. **Ngưỡng dưới:** $0\,\mu g/m^3$ (nồng độ khối lượng không thể âm; giá trị âm là lỗi trôi điểm 0 của sensor $	o$ xử lý về 0 hoặc NaN).
2. **Ngưỡng trên:** $500\,\mu g/m^3$ (ngưỡng tối đa của thang đo Chỉ số Chất lượng Không khí Quốc tế WHO AQI và US-EPA).
3. **Phát hiện lỗi phần cứng:** Sử dụng thuật toán **S-ESD (Seasonal Hybrid Extreme Studentized Deviate)** chỉ nhằm bóc tách các trường hợp cảm biến bị kẹt cứng một giá trị cố định (flatline) hoặc xung điện áp vô lý ($> 500\,\mu g/m^3$).

---
*Liên kết liên quan: [[01_Pipeline_7_Steps]] | [[02_Stationarity_ADF_KPSS]] | [[01_Metrics_Standard_MASE_over_RMSE]] | [[01_Master_Defense_QnA_Hoi_Dong]]*\n