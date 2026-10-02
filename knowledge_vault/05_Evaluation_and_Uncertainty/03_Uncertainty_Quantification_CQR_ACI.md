---
title: "Định lượng độ bất định Conformal Prediction (CQR) & Tương thích động ACI"
tags: [uncertainty, conformal-prediction, cqr, aci, prediction-intervals, coverage, table-46]
aliases: ["Định lượng bất định", "CQR và ACI", "Conformal Prediction", "Bảng 4.6"]
domain: evaluation
created: "2026-10-02"
status: completed
---

# 🎯 Định Lượng Độ Bất Định Conformal Prediction (CQR) & Tương Thích Động ACI

## 1. Giới Hạn Của Dự Báo Điểm & Khoảng Tin Cậy Gauss Truyền Thống
Dự báo điểm (Point Forecast) chỉ đưa ra một con số duy nhất (ví dụ: $PM_{2.5} = 25\,\mu g/m^3$), không cho phép nhà quản lý biết mức độ rủi ro hay độ biến động cực đoan có thể xảy ra.
- Khoảng tin cậy Gauss truyền thống $\hat{y} \pm 1.96 \hat{\sigma}$ hoàn toàn sụp đổ vì:
  1. Phân phối $PM_{2.5}$ bị lệch phải mạnh và đuôi dày ($p < 0.001$).
  2. Phần dư vi phạm giả định độc lập qua kiểm định Ljung-Box ($p < 0.05$).

---

## 2. Conformal Quantile Regression (CQR Tĩnh)
Đề án triển khai **CQR (Romano, Patterson & Candès, 2019)**:
- Huấn luyện 2 mô hình phân vị Pinball Loss: $\hat{q}_{\alpha/2}$ ($5\%$) và $\hat{q}_{1-\alpha/2}$ ($95\%$) để nhắm tới khoảng tin cậy danh định **$90\%$**.
- Sử dụng tập hiệu chuẩn (Calibration Set) để tính toán điểm không tương đồng (Nonconformity Score):
  $$E_i = \max(\hat{q}_{\alpha/2}(x_i) - y_i, \, y_i - \hat{q}_{1-\alpha/2}(x_i))$$
- Khoảng dự báo hiệu chuẩn cuối cùng:
  $$C(x) = [\hat{q}_{\alpha/2}(x) - Q_{1-\alpha}(E), \,\, \hat{q}_{1-\alpha/2}(x) + Q_{1-\alpha}(E)]$$
- **Đặc tính:** Bảo đảm độ phủ toán học mà không cần bất kỳ giả định nào về phân phối của dữ liệu (Distribution-free).

---

## 3. Bẫy Trôi Phân Phối (Concept Drift) & Đột Phá ACI Thích Ứng

### Hiện tượng sụt giảm độ phủ của CQR Tĩnh:
CQR lý thuyết dựa trên giả định các mẫu dữ liệu có tính chất **trao đổi được (Exchangeability)**. Tuy nhiên, chuỗi thời gian thực tế tại Sa Đéc thường xuyên đối mặt với **trôi dạt phân phối (Concept Drift)** giữa các mùa thời tiết:
- Trên tập Anchor Test, độ phủ thực nghiệm của CQR Tĩnh bị sụt giảm từ danh định $90\%$ xuống chỉ còn **$76.0\% - 80.5\%$**.

### Giải pháp Adaptive Conformal Inference (ACI - Gibbs & Candès, 2021):
ACI cập nhật động ngưỡng phủ sai số trực tuyến theo công thức phản hồi:
$$\alpha_{t+1} = \alpha_t + \gamma (\alpha - \text{err}_t)$$
với:
- $\text{err}_t = 1$ nếu $y_t 
otin C_t(x_t)$ (bị lọt ra ngoài khoảng dự báo).
- $\text{err}_t = 0$ nếu $y_t \in C_t(x_t)$ (nằm trọn trong khoảng dự báo).
- $\gamma = 0.01$ là tốc độ học thích ứng.

### Kết quả thực nghiệm đối chứng (Bảng 4.6 Báo cáo Đề án):
| Phương pháp | Tầm dự báo 1h | Tầm dự báo 6h | Tầm dự báo 24h | Đánh giá học thuật |
|---|---|---|---|---|
| **Quantile Regression Gốc** | $86.2\%$ | $84.1\%$ | $81.5\%$ | Dưới chuẩn danh định |
| **CQR Tĩnh (Static CQR)** | $80.5\%$ | $78.2\%$ | $76.0\%$ | Bị sụt giảm do concept drift |
| **ACI Thích Ứng ($\gamma=0.01$)** | **$89.6\%$** | **$89.4\%$** | **$90.2\%$** | **Khôi phục hoàn hảo chuẩn danh định 90%** |

---
*Liên kết liên quan: [[02_Stationarity_ADF_KPSS]] | [[01_Benchmark_Models_Overview]] | [[01_Master_Defense_QnA_Hoi_Dong]] | [[02_Slides_Narrative_and_Key_Arguments]]*\n