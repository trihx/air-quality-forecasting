---
title: "Kỷ luật shift(1) chống rò rỉ dữ liệu & Vụ việc bóc tách R²=1.000 ảo"
tags: [time-series, data-leakage, anti-leakage, shift-1, data-integrity, r2-trap]
aliases: ["Kỷ luật shift(1)", "Anti-Leakage Discipline", "R² = 1.0 Case Study"]
domain: data-engineering
created: "2026-10-02"
status: completed
---

# 🛡️ Kỷ Luật shift(1) Chống Rò Rỉ Dữ Liệu & Vụ Việc Bóc Tách R²=1.000 Ảo

## 1. Vụ Việc Bóc Tách $R^2 = 1.000$ Ảo (The R²=1.000 Illusion)
Trong giai đoạn đầu phát triển module Kỹ nghệ đặc trưng:
- Một mô hình hồi quy thử nghiệm ghi nhận kết quả hoàn hảo bất thường: **$R^2 = 1.0000$** trên cả tập Train và Validation, với sai số $RMSE \approx 0.0001$.
- Trong nghiên cứu khoa học dữ liệu, kết quả "quá hoàn hảo" luôn là **cờ đỏ (Red Flag)** báo hiệu hiện tượng **Rò Rỉ Dữ Liệu (Lookahead Bias / Data Leakage)**.

### Bóc tách nguyên nhân gốc rễ:
Mã nguồn ban đầu tính toán đặc trưng cửa sổ trượt như sau:
```python
# MÃ NGUỒN CŨ GÂY RÒ RỈ NGHIÊM TRỌNG:
df["pm25_roll_mean_6h"] = df["pm25"].rolling(window=6).mean()
```
Tại thời điểm $t$, giá trị của `pm25_roll_mean_6h` được tính từ:
$$\f\frac{1}{6} \l\left( y_{t-5} + y_{t-4} + y_{t-3} + y_{t-2} + y_{t-1} + \mathbf{y_t} 
\right)$$
Nghĩa là **giá trị mục tiêu $y_t$ đã vô tình bị nhét vào đặc trưng đầu vào**! Khi mô hình dự báo nồng độ tại thời điểm $t$, nó chỉ cần giải phương trình đại số để suy ngược $y_t$ từ đặc trưng trượt, tạo ra điểm số $R^2 = 1.000$ hoàn toàn giả tạo.

---

## 2. Kỷ Luật Bắt Buộc `shift(1)` (The Invariant Rule)
Để triệt tiêu vĩnh viễn lookahead bias, đề án thiết lập kỷ luật bất biến:
> **100% các biến trễ (Lag), biến cửa sổ trượt (Rolling statistics), giá trị làm mượt số mũ (EWMA), và sai phân (Diff/Pct_change) BẮT BUỘC PHẢI QUA HÀM `.shift(1)` TRƯỚC KHI ĐƯỢC TÍNH TOÁN HOẶC ĐƯA VÀO FEATURE STORE.**

```python
# MÃ NGUỒN CHUẨN THUẬT TOÁN KHOA HỌC:
df["pm25_roll_mean_6h"] = df["pm25"].shift(1).rolling(window=6).mean()
```
Khi đó, cửa sổ tính toán tại thời điểm $t$ chỉ chứa:
$$\{ y_{t-1}, y_{t-2}, y_{t-3}, y_{t-4}, y_{t-5}, y_{t-6} \}$$
hoàn toàn độc lập với giá trị mục tiêu $y_t$.

### Kết quả sau khi khắc phục:
- Hệ số $R^2$ ảo $1.000$ sụp đổ về giá trị thực chất: **$0.267$** ở tầm ngắn và phản ánh đúng độ phức tạp của bài toán.
- Mô hình buộc phải học các quy luật khí tượng và động lực học vi khí hậu thay vì học mẹo giải mã đại số.

---

## 3. Bộ 5 Quy Tắc Bất Biến Chống Rò Rỉ Trong Đề Án
1. **Shift-First Invariant:** `.shift(1)` trước mọi phép toán rolling, ewm, diff trên biến mục tiêu.
2. **Scaler Isolation:** Mọi bộ chuẩn hóa (`StandardScaler`, `MinMaxScaler`) chỉ được `fit()` trên tập Train; tập Val và Test chỉ được `transform()`.
3. **Past-Only Imputation Donors:** Thuật toán KNN phục hồi dữ liệu chỉ được tìm láng giềng ở các mốc thời gian quá khứ ($t' < t$).
4. **Anchor Temporal Split:** Cấm tuyệt đối random shuffling; giữ nguyên trật tự thời gian.
5. **Automated Leakage Gate:** Toàn bộ 51 bài test trong nhóm Anti-Leakage thuộc Automated Test Suite tự động chạy kiểm thử trước mỗi commit.

---
*Liên kết liên quan: [[01_Pipeline_7_Steps]] | [[02_Temporal_Split_and_Test_on_Real_Only]] | [[01_Feature_Store_119_Features]] | [[01_Master_Defense_QnA_Hoi_Dong]]*\n