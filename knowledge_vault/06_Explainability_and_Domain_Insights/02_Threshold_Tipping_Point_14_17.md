---
title: "Khám phá điểm chuyển pha khí quyển 14–17 µg/m³ & Ngưỡng bùng phát ô nhiễm"
tags: [explainability, tipping-point, threshold-effect, shap-dependence, atmospheric-physics, who-threshold]
aliases: ["Điểm chuyển pha 14-17", "Tipping Point", "Ngưỡng bùng phát ô nhiễm", "Hình 4.8"]
domain: explainability
created: "2026-10-02"
status: completed
---

# ⚠️ Khám Phá Điểm Chuyển Pha Khí Quyển 14–17 µg/m³ & Ngưỡng Bùng Phát Ô Nhiễm

## 1. Phát Hiện Từ Biểu Đồ SHAP Dependence Plot (Hình 4.8)
Khi phân tích biểu đồ phụ thuộc riêng phần (SHAP Dependence Plot) của biến nồng độ nền trung bình 24 giờ (`pm25_roll_24h_mean`), nghiên cứu phát hiện một **đường cong phi tuyến tính gấp khúc rõ rệt tại dải giá trị $14 - 17\,\mu g/m^3$**.

```
SHAP Value (Tác động đẩy dự báo tăng)
   ^
+8 |                                                / (Vùng bùng phát tích tụ phi tuyến)
+4 |                                      _________/
 0 | ------------------------------------/ (Điểm chuyển pha 14 - 17 µg/m³)
-4 | \_________________ (Vùng tự làm sạch)
   +----------------------------------------------------> pm25_roll_24h_mean (µg/m³)
     0                10       14   17       25       35
```

---

## 2. Ma Trận 3 Phân Vùng Trạng Thái Khí Quyển Sa Đéc

| Phân vùng | Dải nồng độ nền | Giá trị SHAP | Cơ chế động lực học khí quyển Sa Đéc | Hành động khuyến nghị điều hành đô thị |
|---|---|---|---|---|
| **Vùng 1: Tự Làm Sạch** | $< 14\,\mu g/m^3$ | Mang giá trị Âm ($-2$ đến $-5$) | Năng lực tự phát tán và đối lưu khí quyển chiếm ưu thế; ô nhiễm có xu hướng tự triệt tiêu về mức sạch. | Trạng thái bình thường; duy trì các hoạt động sản xuất, giao thông thông lệ. |
| **Vùng 2: Chuyển Pha Nhạy Cảm** | **$14 - 17\,\mu g/m^3$** (Trùng khớp chuẩn WHO 24h: $15\,\mu g/m^3$) | Vượt qua trục 0 (Chuyển từ Âm sang Dương) | **Khí quyển chạm ngưỡng bão hòa tự làm sạch**. Hệ thống vi khí hậu trở nên cực kỳ nhạy cảm; một xung phát thải nhỏ sẽ kích hoạt bùng phát ô nhiễm. | **Phát cảnh báo mức 1 (Màu Vàng)**; khuyến nghị nhóm nhạy cảm (trẻ em, người già) hạn chế hoạt động ngoài trời. |
| **Vùng 3: Bùng Phát Phi Tuyến** | $> 17\,\mu g/m^3$ | Dương dốc đứng ($+4$ đến $+10$) | Hiệu ứng bẫy nhiệt và lắng đọng hạt mịn; nồng độ bụi tăng theo cấp số nhân do bụi mới tích tụ đè lên lớp bụi cũ không thoát ra được. | **Kích hoạt phản ứng mức 2 (Màu Cam/Đỏ)**: điều tiết luồng tàu xe qua sông Sa Đéc, phun sương tưới đường tại các nút giao. |

---

## 3. Đóng Góp Học Thuật & Trả Lời Câu Hỏi Nghiên Cứu CH4
Phát hiện về ngưỡng chuyển pha $14 - 17\,\mu g/m^3$ là lời giải đáp học thuật đanh thép cho Câu hỏi nghiên cứu CH4 của đề án:
> *Mô hình Machine Learning không chỉ đưa ra con số dự báo vô hồn, mà đã thành công trong việc trích xuất và chứng minh bằng thực nghiệm ngưỡng giới hạn chịu tải môi trường của một huyện nông nghiệp - công nghiệp hóa ở Nam Bộ.*

Con số $15\,\mu g/m^3$ vốn được Tổ chức Y tế Thế giới (WHO) khuyến cáo dựa trên các nghiên cứu dịch tễ học y tế cộng đồng; đề án này đã chứng minh rằng **ngay cả từ góc độ vật lý vi khí hậu, mốc $15\,\mu g/m^3$ cũng chính là điểm nút biến chuyển trạng thái cân bằng tự nhiên.**

---
*Liên kết liên quan: [[01_Tree_SHAP_vs_Permutation_Importance]] | [[03_Diurnal_Cycle_and_Sa_Dec_Context]] | [[01_Master_Defense_QnA_Hoi_Dong]] | [[02_Slides_Narrative_and_Key_Arguments]]*\n