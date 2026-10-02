---
title: "Đặc thù chu kỳ ngày đêm vi khí hậu Sa Đéc & Tác nhân giao thông thủy bộ, đốt rơm rạ"
tags: [explainability, sa-dec, diurnal-cycle, temperature-inversion, emission-sources, dong-thap]
aliases: ["Chu kỳ ngày đêm Sa Đéc", "Vi khí hậu Sa Đéc", "Sa Dec Context"]
domain: explainability
created: "2026-10-02"
status: completed
---

# 🌅 Đặc Thù Chu Kỳ Ngày Đêm Vi Khí Hậu Sa Đéc & Các Tác Nhân Phát Thải

## 1. Chu Kỳ Nhật Triều Nồng Độ Bụi Mịn (Diurnal Cycle)
Phân tích dữ liệu thực nghiệm 38 tháng tại Sa Đéc chỉ ra quy luật biến thiên ngày đêm rất đặc trưng:

```
Nồng độ PM2.5 (µg/m³)
  ^
30|        /\ (Đỉnh sáng 06:00 - 08:00)
20|       /  \                                  /\ (Đỉnh tối 18:00 - 20:00)
10|______/    \________________________________/  \________
 0+--------------------------------------------------------> Giờ trong ngày
  00:00      06:00            12:00            18:00       24:00
                              (Đáy trưa 12:00)
```

### Các mốc thời gian then chốt:
1. **Đỉnh cực đại buổi sáng (06:00 – 08:00):**
   - Đạt mức nồng độ trung bình cao nhất trong ngày ($22 - 38\,\mu g/m^3$).
   - **Nguyên nhân kép:**
     - Giờ cao điểm giao thông: chợ hoa Sa Đéc, học sinh đi học, lưu lượng xe máy và xe tải vận chuyển nông sản.
     - **Hiện tượng nghịch nhiệt bề mặt (Surface Temperature Inversion):** Sau một đêm dài tỏa nhiệt bức xạ, mặt đất lạnh hơn các tầng không khí bên trên, tạo thành một "chiếc nắp nồi" giữ chặt toàn bộ khói bụi phát thải sát mặt đất.
2. **Đáy cực tiểu buổi trưa (12:00 – 14:00):**
   - Nồng độ giảm xuống mức thấp nhất trong ngày ($8 - 12\,\mu g/m^3$).
   - **Nguyên nhân:** Mặt trời chiếu xạ mạnh, nhiệt độ tăng cao làm bung lớp nghịch nhiệt; đối lưu nhiệt theo phương thẳng đứng phát triển mạnh mẽ, nâng chiều cao lớp biên khí quyển lên $1.000 - 1.500m$, giúp bụi mịn khuếch tán nhanh chóng.
3. **Đỉnh phụ buổi tối (18:00 – 20:00):**
   - Nồng độ tăng nhẹ trở lại do sinh hoạt gia đình, khói bếp, lưu lượng ghe tàu đường thủy vận tải hàng hóa đêm trên sông Tiền và kênh xáng Sa Đéc.

---

## 2. Các Nguồn Phát Thải Đặc Thù Địa Bàn Sa Đéc (Đồng Tháp)
Khác với các đô thị công nghiệp nặng ở miền Bắc (Hà Nội, Quảng Ninh):
- **Giao thông thủy nội địa:** Sa Đéc là trung tâm đầu mối sông nước miền Tây. Khí thải từ động cơ diesel của các ghe tải chở lúa gạo, sà lan cát đá trên sông Tiền đóng góp tỷ trọng lớn các hạt muội than carbon đen.
- **Làng hoa kiểng Sa Đéc:** Các hoạt động đốt bao bì, vỏ trấu ủ phân, chăm sóc hoa màu theo đợt vụ Tết (tháng 11 đến tháng 1 âm lịch).
- **Tập quán đốt rơm rạ theo vụ mùa lúa:** Các đợt bùng phát ô nhiễm đột biến ($> 60\,\mu g/m^3$) thường xuất hiện vào thời điểm thu hoạch lúa Đông Xuân (tháng 3 – tháng 4) và Hè Thu (tháng 8 – tháng 9). Khói đốt đồng theo hướng gió chướng tràn vào nội ô đô thị.

---
*Liên kết liên quan: [[03_Outlier_Trap_and_Domain_Bounds]] | [[01_Feature_Store_119_Features]] | [[02_Threshold_Tipping_Point_14_17]] | [[01_Master_Defense_QnA_Hoi_Dong]]*\n