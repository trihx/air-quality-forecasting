---
title: "Sợi chỉ đỏ, 24 slides thuyết trình & Tuyên bố học thuật đanh thép"
tags: [defense, slides, narrative, golden-minutes, 24-slides, speaker-notes]
aliases: ["Cẩm nang 24 slides", "Sợi chỉ đỏ đề án", "Speaker Notes"]
domain: defense
created: "2026-10-02"
status: completed
---

# 🎯 Sợi Chỉ Đỏ, 24 Slides Thuyết Trình & Tuyên Bố Học Thuật Đanh Thép

## 1. Sợi Chỉ Đỏ Của Đề Án (The Central Narrative Thread)
> *"Từ nghịch lý chuỗi dữ liệu IoT môi trường thưa thớt (74% khuyết thiếu) tại một huyện Nam Bộ, đề án xây dựng một hệ thống Business Intelligence dự báo nồng độ bụi mịn PM2.5 đa độ phân giải hoàn chỉnh: thiết lập kỷ luật kỹ nghệ dữ liệu chống rò rỉ tuyệt đối, giải mã bẫy tự tương quan, định lượng độ bất định thích ứng 90.2% và khám phá điểm chuyển pha khí quyển phục vụ điều hành đô thị thông minh."*

---

## 2. Bản Đồ Phân Bổ Thời Gian Vàng 18 Phút Thuyết Trình

```
[00:00 - 03:00] Slide 01-04: Mở đầu, Bối cảnh Sa Đéc, 4 Câu hỏi nghiên cứu, Kiến trúc BI 3 tầng.
[03:00 - 08:00] Slide 05-09: Kỹ nghệ dữ liệu: Drop 19.810h, Bẫy IQR 3.0, Kỷ luật shift(1), Kiểm định dừng.
[08:00 - 14:00] Slide 10-15: Điểm ngọt 30m, GRU 15m phá bẫy 1h, Kiểm định Diebold-Mariano, XAI Tipping point.
[14:00 - 17:00] Slide 16-19: Conformal Prediction ACI 90%, Cảnh báo F1=0.782, Vận hành Dashboard & 248 tests.
[17:00 - 18:00] Slide 20-24: Tuyên bố kết luận & Ứng phó 4 bẫy phản biện chuyên sâu.
```

---

## 3. Tóm Lược 24 Slides Báo Cáo

1. **Slide 01 — Bìa Báo Cáo:** Giới thiệu tên đề án, HVTH Hoàng Xuân Trí, CBHD TS. Nguyễn Minh Khiêm, ngành HTTT.
2. **Slide 02 — Bối Cảnh Thực Tiễn:** Nghịch lý ô nhiễm bụi mịn tại Sa Đéc và thực trạng trạm IoT thưa thớt.
3. **Slide 03 — 4 Câu Hỏi Nghiên Cứu:** CH1 (Xử lý thưa thớt), CH2 (Đa độ phân giải), CH3 (Định lượng bất định), CH4 (Khám phá XAI).
4. **Slide 04 — Quy Trình 7 Bước:** Dòng chảy Sankey 7 bước và kiến trúc phần mềm BI 3 tầng.
5. **Slide 05 — Khảo Sát Dữ Liệu IoT Sa Đéc:** 38 tháng, 209.594 bản ghi, phân phối fat-tailed ($S=2.00, K=6.15$).
6. **Slide 06 — Nội Suy Phân Tầng:** Akima Spline $\le 6$h, KNN $6-24$h, dũng cảm drop 19.810h.
7. **Slide 07 — Bẫy Xóa Ngoại Lai IQR 3.0:** Bẫy gọt nhầm 66 đỉnh ô nhiễm và bảo tồn bằng Domain Bounds [0, 500].
8. **Slide 08 — Kỹ Nghệ Đặc Trưng Chống Rò Rỉ:** Kho 119 đặc trưng, kỷ luật `shift(1)` bóc tách $R^2=1.000$ ảo.
9. **Slide 09 — Kiểm Định Tính Dừng:** ADF ($p<0.001$) kết hợp KPSS ($p=0.01$) $\implies$ Dừng cục bộ theo mùa.
10. **Slide 10 — Anchor Split 80:10:10 & MASE:** Kỷ luật Test on Real Only, mẫu số chuẩn hóa $1.821\,\mu g/m^3$.
11. **Slide 11 — Bẫy Tự Tương Quan 1h & Đột Phá GRU 15m:** GRU 15m đạt $MASE = 0.667$, đánh bại Persistence.
12. **Slide 12 — Điểm Ngọt Pareto 30 Phút:** Weighted Ensemble thống trị toàn diện (MASE 0.382 ở 6h, 0.469 ở 24h).
13. **Slide 13 — Kiểm Định Diebold-Mariano:** Hiệu chỉnh HLN, $DM < 0$ ($p < 0.001$) khẳng định vượt trội thống kê.
14. **Slide 14 — Khám Phá XAI Điểm Chuyển Pha 14-17 µg/m³:** Phát hiện ngưỡng tự làm sạch và bùng phát ô nhiễm.
15. **Slide 15 — Đối Chứng Chéo XAI:** Top-2 biến hoàn toàn đồng thuận giữa Tree SHAP và Permutation.
16. **Slide 16 — Định Lượng Độ Bất Định CQR & ACI:** ACI thích ứng khôi phục độ phủ danh định $90.2\%$.
17. **Slide 17 — Đánh Giá Cảnh Báo Sớm:** Đạt $F_1 = 0.782$ tại ngưỡng cảnh báo $45\,\mu g/m^3$.
18. **Slide 18 — Đóng Góp Khoa Học & Ma Trận Điều Hành:** 3 mức phản ứng đô thị tương ứng 3 vùng khí quyển.
19. **Slide 19 — Hệ Thống Phần Mềm BI Vận Hành:** Dashboard tương tác, 248 tests tự động, sẵn sàng triển khai.
20. **Slide 20 — Kết Luận & Sẵn Sàng Trả Lời:** Tóm lược 4 đóng góp và mở phiên chất vấn.
21. **Slide 21 (Backup Q&A 1):** Giải trình dấu thống kê Diebold-Mariano âm.
22. **Slide 22 (Backup Q&A 2):** Tại sao bắt buộc drop 19.810 giờ khuyết dài.
23. **Slide 23 (Backup Q&A 3):** Tại sao $R^2$ ngoài mẫu có thể âm.
24. **Slide 24 (Backup Q&A 4):** Bẫy ngoại lai IQR 3.0 và vai trò Domain Bounds.

---

## 4. Bốn Tuyên Bố Học Thuật Đanh Thép
1. **Về Dữ Liệu:** *"Đề án thà chịu mất mát mẫu (loại bỏ 19.810 giờ) còn hơn đưa rác vào mô hình để sinh ra kết quả giả mạo."*
2. **Về Liêm Chính:** *"Kỷ luật shift(1) bảo vệ nghiên cứu khỏi ảo tưởng chính xác $R^2=1.000$, chỉ giữ lại giá trị học tập thực chất."*
3. **Về Hiệu Năng:** *"Độ phân giải 30 phút là điểm ngọt Pareto tối ưu, và GRU 15m đã chứng minh khả năng vượt qua đối thủ sừng sỏ Persistence."*
4. **Về Ứng Dụng:** *"Khoảng dự báo bất định ACI 90.2% cùng ngưỡng bùng phát 14–17 µg/m³ cung cấp công cụ điều hành tin cậy cho chính quyền Sa Đéc."*

---
*Liên kết liên quan: [[00_Index_MOC]] | [[01_Master_Defense_QnA_Hoi_Dong]]*\n