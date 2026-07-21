# 📝 Academic Writing Guide — Luận Văn Thạc Sĩ (IEEE Format, QĐ 1799)

> **Mục đích**: Hướng dẫn agent viết, chỉnh sửa, và review luận văn thạc sĩ theo đúng phong cách học thuật formal Vietnamese, tuân thủ chuẩn IEEE và Quy định 1799 của Trường Đại học Cần Thơ.
> **Đọc khi**: Anh Trí yêu cầu viết, sửa, hoặc review nội dung luận văn.

---

## 1. Nguyên tắc Viết Học Thuật

### 1.1 Phong cách ngôn ngữ
- **Formal Vietnamese**: Dùng ngôi thứ ba hoặc bị động. VD: "Nghiên cứu này đề xuất..." thay vì "Em đề xuất...".
- **Chính xác**: Mọi thuật ngữ kỹ thuật viết bằng tiếng Anh kèm giải thích tiếng Việt ở lần đầu xuất hiện. VD: "Rò rỉ dữ liệu (data leakage)".
- **Ngắn gọn**: Tránh lặp ý. Mỗi đoạn văn chỉ truyền tải 1 ý chính.
- **Khách quan**: Không dùng ngôn ngữ cảm tính ("rất tốt", "cực kỳ chính xác"). Thay bằng số liệu cụ thể.

### 1.2 Cấu trúc câu
- Ưu tiên câu chủ động khi mô tả phương pháp: "Nghiên cứu sử dụng mô hình GRU..."
- Dùng câu bị động khi trình bày kết quả: "Kết quả được đánh giá bằng chỉ số MASE..."
- Tránh câu quá dài (>40 từ). Tách thành 2 câu nếu cần.

---

## 2. Chuẩn Trích Dẫn IEEE

### 2.1 Trong văn bản (In-text citation)
- Dùng số trong ngoặc vuông: `[1]`, `[2, 3]`, `[4]–[7]`
- Đặt trước dấu chấm cuối câu: "...đạt MASE = 0.727 [13]."
- Khi đề cập tác giả: "Lim et al. [13] đề xuất kiến trúc TFT..."

### 2.2 Danh mục tài liệu tham khảo
```
[1] T. H. Nguyen, "Title of paper," Journal Name, vol. X, no. Y, pp. Z–W, Month Year.
[2] A. Author and B. Author, "Conference paper title," in Proc. Conf. Name, City, Country, Year, pp. 1–10.
```

### 2.3 Quy tắc quan trọng
- Mỗi tài liệu tham khảo PHẢI được trích dẫn ít nhất 1 lần trong văn bản.
- Sắp xếp theo thứ tự xuất hiện (KHÔNG theo alphabet).
- Tên tạp chí viết tắt theo chuẩn IEEE (VD: "IEEE Trans. Neural Netw." thay vì viết đầy đủ).

---

## 3. Trình Bày Bảng & Biểu Đồ

### 3.1 Bảng (Table)
- Tiêu đề bảng đặt **TRÊN** bảng: "Bảng 3.1: So sánh hiệu suất các mô hình dự báo PM2.5"
- Đánh số theo chương: Bảng 3.1, Bảng 3.2, ...
- Mọi bảng PHẢI được tham chiếu trong văn bản: "Kết quả trình bày tại Bảng 3.1..."
- Căn phải (right-align) cho cột số liệu, căn trái cho cột text.
- **Bold** giá trị tốt nhất trong mỗi cột so sánh.

### 3.2 Hình (Figure)
- Tiêu đề hình đặt **DƯỚI** hình: "Hình 2.3: Kiến trúc pipeline xử lý dữ liệu"
- Đánh số theo chương: Hình 2.1, Hình 2.2, ...
- Mọi hình PHẢI được tham chiếu trong văn bản.
- Biểu đồ PHẢI có nhãn trục (axis labels) rõ ràng, đơn vị đo.
- Xuất hình ở độ phân giải ≥300 DPI cho in ấn.

### 3.3 Phương trình (Equation)
- Đánh số phương trình ở lề phải: (3.1), (3.2), ...
- Định nghĩa tất cả biến ngay sau phương trình.

---

## 4. Cấu Trúc Luận Văn (QĐ 1799 CTU)

```
Chương 1: Giới thiệu
├── 1.1 Đặt vấn đề
├── 1.2 Mục tiêu nghiên cứu
├── 1.3 Phạm vi nghiên cứu
└── 1.4 Cấu trúc luận văn

Chương 2: Cơ sở lý thuyết và Tổng quan nghiên cứu
├── 2.1 Các khái niệm cơ bản
├── 2.2 Các nghiên cứu liên quan
└── 2.3 Phương pháp đề xuất

Chương 3: Phương pháp nghiên cứu
├── 3.1 Thu thập và tiền xử lý dữ liệu
├── 3.2 Kỹ thuật feature engineering
├── 3.3 Các mô hình dự báo
└── 3.4 Đánh giá và so sánh

Chương 4: Kết quả và Thảo luận
├── 4.1 Kết quả thí nghiệm
├── 4.2 Phân tích và thảo luận
└── 4.3 So sánh với nghiên cứu trước

Chương 5: Kết luận và Hướng phát triển
├── 5.1 Kết luận
├── 5.2 Đóng góp
└── 5.3 Hạn chế và hướng phát triển
```

---

## 5. Checklist Review Luận Văn

### Nội dung
- [ ] Mọi số liệu đều có nguồn (từ code chạy thực tế hoặc tài liệu tham khảo)
- [ ] Mỗi bảng/hình đều được tham chiếu trong văn bản
- [ ] Kết quả tốt nhất được **bold** trong bảng so sánh
- [ ] Giải thích TẠI SAO kết quả tốt/không tốt, không chỉ liệt kê số

### Hình thức
- [ ] Trích dẫn IEEE nhất quán xuyên suốt
- [ ] Đánh số bảng/hình theo chương
- [ ] Font chữ nhất quán (Times New Roman 13pt hoặc theo QĐ 1799)
- [ ] Lề trang: Trái 3.5cm, Phải 2cm, Trên 2cm, Dưới 2cm
- [ ] Giãn dòng 1.5

### Logic
- [ ] Mỗi chương có mở đầu tóm tắt nội dung sẽ trình bày
- [ ] Mỗi chương có kết luận chuyển tiếp sang chương tiếp theo
- [ ] Không có nhảy logic giữa các phần
