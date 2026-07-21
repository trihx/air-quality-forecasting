# 🔥 L1: Project Hot Memory (MEMORY_HOT.md)

> **Trạng thái hiện tại & Mục tiêu nóng của dự án (Cập nhật: 2026-07-22)**
> **Quy tắc:** Đọc mỗi khi bắt đầu phiên làm việc mới. Giữ dưới 80 dòng.

---

## 1. Trạng thái Dự án & Dashboard (Cập nhật 2026-07-21)
* **Thesis Visuals & Dual-Mode:** Đã hoàn thiện chế độ hiển thị kép (Color cho Dashboard, Đen Trắng / B&W cho in ấn báo cáo luận văn QĐ 1799).
* **High-DPI Export:** Tích hợp `kaleido` xuất ảnh PNG 300 DPI, bổ sung nền trắng che text annotation (`bgcolor="rgba(255,255,255,0.92)"`) chống đè số liệu.
* **Đánh số Biểu đồ Chuẩn:** Đã đánh số tự động toàn bộ biểu đồ thesis ("Hình 4.1" đến "Hình 4.5").
* **Bulk Export:** Đã thêm tính năng xuất ZIP toàn bộ 5 hình B&W luận văn 300 DPI tại trang Thesis Figures.
* **Citations & References:** Đã thêm trích dẫn IEEE (`cite()`) và danh mục tài liệu tham khảo (`render_references_section()`) vào trang Kết Luận và Explainability Hub.

---

## 2. Mục tiêu phiên làm việc tiếp theo
* [ ] Rà soát tổng thể lần cuối trước khi in báo cáo luận văn.
* [ ] Đảm bảo Graphify graph được re-build sau mỗi phiên code.

---

## 3. Lưu ý nóng (Active Gotchas)
* **Giới hạn RAM 512MB (Render):** OOM nếu chạy SHAP/Inference lớn. Render chỉ dùng cho demo web, không train lại.
* **B&W Plotly Export:** Luôn dùng `to_bw()` từ `src.viz.chart_factory` để đảm bảo hatch pattern, marker, và background box cho text annotations.
* **Kaleido PNG Rendering:** Cần `uv add kaleido` để `to_image()` hoạt động trơn tru.
* **Graphify Re-build:** Sau mỗi phiên sửa code `.py`, PHẢI chạy `make graphify` để cập nhật knowledge graph. Agent đọc `graphify-out/GRAPH_REPORT.md` TRƯỚC khi sửa code.
