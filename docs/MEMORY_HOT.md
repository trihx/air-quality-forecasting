# 🔥 L1: Project Hot Memory (MEMORY_HOT.md)

> **Trạng thái hiện tại & Mục tiêu nóng của dự án (Cập nhật: 2026-07-19)**
> **Quy tắc:** Đọc mỗi khi bắt đầu phiên làm việc mới. Giữ dưới 80 dòng.

---

## 1. Trạng thái Deploy hiện tại
* **Database:** Supabase PostgreSQL (AWS Singapore region) hoạt động tốt. Dữ liệu `info_cards` tự động seed thành công.
* **App Web:** Deploy Docker (Streamlit + FastAPI qua supervisord) thành công lên **Render.com** (Free Tier).
* **Keep-Alive:** Kích hoạt GitHub Actions ping tự động app Render mỗi 6 giờ chống ngủ đông (sleep).
* **Domain:** Sử dụng domain mặc định của Render.

---

## 2. Mục tiêu tiếp theo (Next Targets)
* [ ] Kiểm tra tính ổn định của app trên Render sau 1 tuần chạy thực tế.
* [ ] Viết tài liệu tổng kết và chuẩn bị slide demo live trước hội đồng bảo vệ luận văn.
* [ ] Đóng gói và hướng dẫn các thành viên khác cách dựng local nhanh bằng `uv sync`.

---

## 3. Lưu ý nóng (Active Gotchas)
* **Giới hạn RAM 512MB:** Render Free Tier rất dễ bị OOM nếu chạy inference đồng thời hoặc chạy SHAP computation nặng. Không chạy huấn luyện (training) trực tiếp trên Render.
* **Supabase Auto-pause:** Project Supabase sẽ bị tạm dừng sau 1 tuần không hoạt động. Nếu app lỗi DB, truy cập Supabase click **Restore**.
* **Cold start:** Lần đầu truy cập sau khi app ngủ đông (nếu Actions ping bị lỗi) sẽ mất ~40s để khởi động lại.
