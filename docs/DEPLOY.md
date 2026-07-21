# 🚀 Hướng dẫn Deploy — PM2.5 Forecasting Dashboard

> **Mục tiêu:** Deploy dự án lên cloud miễn phí để demo live trước hội đồng bảo vệ luận văn.
> **Stack:** HuggingFace Spaces (App) + Supabase (PostgreSQL Database)

---

## 1. Tạo PostgreSQL Database trên Supabase (Miễn phí)

### Bước 1: Tạo tài khoản & Project
1. Truy cập [supabase.com](https://supabase.com) → **Start your project** (đăng nhập bằng GitHub).
2. Click **New Project** → Đặt tên `pm25-forecasting` → Chọn region **Southeast Asia (Singapore)**.
3. Đặt **Database Password** → Lưu lại password này.

### Bước 2: Lấy Connection String
1. Vào **Project Settings** → **Database** → Tab **Connection string** → Chọn **URI**.
2. Copy chuỗi kết nối có dạng:
   ```
   postgresql://postgres.[project-ref]:[password]@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres

   postgresql://postgres.frmumzvjrmurybkpmwls:[YOUR-PASSWORD]@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres
   ```
3. **Lưu ý:** Chọn mode **Transaction** (port 6543) thay vì Session (port 5432) để tối ưu cho serverless.

### Bước 3: Seed dữ liệu ban đầu
Database sẽ tự động tạo tables khi API khởi động (nhờ SQLAlchemy `create_all`). Dữ liệu info_cards cũng được auto-seed.

---

## 2. Deploy App lên Render (Miễn phí, hỗ trợ Docker)

Render.com cho phép deploy ứng dụng Docker hoàn toàn miễn phí (Free Tier: 512 MB RAM). Chúng ta sẽ chạy song song cả Streamlit và FastAPI trong một container duy nhất để tiết kiệm tài nguyên.

### Bước 1: Chuẩn bị Dockerfile
Hệ thống đã cấu hình sẵn file [Dockerfile.hf](file:///Users/trihx/Desktop/time-series-forecasting/Dockerfile.hf) chạy song song cả 2 dịch vụ qua `supervisord`. 
Để Render nhận diện, anh chạy lệnh sau tại thư mục root dự án để copy đè lên file Dockerfile mặc định:
```bash
cp Dockerfile.hf Dockerfile
```

### Bước 2: Tạo Web Service trên Render
1. Truy cập [render.com](https://render.com) và đăng nhập bằng tài khoản GitHub.
2. Click **New** → Chọn **Web Service**.
3. Kết nối tài khoản GitHub và chọn repository `time-series-forecasting` của anh.
4. Cấu hình dịch vụ:
   - **Name:** `pm25-forecasting`
   - **Region:** Chọn **Singapore** để kết nối tới Supabase nhanh nhất.
   - **Branch:** `main`
   - **Runtime:** Chọn **Docker**
   - **Instance Type:** Chọn **Free** (0$ - 512 MB RAM)

### Bước 3: Cấu hình biến môi trường
Click vào nút **Advanced** → **Add Environment Variable** để thêm các biến sau:

| Key | Value | Bắt buộc |
|---|---|---|
| `DATABASE_URL` | Connection string từ Supabase (Bước 1.2) | ✅ |
| `OPENAI_API_KEY` | API key OpenAI (nếu dùng RAG chatbot) | ❌ |
| `GEMINI_API_KEY` | API key Google Gemini (nếu dùng RAG chatbot) | ❌ |

### Bước 4: Deploy & Domain Live
1. Click **Deploy Web Service** (hoặc push code lên nhánh master/main trên GitHub, Render sẽ tự động trigger build mới).
2. Render sẽ tự động build Docker image và deploy. Quá trình này mất khoảng 5-10 phút.
3. Domain Live chính thức của dự án: **`https://time-series-forecasting-c8gz.onrender.com/`**

---

## 3. Keep-Alive (Chống ngủ đông cho Render)

Mặc định, các app chạy trên gói Free của Render sẽ tự động ngủ đông (sleep) sau 15 phút không có traffic. Khi có người truy cập lại, app sẽ mất khoảng 1 phút để khởi động lại (cold start).

Để app luôn sẵn sàng phục vụ demo trước hội đồng bảo vệ, dự án đã tích hợp sẵn GitHub Actions workflow để tự động ping app mỗi 6 tiếng.

### Kích hoạt Keep-Alive:
1. Push code lên GitHub repo (bao gồm file `.github/workflows/keep-alive.yml`).
2. Vào GitHub repo của anh → **Settings** → **Secrets and variables** → **Actions** → Thêm secret mới:
   - **Name:** `RENDER_URL` (hoặc `HF_SPACE_URL`)
   - **Value:** `https://time-series-forecasting-c8gz.onrender.com`
3. GitHub Actions sẽ tự động ping app mỗi 6 tiếng để giữ cho Render không bị ngủ đông.

---

## 4. Kiến trúc Deploy

```
       https://time-series-forecasting-c8gz.onrender.com
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│               Render Web Service (Docker)           │
│  ┌──────────────┐    ┌──────────────────────┐       │
│  │  Streamlit    │    │  FastAPI Backend     │       │
│  │  (port 7860)  │◄──►│  (port 8000)         │       │
│  └──────────────┘    └──────────┬───────────┘       │
│       supervisord               │                   │
└─────────────────────────────────┼───────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │   Supabase PostgreSQL    │
                    │   (Singapore region)    │
                    └─────────────────────────┘
```

---

## Lưu ý quan trọng

> [!WARNING]
> - **RAM 512MB:** Gói Free của Render chỉ cung cấp 512 MB RAM. Để tránh lỗi OOM (Out Of Memory) làm crash container, em đã tối ưu cấu hình supervisord chạy FastAPI với 1 worker duy nhất. Tránh bấm chạy lại các pipeline huấn luyện mô hình nặng trực tiếp trên Render.
> - **Cold Start:** Nếu app của anh bị ngủ đông (ví dụ khi GitHub Actions bị lỗi hoặc chưa chạy), lần đầu truy cập sẽ mất khoảng 30-50 giây để app hoạt động lại.
> - **Database:** Supabase free tier sẽ tạm dừng (pause) project sau 1 tuần không hoạt động. Nếu app báo lỗi kết nối database, anh chỉ cần vào Supabase dashboard nhấn **Restore** là được.


