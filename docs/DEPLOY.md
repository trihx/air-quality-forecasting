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

## 3. Keep-Alive (Chống ngủ đông cho Render & Supabase)

### A. Cơ chế ngủ đông của từng nền tảng:
1. **Render.com Web Service (Free Tier):**
   - **Quy tắc:** Tự động ngủ đông (Spin down / Sleep) sau **15 phút liên tục** không có HTTP Request.
   - **Hậu quả:** Người truy cập sau đó chịu Cold Start mất khoảng 30–50 giây.
   - **Giải pháp:** GitHub Actions ping tự động **12 phút / 1 lần** (`*/12 * * * *`).

2. **Supabase PostgreSQL (Free Tier):**
   - **Quy tắc:** Tự động tạm dừng (Auto-Pause) dự án sau **7 ngày liên tục** không có lượt truy vấn SQL / HTTP API tác động vào Database.
   - **Hậu quả:** Database ngắt kết nối, app báo lỗi HTTP 500 DB connection failure.
   - **Giải pháp:** Mỗi lần GitHub Actions ping app Render, Streamlit / FastAPI sẽ chạy query kết nối DB (FastAPI có route `/health` chứa query `SELECT 1` hoặc Streamlit load info_cards), giúp Supabase ghi nhận activity liên tục và **không bao giờ bị Auto-Pause**.

---

### B. Kích hoạt Keep-Alive Đa tầng (Defense-in-Depth):

#### 1. GitHub Actions (Tự động 12 phút/lần - Đã tích hợp):
1. Push file `.github/workflows/keep-alive.yml` lên GitHub repo.
2. (Tùy chọn) Vào GitHub repo → **Settings** → **Secrets and variables** → **Actions** → Thêm secret:
   - **Name:** `RENDER_URL`
   - **Value:** `https://time-series-forecasting-c8gz.onrender.com` (hoặc custom domain `https://pm25.hoangxuantri.id.vn`)
3. GitHub Actions sẽ tự động ping mỗi 12 phút để giữ cho cả Render và Supabase luôn thức.

#### 2. UptimeRobot / Cron-Job.org (Khuyên dùng kết hợp 🌟):
Để chống trường hợp GitHub Runner bị xếp hàng (queue delay), anh nên thiết lập thêm 1 Monitor miễn phí 100%:
1. Đăng ký tài khoản tại [UptimeRobot.com](https://uptimerobot.com) hoặc [Cron-Job.org](https://cron-job.org).
2. Tạo HTTP Monitor:
   - **URL:** `https://time-series-forecasting-c8gz.onrender.com`
   - **Monitoring Interval:** `5 minutes` (hoặc 10 minutes)
3. Hệ thống này sẽ ping 24/7 hoàn toàn tự động và gửi thông báo cho anh nếu app gặp rủi ro gián đoạn.

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

## 5. Cấu hình Custom Domain (Miễn phí)

Render hỗ trợ gắn Tên miền riêng (Custom Domain) hoàn toàn miễn phí và tự động cấp chứng chỉ bảo mật **SSL/TLS (HTTPS)** qua Let's Encrypt.

### Đề xuất lựa chọn Tên miền:
- **Tên miền phụ (Subdomain - Khuyên dùng ⭐):** `pm25.hoangxuantri.id.vn` hoặc `air.hoangxuantri.id.vn`
- **Tên miền gốc (Root Domain):** `hoangxuantri.id.vn`

### Bước 1: Thêm Custom Domain trên Render Dashboard
1. Truy cập [render.com](https://render.com) → Chọn Web Service `pm25-forecasting`.
2. Vào mục **Settings** → Cuộn xuống phần **Custom Domains**.
3. Click **Add Custom Domain** → Nhập tên miền của anh (ví dụ: `pm25.hoangxuantri.id.vn`).
4. Click **Save**. Render sẽ hiển thị thông số CNAME cần trỏ.

### Bước 2: Cấu hình DNS tại Nhà cung cấp Tên miền (iNET / PA VietNam / Cloudflare...)
Đăng nhập vào trang quản lý DNS tên miền `hoangxuantri.id.vn` và thêm bản ghi sau:

#### Trường hợp 1: Dùng Subdomain (Khuyên dùng - ví dụ `pm25.hoangxuantri.id.vn`)
| Loại (Type) | Tên bản ghi (Host/Name) | Giá trị (Value/Target) | TTL |
|---|---|---|---|
| **CNAME** | `pm25` | `time-series-forecasting-c8gz.onrender.com` | Auto / 300 |

#### Trường hợp 2: Dùng Tên miền chính gốc (`hoangxuantri.id.vn`)
| Loại (Type) | Tên bản ghi (Host/Name) | Giá trị (Value/Target) | TTL |
|---|---|---|---|
| **A** | `@` | `216.24.57.1` | Auto / 300 |
| **CNAME** | `www` | `time-series-forecasting-c8gz.onrender.com` | Auto / 300 |

### Bước 3: Xác minh & Nhận SSL HTTPS
- Sau khi lưu bản ghi DNS, mất từ 2–15 phút để DNS lan tỏa (Propagation).
- Render sẽ tự động xác minh DNS và cấp chứng chỉ HTTPS miễn phí trong vòng 5–10 phút.
- Trang web live chính thức sẽ đổi thành: **`https://pm25.hoangxuantri.id.vn`** (hoặc `https://hoangxuantri.id.vn`).

---

## Lưu ý quan trọng

> [!WARNING]
> - **RAM 512MB:** Gói Free của Render chỉ cung cấp 512 MB RAM. Để tránh lỗi OOM (Out Of Memory) làm crash container, em đã tối ưu cấu hình supervisord chạy FastAPI với 1 worker duy nhất. Tránh bấm chạy lại các pipeline huấn luyện mô hình nặng trực tiếp trên Render.
> - **Cold Start:** Nếu app của anh bị ngủ đông (ví dụ khi GitHub Actions bị lỗi hoặc chưa chạy), lần đầu truy cập sẽ mất khoảng 30-50 giây để app hoạt động lại.
> - **Database:** Supabase free tier sẽ tạm dừng (pause) project sau 1 tuần không hoạt động. Nếu app báo lỗi kết nối database, anh chỉ cần vào Supabase dashboard nhấn **Restore** là được.


