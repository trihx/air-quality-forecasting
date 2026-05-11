# 📦 Hướng Dẫn Cài Đặt & Chạy Demo — PM2.5 Forecasting

> Dashboard dự báo nồng độ PM2.5 sử dụng Machine Learning & Deep Learning
> Đề án Thạc sĩ — Đại học Cần Thơ

---

## 🚀 Cách 1: Chạy nhanh nhất (Docker)

### Yêu cầu
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) đã cài đặt
- Tối thiểu **4GB RAM** khả dụng

### Bước 1: Pull & chạy image

```bash
docker run -d \
  --name pm25-demo \
  -p 8501:8501 \
  trihx/pm25-forecasting:latest
```

### Bước 2: Mở Dashboard

Mở trình duyệt → truy cập: **http://localhost:8501**

### Bước 3 (Tùy chọn): Cấu hình AI Assistant

Để sử dụng tính năng Trợ Lý AI, nhập API key vào sidebar:

```bash
# Cách 1: Inject qua biến môi trường
docker run -d \
  --name pm25-demo \
  -p 8501:8501 \
  -e GEMINI_API_KEY=your_api_key_here \
  trihx/pm25-forecasting:latest

# Cách 2: Dùng file .env
docker run -d \
  --name pm25-demo \
  -p 8501:8501 \
  --env-file .env \
  trihx/pm25-forecasting:latest
```

**Lấy API key miễn phí:**
| Provider | URL | Ghi chú |
|----------|-----|---------|
| **Google Gemini** ⭐ | https://aistudio.google.com/app/apikey | Miễn phí 15 RPM, không cần thẻ tín dụng |
| **Groq** | https://console.groq.com/keys | Miễn phí 30 RPM |
| **OpenAI** | https://platform.openai.com/api-keys | $5 credit khi đăng ký mới |

---

## 🖥️ Cách 2: Chạy kèm LM Studio (AI Offline)

Nếu muốn dùng AI offline (không cần internet), cài thêm LM Studio:

### Bước 1: Cài LM Studio
1. Download từ https://lmstudio.ai
2. Mở LM Studio → Search → Tải model **Qwen3-4B (Q4_K_M)** (~2.5GB)
3. Vào tab **Local Server** → Start Server (port **8888**)

### Bước 2: Chạy Docker với kết nối LM Studio

```bash
docker run -d \
  --name pm25-demo \
  -p 8501:8501 \
  --add-host=host.docker.internal:host-gateway \
  -e LM_STUDIO_URL=http://host.docker.internal:8888/v1 \
  trihx/pm25-forecasting:latest
```

### Bước 3: Mở Dashboard → Trợ Lý AI sẽ tự kết nối LM Studio

---

## 🐳 Cách 3: Docker Compose (đầy đủ)

```bash
# Clone project
git clone https://github.com/trihx/time-series-forecasting.git
cd time-series-forecasting

# Copy file cấu hình
cp .env.example .env
# Mở .env → điền API key (nếu muốn dùng AI)

# Chạy
docker compose up -d

# Mở: http://localhost:8501
```

**Nếu có NVIDIA GPU:**
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

---

## 📜 Hướng Dẫn Sử Dụng Dashboard

### Các trang chính

| # | Trang | Mô tả |
|---|-------|-------|
| 1 | 🏠 Tổng Quan | KPIs, overview dự án |
| 2 | 📜 Quy Trình Pipeline | **7 bước nghiên cứu** — từ thu thập đến kết luận |
| 3 | 📊 EDA | Phân tích khám phá dữ liệu (biểu đồ tương tác) |
| 4 | ⚙️ Hyperparameters | Cấu hình & tối ưu hóa mô hình |
| 5 | 🏋️ Huấn Luyện | Train LightGBM/GRU trực tiếp trên Dashboard |
| 6 | 📋 Lịch Sử Thí Nghiệm | So sánh các thí nghiệm đã chạy |
| 7 | 📈 Multi-Horizon | Kết quả 1h/6h/24h + so sánh 28 mô hình |
| 8 | 📉 Actual vs Predicted | Biểu đồ dự đoán vs thực tế |
| 9 | 🔍 SHAP | Giải thích mô hình — features nào quan trọng nhất |
| 10 | 📊 Khoảng Tin Cậy | Prediction Intervals (Conformal/Quantile) |
| 11 | 🔮 Dự Báo PM2.5 | Dự báo real-time với dữ liệu mới |
| 12 | 💬 Trợ Lý AI | Hỏi đáp về dự án, hỗ trợ phản biện |

### Gợi ý cho Hội đồng

1. **Bắt đầu** từ trang "📜 Quy Trình Pipeline" — xem toàn bộ workflow nghiên cứu
2. **Đánh giá** ở trang "📈 Multi-Horizon" — so sánh hiệu suất 28 mô hình
3. **Kiểm tra** ở trang "🏋️ Huấn Luyện" — thử train LightGBM trực tiếp (~15 giây)
4. **Hỏi đáp** ở trang "💬 Trợ Lý AI" — đặt câu hỏi phản biện

---

## ❓ Troubleshooting

| Vấn đề | Giải pháp |
|--------|----------|
| Port 8501 đã bị chiếm | Đổi port: `-p 8502:8501` → truy cập `:8502` |
| Docker pull chậm | Thử Mirror: Docker Desktop → Settings → Docker Engine → thêm registry-mirrors |
| AI không phản hồi | Kiểm tra API key đã nhập đúng, hoặc LM Studio đang chạy |
| Container bị tắt | `docker logs pm25-demo` để xem lỗi |
| Hết RAM | Tăng RAM Docker: Docker Desktop → Settings → Resources → Memory → 6GB |

### Lệnh Docker hữu ích

```bash
# Xem logs
docker logs -f pm25-demo

# Restart
docker restart pm25-demo

# Dừng & xóa
docker stop pm25-demo && docker rm pm25-demo

# Kiểm tra image size
docker images trihx/pm25-forecasting
```

---

## 📊 Thông tin Kỹ thuật

| Thành phần | Chi tiết |
|-----------|---------|
| Python | 3.11 |
| Framework | Streamlit 1.56+ |
| ML/DL | LightGBM 4.6, PyTorch 2.11 (CPU-only) |
| RAG | ChromaDB 1.5 + sentence-transformers |
| LLM | OpenAI-compatible SDK (Gemini/OpenAI/Groq/LM Studio) |
| Data | 209K records, PM2.5 + 4 biến khí tượng |
| Models | 28 models × 3 horizons (1h/6h/24h) |
| Tests | 167/167 passed ✅ |

---

*Được phát triển bởi Anh Trí (trihx) — Đề án Thạc sĩ 2026, Đại học Cần Thơ*
