# 🌫️ PM2.5 Air Quality Time Series Forecasting System

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Tests](https://img.shields.io/badge/tests-192%20passed-brightgreen.svg)](tests/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.56-FF4B4B.svg)](https://streamlit.io)
[![Database](https://img.shields.io/badge/PostgreSQL-Supabase-3ECF8E.svg)](https://supabase.com)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Hệ thống Dự báo Chuỗi thời gian Nồng độ Bụi mịn PM2.5 Đa Bước từ Dữ liệu Cảm biến IoT.**  
> Nghiên cứu kết hợp kỹ nghệ đặc trưng chống rò rỉ dữ liệu (*Anti-Leakage Discipline*), chiến lược nội suy phân tầng (*Tiered Imputation*), học sâu (*Deep Learning*), giải thích mô hình (*XAI SHAP*) và triển khai ứng dụng thực tế (*FastAPI + Streamlit + Supabase + Render.com*).

---

## 📌 Điểm Nổi Bật của Đề Tài

- **Kỷ luật Chống Rò rỉ Dữ liệu (Anti-Leakage Audit)**: Toàn bộ 119 đặc trưng trễ (lag), trung bình trượt (rolling) và vi phân (diff) áp dụng nghiêm ngặt phép trễ `shift(1)`, triệt tiêu hoàn toàn hiện tượng $R^2 \approx 1,0$ ảo.
- **Chiến lược Nội suy Phân tầng (Tiered Imputation)**: Kết hợp *Cubic Spline* (khoảng trống $\le 6\text{h}$), *KNN Imputer* đa biến vi khí tượng (khoảng trống $6\text{h} - 24\text{h}$) và cắt bỏ khoảng trống dài ($>24\text{h}$) kèm định danh phân đoạn `segment_id`.
- **Xác lập Điểm Ngọt Độ Phân Giải (30-minute Sweet Spot)**: Chứng minh bằng thực nghiệm tần suất 30 phút đạt tỷ lệ tín hiệu/nhiễu tối ưu, chiếm 80% vị trí dẫn đầu trong Top 5 mô hình tại các mốc dự báo 6h và 24h.
- **Mô hình Ensemble Tối Ưu**: Mô hình kết hợp trọng số (*Weighted Ensemble*) đạt $\text{MASE} = 0,382$ tại mốc 6h (giảm 49,6% MAE so với Baseline Persistence), có ý nghĩa thống kê theo kiểm định *Diebold-Mariano* ($p < 0,001$).
- **Độ Tin Cậy & Liêm Chính Khoa Học**: Toàn bộ hệ thống được bảo đảm bằng **192 bài kiểm thử tự động (100% Pass)**.

---

## 🏗️ Kiến Trúc Dự Án (Project Structure)

```
time-series-forecasting/
├── configs/                   # ⚙️ Cấu hình siêu tham số mô hình (YAML)
├── dataset/                   # 📊 Dữ liệu quan trắc IoT (raw & processed)
├── models/                    # 💾 Trọng số mô hình đã huấn luyện (pre-trained artifacts)
├── research/                  # 🔬 Kết quả thí nghiệm, ma trận đối soát và đồ thị
├── src/                       # 🔧 Mã nguồn Clean Architecture
│   ├── api/                   #    Backend RESTful API (FastAPI, SQLAlchemy, Supabase)
│   ├── dashboard/             #    Giao diện người dùng đa chức năng (Streamlit)
│   ├── data_quality/          #    Module tiền xử lý, S-ESD, STL và nội suy phân tầng
│   ├── features/              #    Kỹ nghệ 119 đặc trưng chuỗi thời gian chống rò rỉ
│   ├── models/                #    5 họ mô hình: Baseline, Statistical, ML, Deep Learning, Ensemble
│   ├── evaluation/            #    Hệ thống độ đo (MAE, RMSE, MASE, DA, Diebold-Mariano, CQR)
│   └── pipelines/             #    Pipeline huấn luyện và suy luận tự động
├── tests/                     # 🧪 192 bài Unit Tests & Anti-Leakage Audit
├── Dockerfile                 # 🐳 Cấu hình đóng gói Container Production
├── render.yaml                # ☁️ Cấu hình triển khai tự động trên Render.com
├── Makefile                   # 🛠️ Tự động hóa cài đặt, kiểm thử và chạy ứng dụng
└── pyproject.toml             # 📦 Quản lý gói phụ thuộc (uv package manager)
```

---

## ⚡ Hướng Dẫn Cài Đặt & Chạy Nhanh (Quick Start)

### 1. Yêu cầu Môi trường
- **Python**: $\ge 3.11$
- **uv**: Trình quản trị gói tốc độ cao của Python ([Cài đặt uv](https://github.com/astral-sh/uv))

### 2. Cài đặt Phụ thuộc
```bash
# Clone kho lưu trữ
git clone https://github.com/trihx/air-quality-forecasting.git
cd air-quality-forecasting

# Cài đặt toàn bộ môi trường và dependencies
uv sync
```

### 3. Chạy Kiểm thử Tự động (192 Tests)
```bash
uv run pytest tests/ -v
```

### 4. Khởi chạy Ứng dụng Local (Dashboard & API)
```bash
# Chạy đồng thời FastAPI (port 8000) và Streamlit Dashboard (port 7860)
make dev
```
Truy cập giao diện tại: `http://localhost:7860`  
Truy cập tài liệu API Swagger tại: `http://localhost:8000/docs`

---

## 📊 Tập Dữ Liệu Thực Nghiệm (Dataset Overview)

| Thông số | Giá trị | Ghi chú |
|---|:---:|---|
| **Vị trí Quan trắc** | Sa Đéc, Đồng Tháp | Trạm cảm biến IoT ngoài trời |
| **Tổng số Bản ghi** | 209.594 dòng | Dữ liệu gốc tần suất 2 phút |
| **Khoảng Thời gian** | 25/03/2022 $\rightarrow$ 11/05/2025 | Hơn 3 năm quan trắc liên tục |
| **Các Biến Khí tượng** | 4 biến | Nhiệt độ (°C), Độ ẩm (%), Điểm sương (°C), $\text{CO}_2$ (ppm) |
| **Biến Mục tiêu** | $\text{PM}_{2.5}$ ($\mu\text{g/m}^3$) | Nồng độ bụi mịn khí quyển |

---

## 🧪 Kết Quả Thực Nghiệm Tổng Hợp

Kết quả đánh giá trên tập kiểm thử mỏ neo (*Anchor Test Set* - 10% dữ liệu cuối):

| Chân trời (Horizon) | Tần suất | Mô hình Tối ưu | MAE ($\mu\text{g/m}^3$) | RMSE ($\mu\text{g/m}^3$) | MASE | R² | Directional Accuracy |
|---|:---:|---|:---:|:---:|:---:|:---:|:---:|
| **1 giờ** | 15m | GRU (15m) | **2,944** | 4,690 | **0,667** | 0,267 | 49,3% |
| **6 giờ** | 30m | **Ensemble Weighted (30m)** | **3,493** | **5,079** | **0,382** | -0,044 | **56,7%** |
| **24 giờ** | 30m | **Ensemble Weighted (30m)** | **3,417** | **4,872** | **0,469** | 0,070 | **54,8%** |

---

## ☁️ Triển Khai Thực Tế (Deployment)

Hệ thống được thiết kế theo kiến trúc Microservices đám mây:
- **Cloud Database**: **Supabase (PostgreSQL 17)** với Transaction & Session Connection Pooling.
- **Web App Hosting**: **Render.com** (hoặc HuggingFace Spaces) tự động build từ `Dockerfile` và `render.yaml`.
- **Inference Latency**: Thời gian phản hồi API trung bình $< 50\text{ ms}$.

---

## 📜 Giấy Phép (License)
Dự án được phân phối dưới giấy phép **MIT License**. Mọi đóng góp học thuật và mã nguồn đều tuân thủ các quy định về liêm chính học thuật.

