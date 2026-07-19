# 📝 L2: Lessons Learned (LESSONS_LEARNED.md)

> **Tổng hợp các bài học kinh nghiệm và lỗi cần tránh (Cập nhật: 2026-07-19)**
> **Quy tắc:** Đọc trước khi viết code mới. Giữ dưới 100 dòng.

---

## 1. Nhật ký Bài học (Lessons Log)

| Ngày | Danh mục | Lỗi & Cách Khắc phục | Tài liệu / Source |
|---|---|---|---|
| 2026-07-14 | `DEPLOY` | HF Spaces thu phí Docker → Chuyển sang Render.com Free Tier | [DEPLOY.md](file:///Users/trihx/Desktop/time-series-forecasting/docs/DEPLOY.md) |
| 2026-07-14 | `DEPLOY` | Render Free sleep sau 15p → Ping mỗi 6h qua GitHub Actions | [.github/workflows/keep-alive.yml](file:///Users/trihx/Desktop/time-series-forecasting/.github/workflows/keep-alive.yml) |
| 2026-07-14 | `DEPLOY` | Lỗi SSH Public Key khi push → Đổi remote sang HTTPS + dùng PAT | [SYNC_GUIDE.md](file:///Users/trihx/Desktop/time-series-forecasting/docs/SYNC_GUIDE.md) |
| 2026-07-14 | `CONFIG` | PyTorch đa nền tảng bị lỗi build → Dùng `marker` trong `uv` sources | [pyproject.toml](file:///Users/trihx/Desktop/time-series-forecasting/pyproject.toml) |
| 2026-07-14 | `ML` | Data leakage qua tỷ lệ thay đổi → Sử dụng `.shift(1)` trước khi tính | Playbook |
| 2026-07-14 | `ML` | statsmodels mất index freq → Dùng `.values` (numpy) khi fit và forecast | Playbook |

---

## 2. Quy tắc cốt lõi về chất lượng dữ liệu ML (IoT PM2.5)

* **Anti-Leakage:** KHÔNG BAO GIỜ dùng target tại thời điểm `t` trong features. Mọi phép tính sai phân, tỷ lệ thay đổi đều phải qua `shift(1)`. R² > 0.99 = lỗi rò rỉ dữ liệu.
* **Tiered Imputation:** Missing data trong IoT: Spline (gap ngắn ≤6h), KNN (gap trung bình 6-24h), Drop (gap dài >24h). Không dùng univariate interpolation cho gap >6h.
* **Test-on-Real-Only:** Dữ liệu đã điền khuyết (imputed) chỉ dùng để train. Tập Test/Eval bắt buộc dùng dữ liệu thực tế (`is_imputed == 0`).
* **Autocorrelation trap:** Ở short horizon (1h), baseline Persistence cực mạnh do tự tương quan cao. Mục tiêu ML là tập trung thắng ở các horizon xa hơn (6h, 24h).
