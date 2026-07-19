# 🧠 L0: Time Series Forecasting Project Index (AGENTS.md)

> **Tech Stack:** Python (FastAPI, Streamlit, PostgreSQL, MongoDB, PyTorch, LightGBM, uv)
> **Quy tắc đọc:** Đọc file này cùng với `docs/MEMORY_HOT.md` ở LÚC BẮT ĐẦU mỗi phiên. Giữ dưới 100 dòng.

---

## 1. 🔒 Quy Tắc Giao Tiếp (Global Rules Kế Thừa)
* **Xưng hô:** Gọi là anh Trí (biệt danh: `trihx`), xưng em. Trả lời ngắn gọn, trực diện, không vòng vo.
* **Tính phản biện:** Phản biện có chiều sâu, phân tích kỹ thuật và thực tế để đưa ra giải pháp tốt nhất.
* **Vai trò:** Lập trình viên VNPT (wifi, fiber, camera IP 24/7) kiêm trợ lý Data Scientist & AI Developer cho anh Trí.

---

## 2. ⚡ Tối Ưu Hóa Token (Lazy-Loading Guides)
Để tránh quá tải context và lãng phí Input Tokens, Agent **CẤM** tự động nạp các tài liệu hướng dẫn chi tiết. Chỉ dùng tool `view_file` để đọc (Lazy Load) khi xử lý tác vụ tương ứng:

* 📅 Lập kế hoạch: [concise-planning.md](file:///Users/trihx/Desktop/time-series-forecasting/.agents/skills/time-series-forecasting/guides/concise-planning.md)
* 📊 EDA & Phân tích: [visualization-storytelling.md](file:///Users/trihx/Desktop/time-series-forecasting/.agents/skills/time-series-forecasting/guides/visualization-storytelling.md)
* 🧪 Thí nghiệm & Baseline: [analytics-experiment-design.md](file:///Users/trihx/Desktop/time-series-forecasting/.agents/skills/time-series-forecasting/guides/analytics-experiment-design.md)
* 🔧 Feature Engineering: [data-engineering.md](file:///Users/trihx/Desktop/time-series-forecasting/.agents/skills/time-series-forecasting/guides/data-engineering.md)
* 🤖 Huấn luyện Mô hình: [model-training.md](file:///Users/trihx/Desktop/time-series-forecasting/.agents/skills/time-series-forecasting/guides/model-training.md)
* 📈 Đánh giá Metrics: [evaluation-metrics.md](file:///Users/trihx/Desktop/time-series-forecasting/.agents/skills/time-series-forecasting/guides/evaluation-metrics.md)

---

## 📂 3. Hệ Thống Phân Tầng Bộ Nhớ (Tiered Memory System)
* **L0 (Index):** File này (`.agents/AGENTS.md`) — Tech stack, rules giao tiếp, tối ưu token.
* **L1 (Hot Memory):** [docs/MEMORY_HOT.md](file:///Users/trihx/Desktop/time-series-forecasting/docs/MEMORY_HOT.md) — Tiến độ, Gotchas nóng (OOM 512MB, Supabase Auto-pause).
* **L2 (Lessons):** [docs/LESSONS_LEARNED.md](file:///Users/trihx/Desktop/time-series-forecasting/docs/LESSONS_LEARNED.md) — Tổng hợp bug, bài học ML (Anti-leakage, Imputation).
* **L3 (Logs):** [docs/DECISIONS_LOG.md](file:///Users/trihx/Desktop/time-series-forecasting/docs/DECISIONS_LOG.md) — Nhật ký các quyết định kiến trúc dự án từ v1 đến nay.

---

## 🛠️ 4. Phím Tắt Lệnh Nhanh (Makefile Automation)
* Cài đặt môi trường: `make install`
* Chạy song song Backend + Frontend: `make dev`
* Chạy Lint + Format + Types + Tests: `make check`
* Chạy riêng Unit Tests: `make test`
* Tối ưu hóa & dọn dẹp bộ nhớ (L1->L3): `make update-memory`
* Dọn dẹp cache hệ thống: `make clean`
