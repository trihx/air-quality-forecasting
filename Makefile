# ═══════════════════════════════════════════
# PM2.5 Forecasting — Project Automation Makefile
# ═══════════════════════════════════════════

.PHONY: help install dev test check seed update-memory graphify clean

help:
	@echo "📌 Các lệnh tự động hóa dự án PM2.5 Forecasting:"
	@echo "  make install         - Cài đặt môi trường & dependencies bằng uv"
	@echo "  make dev             - Chạy đồng thời Backend (FastAPI) và Frontend (Streamlit)"
	@echo "  make test            - Chạy toàn bộ Unit và Integration tests"
	@echo "  make check           - Kiểm tra chất lượng code (Ruff Lint, Format, Mypy, Tests)"
	@echo "  make seed            - Nạp dữ liệu mẫu (Seed Data) lên PostgreSQL Database"
	@echo "  make update-memory   - Tự động tối ưu hóa và dọn dẹp bộ nhớ dự án (L1 -> L3)"
	@echo "  make graphify        - Cập nhật Knowledge Graph (Graphify) từ codebase"
	@echo "  make clean           - Dọn dẹp các file cache tạm (__pycache__, pytest, ruff...)"

install:
	@echo "🚀 Đang cài đặt dependencies..."
	uv sync

dev:
	@echo "🚀 Đang dọn dẹp các cổng kết nối cũ..."
	@lsof -ti :8501 | xargs kill -9 2>/dev/null || true
	@lsof -ti :8000 | xargs kill -9 2>/dev/null || true
	@sleep 1
	@echo "🌐 Khởi chạy ứng dụng..."
	@echo "FastAPI Backend: http://localhost:8000"
	@echo "Streamlit Frontend: http://localhost:8501"
	@uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --log-level info & \
	 uv run streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true & \
	 wait

test:
	@echo "🧪 Đang chạy Unit Tests..."
	uv run pytest tests/ -v

check:
	@echo "🔍 1. Chạy Ruff Format..."
	uv run ruff format src/
	@echo "🔍 2. Chạy Ruff Lint..."
	uv run ruff check src/ --fix
	@echo "🔍 3. Chạy Mypy Type Check..."
	uv run mypy src/
	@echo "🧪 4. Chạy Pytest..."
	uv run pytest tests/

seed:
	@echo "🌱 Đang nạp dữ liệu mẫu lên PostgreSQL..."
	uv run python scripts/seed_info_cards.py

update-memory:
	@uv run python scripts/utilities/update_memory.py

graphify:
	@echo "🔗 Đang cập nhật Knowledge Graph (Graphify)..."
	uvx --from graphifyy graphify extract .
	@echo "✅ Graphify graph đã được cập nhật tại graphify-out/"

clean:
	@echo "🧹 Đang dọn dẹp các file cache..."
	find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -r {} + 2>/dev/null || true
	@echo "✨ Đã dọn dẹp xong!"

