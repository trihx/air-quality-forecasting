# ═══════════════════════════════════════════
# PM2.5 Forecasting — Project Automation Makefile
# ═══════════════════════════════════════════

.PHONY: help install dev run-api run-ui lint test check verify guard security pre-commit pre-commit-install seed update-memory map-skills graphify sync-figures clean

help:
	@echo "📌 Các lệnh tự động hóa dự án PM2.5 Forecasting:"
	@echo "  make install            - Cài đặt môi trường & dependencies bằng uv"
	@echo "  make dev                - Chạy đồng thời Backend (FastAPI) và Frontend (Streamlit)"
	@echo "  make run-api            - Chạy độc lập Backend FastAPI (port 8000)"
	@echo "  make run-ui             - Chạy độc lập Frontend Streamlit (port 8501)"
	@echo "  make lint               - Kiểm tra linter và type check (Ruff, Mypy)"
	@echo "  make test               - Chạy toàn bộ Unit và Integration tests"
	@echo "  make check              - Kiểm tra toàn diện chất lượng code (Format, Lint, Mypy, Bandit, Tests)"
	@echo "  make security           - Quét an toàn bảo mật (Bandit SAST, Pip-Audit)"
	@echo "  make pre-commit         - Chạy toàn bộ Pre-commit hooks"
	@echo "  make pre-commit-install - Cài đặt Pre-commit git hook vào .git/hooks"
	@echo "  make seed               - Nạp dữ liệu mẫu (Seed Data) lên PostgreSQL Database"
	@echo "  make sync-figures       - Đồng bộ 38 hình ảnh chuẩn thức giữa docs/pics và research/figures/thesis"
	@echo "  make update-memory      - Tự động tối ưu hóa và dọn dẹp bộ nhớ dự án (L1 -> L3)"
	@echo "  make map-skills         - Cập nhật linking & mapping skills, bundles từ global vào dự án"
	@echo "  make graphify           - Cập nhật Knowledge Graph (Graphify) từ codebase"
	@echo "  make clean              - Dọn dẹp các file cache tạm (__pycache__, pytest, ruff...)"

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
	@OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --log-level info & \
	 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 uv run streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true & \
	 wait

run-api:
	@echo "🚀 Khởi chạy FastAPI Backend (http://localhost:8000)..."
	@OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --log-level info

run-ui:
	@echo "🌐 Khởi chạy Streamlit Frontend (http://localhost:8501)..."
	@OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 uv run streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true

lint:
	@echo "🔍 1. Chạy Ruff Lint (check)..."
	uv run ruff check src/ app.py pages.py tests/
	@echo "🔍 2. Chạy Ruff Format (check)..."
	uv run ruff format --check src/ app.py pages.py tests/
	@echo "🔍 3. Chạy Mypy Type Check..."
	uv run mypy src/ app.py pages.py

test:
	@echo "🧪 Đang chạy Unit Tests..."
	uv run pytest tests/ -v

check:
	@echo "🔍 1. Chạy Ruff Format..."
	uv run ruff format src/ app.py pages.py tests/
	@echo "🔍 2. Chạy Ruff Lint..."
	uv run ruff check src/ app.py pages.py tests/ --fix
	@echo "🔍 3. Chạy Mypy Type Check..."
	uv run mypy src/ app.py pages.py
	@echo "🛡️ 4. Chạy Bandit SAST Scan..."
	uv run bandit -c pyproject.toml -r src/ app.py pages.py
	@echo "🧪 5. Chạy Pytest..."
	uv run pytest tests/

security:
	@echo "🛡️ 1. Chạy Bandit SAST Scan..."
	uv run bandit -c pyproject.toml -r src/ app.py pages.py
	@echo "🛡️ 2. Chạy Pip-Audit kiểm tra phụ thuộc..."
	uv run pip-audit --skip-editable || true

pre-commit:
	@echo "🪝 Chạy Pre-commit hooks trên toàn bộ files..."
	uv run pre-commit run --all-files

pre-commit-install:
	@echo "🪝 Cài đặt Pre-commit git hook..."
	uv run pre-commit install

verify: check

guard: check

seed:
	@echo "🌱 Đang nạp dữ liệu mẫu lên PostgreSQL..."
	uv run python scripts/seed_info_cards.py

update-memory:
	@uv run python scripts/utilities/update_memory.py

graphify:
	@echo "🔗 Đang cập nhật Knowledge Graph (Graphify)..."
	uvx --from graphifyy graphify extract . --code-only
	uvx --from graphifyy graphify cluster-only .

sync-figures:
	@echo "📸 Đang đồng bộ và kiểm định 38 hình ảnh luận văn..."
	@uv run python scripts/sync_thesis_figures.py

clean:
	@echo "🧹 Đang dọn dẹp các file cache..."
	find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -r {} + 2>/dev/null || true
	@echo "✨ Đã dọn dẹp xong!"


# AI DevKit & Knowledge Graph Targets (Playbook v4)
agent-console: ## Mở live console giám sát AI agents
	@npx ai-devkit@latest agent console

memory-sync: ## Đồng bộ Markdown SoT vào local SQLite cache
	@python3 ~/.gemini/scripts/utilities/project_workflow.py sync-memory .

map-skills: ## Cập nhật linking & mapping skills, bundles từ global vào dự án
	@python3 ~/.gemini/scripts/utilities/project_workflow.py map .
