# Nguyên tắc Code Quality — Lint & Validate

> **Nguồn**: Skill `lint-and-validate` — adapted cho Python Time Series project
>
> **BẮT BUỘC**: Chạy lint/validate SAU MỖI thay đổi code. Code không qua lint = chưa hoàn thành.

---

## Quality Loop (Vòng lặp Chất lượng)

```
Viết/Sửa Code
    ↓
Ruff Check (lint)  ──→ Fix lỗi → chạy lại
    ↓ PASS
Ruff Format (format) ──→ Auto-fix
    ↓ PASS
Bandit (security)  ──→ Fix security issues
    ↓ PASS
MyPy (types)       ──→ Fix type errors
    ↓ PASS
Pytest (tests)     ──→ Fix failing tests
    ↓ ALL PASS
✅ Code sẵn sàng commit
```

---

## Công cụ Bắt buộc

| Công cụ | Mục đích | Lệnh | Khi nào chạy |
|---------|----------|-------|-------------|
| **Ruff** | Lint (thay flake8 + isort) | `uv run ruff check src/ --fix` | Sau mỗi thay đổi |
| **Ruff Format** | Auto-format (thay black) | `uv run ruff format src/` | Sau mỗi thay đổi |
| **Bandit** | Security scanning | `uv run bandit -r src/ -ll` | Trước commit |
| **MyPy** | Type checking | `uv run mypy src/` | Trước commit |
| **Pytest** | Unit tests + coverage | `uv run pytest tests/ -v --cov=src` | Trước commit |

---

## Cấu hình trong pyproject.toml

```toml
[tool.ruff]
target-version = "py311"
line-length = 120
src = ["src"]

[tool.ruff.lint]
select = [
    "E",    # pycodestyle errors
    "F",    # pyflakes
    "W",    # pycodestyle warnings
    "I",    # isort
    "N",    # pep8-naming
    "UP",   # pyupgrade
    "S",    # bandit (subset via ruff)
    "B",    # flake8-bugbear
    "A",    # flake8-builtins
    "C4",   # flake8-comprehensions
    "SIM",  # flake8-simplify
    "TCH",  # flake8-type-checking
]
ignore = ["S101"]  # Allow assert in tests

[tool.ruff.lint.per-file-ignores]
"tests/**" = ["S101", "S603"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
ignore_missing_imports = true

[tool.bandit]
exclude_dirs = ["tests", ".venv"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
```

---

## Quick Commands

```bash
# === Lint & Format ===
uv run ruff check src/ --fix          # Lint + auto-fix
uv run ruff format src/               # Format code
uv run ruff check src/ tests/ --fix   # Lint cả tests

# === Security ===
uv run bandit -r src/ -ll             # Security scan (low + medium)
uv run bandit -r src/ -ll -f json     # Output JSON cho CI

# === Type Checking ===
uv run mypy src/                      # Type check
uv run mypy src/ --html-report htmlcov/mypy  # HTML report

# === Testing ===
uv run pytest tests/ -v --cov=src     # Tests + coverage
uv run pytest tests/ -v --cov=src --cov-report=html  # HTML coverage

# === All-in-one ===
uv run ruff check src/ --fix && uv run ruff format src/ && uv run bandit -r src/ -ll && uv run mypy src/ && uv run pytest tests/ -v
```

---

## Error Handling

| Tình huống | Hành động |
|-----------|----------|
| Ruff check fail | Fix style/syntax issues ngay lập tức |
| Ruff format thay đổi files | Review changes, thường an toàn auto-accept |
| Bandit warning | Đánh giá risk, fix hoặc suppress cụ thể với `# nosec` + comment lý do |
| MyPy error | Fix type mismatches trước khi tiếp tục |
| Không có tool configured | Kiểm tra `pyproject.toml`, chạy `uv add --dev ruff bandit mypy` |

---

## Tích hợp với Workflow

### Khi Refactor (`/refactor`)

1. Chạy full lint TRƯỚC refactor → baseline
2. Refactor từng module
3. Chạy full lint SAU MỖI module → so sánh

### Khi Chạy Experiment (`/run-experiment`)

1. Lint check trước khi chạy → đảm bảo code sạch
2. Chạy experiment
3. Lint check code mới (nếu có)

### Khi Debug

1. Fix bug → chạy lint → đảm bảo fix không tạo lỗi mới

---

## Strict Rule

> **Không có code nào được commit hoặc báo cáo "done" mà không pass toàn bộ lint checks.**
