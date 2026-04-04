---
description: Kiểm tra chất lượng code - lint, format, security scan, type check
---

# Workflow: Lint & Validate

Quy trình chuẩn kiểm tra chất lượng code trước khi commit hoặc khi cần audit.

## Pre-flight

1. Đảm bảo dev dependencies đã cài: `uv sync`

## Quality Loop

// turbo
2. Lint check + auto-fix: `uv run ruff check src/ --fix`

// turbo
3. Format code: `uv run ruff format src/`

// turbo
4. Security scan: `uv run bandit -r src/ -ll`

// turbo
5. Type check: `uv run mypy src/`

// turbo
6. Chạy tests: `uv run pytest tests/ -v --cov=src`

## Nếu có lỗi

7. Fix tất cả lỗi lint/security/type
8. Chạy lại từ bước 2 cho đến khi TẤT CẢ pass
9. Chỉ khi tất cả pass → code sẵn sàng commit

## Post-flight

10. Nếu fix lỗi đáng chú ý → ghi vào `.agent/memory/LESSONS_LEARNED.md`

## Tham khảo
- Chi tiết cấu hình: `.agent/guides/lint-and-validate.md`
