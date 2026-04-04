---
description: Quy trình refactor code - đảm bảo không phá vỡ chức năng và tuân thủ code standards
---

# Workflow: Refactor Code

Quy trình chuẩn khi refactor code, đảm bảo an toàn và truy vết được.

## Pre-flight Checklist

1. Đọc `.agent/SKILL.md` — đặc biệt sections: Architecture, Security, Unit Testing
2. Đọc `.agent/memory/CONTEXT.md` để nắm trạng thái hiện tại
3. Đọc `.agent/memory/LESSONS_LEARNED.md` để tránh lỗi đã biết
4. Xác định phạm vi refactor: files nào bị ảnh hưởng?

## Execution Steps

// turbo
5. Chạy full test suite TRƯỚC refactor để có baseline: `uv run pytest tests/ -v`

6. Ghi nhận test results trước refactor (số tests pass/fail)

7. Thực hiện refactor theo từng module nhỏ (KHÔNG refactor toàn bộ cùng lúc)

// turbo
8. Chạy full test suite SAU MỖI module refactor: `uv run pytest tests/ -v`

9. Kiểm tra security checklist:
   - Không có bare `except:` → dùng `except SpecificException`
   - Không có `joblib.load()` không kiểm tra nguồn
   - Không có `warnings.filterwarnings('ignore')` global
   - Input paths được validate
   - Không hard-code secrets/credentials

// turbo
10. Chạy full quality check:
```bash
uv run ruff check src/ --fix && uv run ruff format src/ && uv run bandit -r src/ -ll && uv run mypy src/
```

> **Tham khảo**: `.agent/guides/lint-and-validate.md` cho chi tiết cấu hình
> **Tham khảo**: `.agent/guides/kaizen.md` cho nguyên tắc refactor (3 vòng, Rule of Three)

## Post-flight Checklist

11. So sánh test results trước/sau — PHẢI giống nhau hoặc tốt hơn
12. Cập nhật `.agent/memory/CONTEXT.md` với thay đổi
13. Ghi quyết định refactor vào `.agent/memory/DECISIONS.md`
14. Nếu phát hiện lỗi → ghi vào `.agent/memory/LESSONS_LEARNED.md`
15. Cập nhật `.agent/memory/TODO.md`
