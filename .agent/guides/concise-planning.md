# Nguyên tắc Lập kế hoạch Ngắn gọn (Concise Planning)

> **Nguồn**: Skill `concise-planning` — adapted cho dự án Time Series Forecasting PM2.5

---

## Mục đích

Biến mọi yêu cầu thành **một kế hoạch duy nhất**, ngắn gọn, có thể hành động ngay. Tránh lập kế hoạch dài dòng, mơ hồ, hoặc thiếu bước kiểm chứng.

---

## Quy trình

### Bước 1: Quét Context (Scan)

- Đọc `.agent/SKILL.md`, `.agent/memory/CONTEXT.md`, code liên quan
- Xác định constraints: thư viện, kiến trúc, tests hiện có

### Bước 2: Tương tác Tối thiểu

- Hỏi **tối đa 1-2 câu**, chỉ khi thực sự blocking
- Với unknowns không blocking → đưa ra giả định hợp lý

### Bước 3: Sinh Plan theo Template

---

## Plan Template (BẮT BUỘC)

```markdown
# Plan: [Tên công việc]

[1-3 câu mô tả high-level approach và lý do chọn approach này]

## Scope
- **In**: [Liệt kê cụ thể: files, modules, chức năng sẽ thay đổi]
- **Out**: [Những gì KHÔNG nằm trong scope lần này]

## Action Items (6-10 bước, verb-first)
- [ ] Bước 1: [Discovery — đọc file X, kiểm tra module Y]
- [ ] Bước 2: [Implementation — thêm/sửa code tại Z]
- [ ] Bước 3: [Implementation — ...]
- [ ] Bước 4: [Validation — chạy pytest, kiểm tra output]
- [ ] Bước 5: [Rollout — commit, cập nhật memory]

## Open Questions (tối đa 3)
- Câu hỏi 1? (chỉ hỏi khi thực sự blocking)
```

---

## Quy tắc Action Items

| Quy tắc | Mô tả | Ví dụ |
|---------|--------|-------|
| **Atomic** | Mỗi bước là 1 đơn vị công việc logic | ❌ "Implement data pipeline" → ✅ "Tạo `src/data/loader.py` với hàm `load_csv()`" |
| **Verb-first** | Bắt đầu bằng động từ | "Thêm...", "Refactor...", "Kiểm tra...", "Tạo..." |
| **Concrete** | Ghi rõ file/module/hàm cụ thể | ❌ "Update models" → ✅ "Thêm `XGBoostModel` vào `src/models/ml/xgboost.py`" |
| **Testable** | Ít nhất 1 bước validation | "Chạy `uv run pytest tests/unit/test_loader.py -v`" |

---

## Áp dụng trong Dự án

### Khi nào dùng Plan Template

- Trước khi implement bất kỳ feature mới
- Trước khi refactor module
- Trước khi chạy experiment mới (kết hợp với Experiment Design)

### Tích hợp với Gate-keeping (SKILL.md Section 1.1)

```
1. Đọc context (SKILL.md, memory/)
2. Sinh Plan theo template trên
3. Trình cho user duyệt
4. User approve → Implement
5. Cập nhật memory/ sau khi xong
```

---

## Anti-patterns

| Sai | Đúng |
|-----|------|
| Plan 3 trang A4 | Plan 1 trang, 6-10 action items |
| "Có thể sẽ cần refactor X" | "Refactor X" hoặc bỏ ra khỏi scope |
| 15 open questions | Tối đa 3 câu hỏi blocking |
| Action items mơ hồ | Mỗi item ghi rõ file, hàm, thay đổi gì |
