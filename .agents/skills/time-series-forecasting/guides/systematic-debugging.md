# Phương pháp Debug Có Hệ thống (Systematic Debugging)

> **Nguồn**: Skill `systematic-debugging` — adapted cho dự án Time Series Forecasting PM2.5
>
> **LUẬT SẮT**: KHÔNG sửa code khi chưa tìm ra root cause. Sửa triệu chứng = thất bại.

---

## Khi nào Dùng

Dùng cho **MỌI** vấn đề kỹ thuật:

- Test failures
- Bug trong pipeline
- Unexpected behavior (metrics bất thường, predictions sai)
- Performance problems (training chậm, memory leak)
- Build/import failures

**ĐẶC BIỆT** khi:

- Đang gấp (gấp + đoán = thrashing)
- "Chỉ cần fix nhanh cái này" (dấu hiệu đoán mò)
- Đã thử fix 2+ lần mà chưa xong
- Không hiểu rõ vấn đề

---

## 4 Pha Debug

### Phase 1: Tìm Nguyên nhân Gốc rễ (Root Cause)

> **TRƯỚC KHI thử BẤT KỲ fix nào:**

1. **Đọc KỸ Error Message**
   - Đọc TOÀN BỘ stack trace, không skip
   - Ghi chú: file, line number, error type
   - Error messages thường chứa chính xác giải pháp

2. **Reproduce Ổn định**
   - Có thể trigger lỗi lại được không?
   - Ghi chính xác các bước reproduce
   - Nếu không reproduce → thu thập thêm data, KHÔNG đoán

3. **Kiểm tra Thay đổi Gần đây**
   - `git diff` — có gì thay đổi?
   - Dependencies mới? Config thay đổi?
   - Environment khác? (Python version, package version)

4. **Thu thập Bằng chứng (Multi-component)**
   ```python
   # Khi lỗi nằm ở pipeline nhiều bước:
   # Data Loading → Cleaning → Feature Eng → Training → Prediction
   
   # Kiểm tra TẠI MỖI boundary:
   print(f"=== After Loading ===")
   print(f"Shape: {df.shape}, NaN: {df.isna().sum().sum()}")
   
   print(f"=== After Cleaning ===")
   print(f"Shape: {df_clean.shape}, NaN: {df_clean.isna().sum().sum()}")
   
   print(f"=== After Features ===")
   print(f"Shape: {features.shape}, Inf: {np.isinf(features).sum().sum()}")
   
   # → Tìm ra CHÍNH XÁC bước nào dữ liệu bắt đầu sai
   ```

5. **Trace Data Flow**
   - Giá trị sai xuất phát từ đâu?
   - Hàm nào gọi hàm nào với giá trị sai?
   - Truy ngược lên cho đến nguồn gốc

---

### Phase 2: Phân tích Pattern

1. **Tìm Code Tương tự Đang Chạy Đúng**
   - Trong cùng project có code similar working không?
   - Ví dụ: `loader.py` load file khác có hoạt động không?

2. **So sánh Khác biệt**
   - Liệt kê MỌI khác biệt giữa working và broken
   - Đừng giả định "cái này chắc không quan trọng"

3. **Kiểm tra Dependencies**
   - Component này cần gì để hoạt động?
   - Config, environment, data format có đúng không?

---

### Phase 3: Giả thuyết & Kiểm chứng

1. **Phát biểu Giả thuyết Rõ ràng**
   ```
   "Tôi nghĩ [X] là root cause vì [Y]"
   ```
   - Viết ra, cụ thể, không mơ hồ

2. **Test Tối thiểu**
   - Thay đổi NHỎ NHẤT có thể để test giả thuyết
   - MỘT biến tại một thời điểm
   - KHÔNG fix nhiều thứ cùng lúc

3. **Verify**
   - Đúng → Phase 4
   - Sai → Giả thuyết MỚI (không thêm fix chồng lên)

---

### Phase 4: Implement Fix

1. **Viết Failing Test Case TRƯỚC**
   ```python
   def test_regression_issue_xyz():
       """Regression test cho issue: [mô tả ngắn]"""
       # Setup: reproduce lỗi
       data = create_problem_data()
       
       # Act
       result = function_under_test(data)
       
       # Assert: condition mà trước đây FAIL
       assert result is not None
       assert not np.isnan(result).any()
   ```

2. **Implement MỘT Fix**
   - Sửa root cause đã xác định
   - MỘT thay đổi tại một thời điểm
   - KHÔNG "trong khi đang sửa, refactor luôn"

3. **Verify Fix**
   - Test mới pass?
   - Tests cũ vẫn pass?
   - Issue thực sự resolved?

4. **Nếu Fix Không Work → Đếm**
   - < 3 fix attempts: Quay lại Phase 1, phân tích lại
   - **≥ 3 fix thất bại: DỪNG LẠI**

5. **≥ 3 Fix Thất Bại → Xem xét Architecture**
   
   Dấu hiệu vấn đề kiến trúc:
   - Mỗi fix lộ ra vấn đề mới ở chỗ khác
   - Fix yêu cầu "massive refactoring"
   - Mỗi fix tạo side effect mới
   
   → **Thảo luận với user trước khi thử fix tiếp**

---

## Red Flags — DỪNG LẠI ngay khi nghĩ:

| Suy nghĩ | Vấn đề | Hành động |
|----------|--------|----------|
| "Thử sửa X xem sao" | Chưa có root cause | Quay Phase 1 |
| "Fix nhanh rồi tìm sau" | Ngược quy trình | Quay Phase 1 |
| "Sửa nhiều chỗ cho nhanh" | Không isolate được | Quay Phase 3 |
| "Liệt kê 5 fix luôn" | Chưa investigate | Quay Phase 1 |
| "Thêm 1 fix nữa" (đã 2+) | Sắp thrashing | Xem xét architecture |

---

## Lỗi Thường Gặp trong Time Series (Quick Reference)

| Lỗi | Triệu chứng | Debug approach |
|-----|-------------|---------------|
| **Data Leakage** | Test accuracy cao bất thường (R² > 0.99) | Kiểm tra feature tạo từ future data |
| **Look-ahead Bias** | Tốt trên test, xấu trên live | Kiểm tra scaler fit trên toàn bộ data |
| **NaN Propagation** | Metrics = NaN | Trace NaN từ features → xem rolling/lag |
| **Shape Mismatch** | ValueError khi predict | In shape tại mỗi stage |
| **Temporal Mismatch** | Predictions lệch | Unit test cho lag features |
| **Overfitting** | Train loss ↓, val loss ↑ | Kiểm tra early stopping config |

---

## Template Ghi vào LESSONS_LEARNED.md

```markdown
## [YYYY-MM-DD] [Tên lỗi ngắn gọn]
- **Triệu chứng**: [Mô tả chính xác error/behavior]
- **Root Cause (Phase 1)**: [Phân tích nguyên nhân gốc rễ]
- **Working Example (Phase 2)**: [Code/module tương tự đang chạy đúng]
- **Hypothesis (Phase 3)**: "[X] là root cause vì [Y]"
- **Fix Applied (Phase 4)**: [Thay đổi cụ thể, file nào, dòng nào]
- **Regression Test**: [Link đến test file]
- **Thời gian debug**: [N phút — systematic vs sẽ mất bao lâu nếu đoán]
- **Files liên quan**: [Danh sách]
```

---

## Thống kê Thực tế

| Approach | Thời gian fix | Tỷ lệ fix đúng lần đầu | Bug mới tạo ra |
|----------|-------------|----------------------|----------------|
| **Systematic** | 15-30 phút | ~95% | Gần 0 |
| **Đoán mò** | 2-3 giờ thrashing | ~40% | Thường có |
