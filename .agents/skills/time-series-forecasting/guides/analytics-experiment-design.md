# Analytics & Experiment Design cho ML

> **Nguồn**: Skill `analytics-tracking` + `ab-test-setup` — adapted cho ML model comparison trong dự án PM2.5

---

## 1. Measurement Readiness Index (Trước khi Tin Kết quả)

> Trước khi rút kết luận từ bất kỳ experiment nào, **PHẢI** đánh giá Signal Quality.

### Scoring

| Tiêu chí | Weight | Câu hỏi kiểm tra |
|----------|--------|-------------------|
| **Decision Alignment** | 25% | Kết quả này sẽ ảnh hưởng quyết định gì? Model nào deploy? |
| **Metric Clarity** | 20% | Metrics có đúng ý nghĩa? MAE vs RMSE vs MASE phù hợp cho use case? |
| **Data Accuracy** | 20% | Dữ liệu sạch? Không data leakage? Preprocessing consistent? |
| **Reproducibility** | 15% | Chạy lại có ra kết quả giống? Seed fixed? Config logged? |
| **Baseline Comparison** | 10% | Đã so với naive baseline? MASE < 1.0? |
| **Documentation** | 10% | Đã ghi đầy đủ vào RUNS_LOG.md? Config saved? |

### Verdict

| Score | Verdict | Hành động |
|-------|---------|----------|
| **85-100** | ✅ Measurement-Ready | An toàn để quyết định và optimize |
| **70-84** | ⚠️ Usable with Gaps | Fix gaps trước khi ra quyết định lớn |
| **55-69** | ❌ Unreliable | Dữ liệu không đáng tin — fix pipeline trước |
| **< 55** | 🚫 Broken | KHÔNG hành động dựa trên dữ liệu này |

**Nếu verdict là Broken → DỪNG, fix data/pipeline trước khi chạy thêm experiments.**

---

## 2. Model Comparison Experiment Design

### 2.1 Hypothesis Lock (BẮT BUỘC trước khi chạy)

Trước khi so sánh models, **PHẢI** viết giả thuyết:

```
Giả thuyết: "[Model A] sẽ có [primary metric] tốt hơn [Model B] 
             ít nhất [X]% trên test set [mô tả],
             vì [lý do dựa trên domain knowledge/evidence]"
             
Ví dụ: "XGBoost sẽ có MAE thấp hơn Random Forest ít nhất 5%
        trên test set 2024-Q4, vì XGBoost xử lý interactions
        giữa nhiệt độ và độ ẩm tốt hơn qua gradient boosting."
```

> **"Đây có phải giả thuyết cuối cùng không?"** → Phải confirm trước khi tiếp.

### 2.2 Experiment Checklist

Chỉ được chạy experiment khi **TẤT CẢ** đều TRUE:

- [ ] **Hypothesis locked**: Giả thuyết đã viết và confirm
- [ ] **Primary metric frozen**: Chọn **1 metric chính** (MAE) — KHÔNG đổi sau khi chạy
- [ ] **Secondary metrics defined**: RMSE, MAPE, R² (context, không override primary)
- [ ] **Guardrail metrics set**: Metrics KHÔNG được tệ hơn
  - Inference time < [X] ms
  - Memory usage < [X] MB
  - Training time < [X] phút
- [ ] **Same data split**: Cả 2+ models dùng **CÙNG** train/val/test split
- [ ] **Same preprocessing**: Cùng features, cùng scaling, cùng pipeline
- [ ] **Seed fixed**: `RANDOM_STATE = 42` consistent
- [ ] **Baseline included**: Naive model luôn có mặt để benchmark
- [ ] **Tracking verified**: RUNS_LOG.md template sẵn sàng

### 2.3 Trong khi Chạy Experiment

**ĐƯỢC:**
- Monitor technical health (memory, GPU, disk)
- Log mọi anomaly xảy ra
- Document external factors (power outage, data refresh)

**KHÔNG ĐƯỢC:**
- ❌ Dừng sớm vì "kết quả trông tốt rồi"
- ❌ Thay đổi features/model giữa chừng
- ❌ Thay đổi data split
- ❌ Đổi success criteria

### 2.4 Kết quả Diễn giải

| Kết quả | Hành động |
|---------|----------|
| Model A tốt hơn có ý nghĩa (> MDE) | Xem xét adopt, ghi `DECISIONS.md` |
| Khác biệt không đáng kể (< MDE) | **Chọn model đơn giản hơn** |
| Model A tệ hơn guardrail | **KHÔNG ship**, dù primary metric "tốt" |
| Inconclusive | Cần thêm data hoặc thay đổi bolder |

---

## 3. Anti-patterns Experiment

| Anti-pattern | Vấn đề | Đúng cách |
|-------------|--------|----------|
| ❌ "Chạy 10 models, pick best MAE" | p-hacking, random luck | Lock hypothesis trước |
| ❌ "Model A tốt hơn 0.01%" | Noise, không ý nghĩa | Định nghĩa MDE trước |
| ❌ "Thay metric vì RMSE trông tốt hơn" | HARBing | Freeze metric trước khi chạy |
| ❌ "Model phức tạp nên tốt hơn" | Assumption bias | Luôn so với baseline đơn giản |
| ❌ "Thêm features cho đến khi tốt" | Overfit, leakage risk | Feature selection có hệ thống |

---

## 4. Experiment Record (BẮT BUỘC sau mỗi experiment)

```markdown
## Experiment: [Tên]
- **Date**: YYYY-MM-DD
- **Hypothesis**: [Copy từ hypothesis lock]
- **Models Compared**: [Model A vs Model B vs Baseline]
- **Data Split**: [Train: N rows, Val: M rows, Test: K rows]
- **Features**: [Số features, feature set name]

### Results

| Model | MAE ↓ | RMSE ↓ | MASE ↓ | R² ↑ | Train Time | Inference Time |
|-------|-------|--------|--------|------|------------|----------------|
| Naive Baseline | - | - | 1.0 | - | - | - |
| Model A | - | - | - | - | - | - |
| Model B | - | - | - | - | - | - |

### Guardrail Check
- [ ] Inference time < [X] ms ✅/❌
- [ ] Memory < [X] MB ✅/❌

### Decision
- **Result**: [Positive / Negative / Inconclusive]
- **Action**: [Adopt Model A / Keep current / Need more data]
- **Reason**: [Giải thích]

### Learnings
- [Điều gì đã học được]
- [Follow-up ideas]

### Run Directory
- `research/runs/YYYYMMDD_HHMMSS/`
```

---

## 5. Tracking cho Predictions (Khi Deploy)

### Metrics Monitoring sau Triển khai

| Metric | Mô tả | Alert Threshold |
|--------|--------|-----------------|
| **Prediction Drift** | Distribution shift so với training | KS-test p < 0.05 |
| **Feature Drift** | Input features thay đổi phân phối | PSI > 0.2 |
| **Performance Degradation** | MAE tăng so với baseline | > 20% increase |
| **Data Freshness** | Dữ liệu mới nhất | > 2 giờ stale |

> **Nguyên tắc**: Track for decisions, not curiosity. Mỗi metric phải map đến 1 quyết định cụ thể.

---

## 6. Core Principles (Non-Negotiable)

1. **Track for Decisions**: Nếu không có quyết định depend on metric → đừng track
2. **Questions First**: Định nghĩa câu hỏi → mới thiết kế experiment
3. **Quality > Volume**: 1 experiment chặt chẽ > 10 experiments cẩu thả
4. **Learning > Winning**: Mục đích là tìm ra sự thật, không phải chứng minh mình đúng
