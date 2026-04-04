# Time Series Embedding & Similarity Search

> **Nguồn**: Skill `embedding-strategies` — adapted cho Time Series PM2.5
>
> **Trạng thái**: 📌 PLACEHOLDER — Triển khai SAU khi có baseline models hoạt động
>
> Xem `.agent/memory/TODO.md` — mục "Ưu tiên thấp"

---

## Khi nào Triển khai

**Prerequisites** (phải hoàn thành trước):

- [ ] Baseline models hoạt động (Naive, ARIMA, Random Forest)
- [ ] Evaluation pipeline chạy ổn định
- [ ] MASE < 1.0 cho ít nhất 1 model
- [ ] Feature engineering module hoàn thiện

**Use cases dự kiến**:

1. **Similar Day Finding**: Tìm ngày có PM2.5 pattern giống → dự báo tham khảo
2. **Anomaly Detection**: Episodes lệch xa cluster → cảnh báo
3. **Regime Classification**: "bình thường" vs "ô nhiễm cao" vs "spike"
4. **Transfer Learning**: Pre-trained embeddings cho các trạm quan trắc khác

---

## Planned Approaches

| Approach | Phù hợp khi | Thư viện | Complexity |
|---------|------------|---------|-----------|
| **Statistical features** | Nhanh, interpret được, baseline | `tsfresh`, custom | Low |
| **Autoencoder** | Dimensionality reduction | PyTorch | Medium |
| **ts2vec** | Learned universal representations | `ts2vec` | Medium |
| **Shapelet-based** | Pattern matching cụ thể | `tslearn` | Medium |

**Đề xuất**: Bắt đầu với Statistical features (low effort, good enough cho most use cases), scale lên nếu cần.

---

## Planned Implementation

```python
# Sẽ implement tại: src/features/embeddings.py (sau khi có baseline)

def embed_timeseries_window(
    window: np.ndarray,
    method: str = "statistical"
) -> np.ndarray:
    """Embed 1 cửa sổ thời gian thành feature vector."""
    if method == "statistical":
        return np.array([
            window.mean(),
            window.std(),
            window.min(),
            window.max(),
            np.percentile(window, 25),
            np.percentile(window, 75),
            # Trend coefficient
            np.polyfit(range(len(window)), window, 1)[0],
            # Autocorrelation lag-1
            np.corrcoef(window[:-1], window[1:])[0, 1] if len(window) > 1 else 0,
        ])
    # Thêm methods khác sau...
```

---

## References

- Embedding Strategies skill: evaluation metrics (NDCG, MRR, Precision@K)
- `tsfresh`: Automatic time series feature extraction
- `ts2vec`: Universal representation learning for time series
- `tslearn`: Machine learning for time series data
