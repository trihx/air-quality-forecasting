# Visualization & Data Storytelling Guide — PM2.5

> Áp dụng data-storytelling framework: mỗi chart phải trả lời **MỘT câu hỏi cụ thể**.
> Three Pillars: **Data** (evidence) + **Narrative** (meaning) + **Visuals** (clarity).

---

## Chart Selection Matrix

| Câu hỏi | Chart Type | Library | Phase |
|----------|-----------|---------|-------|
| PM2.5 phân phối như thế nào? | Violin + Strip | seaborn | A: Understanding |
| PM2.5 thay đổi qua 3 năm? | Multi-panel Line + Annotations | matplotlib | A |
| Giờ/tháng nào PM2.5 cao nhất? | Hour×Month Heatmap | seaborn | A |
| Sensor có downtime không? | Gap Analysis Heatmap | matplotlib | A |
| Weekday vs Weekend patterns? | Faceted Boxplot Grid | seaborn | A |
| Features nào tương quan PM2.5? | Clustered Correlation Heatmap | seaborn | B: Patterns |
| PM2.5 autocorrelation? | ACF/PACF Dual | statsmodels | B |
| Trend/Seasonal components? | STL 4-panel | statsmodels | B |
| Quan hệ features ổn định? | Rolling Cross-Correlation | matplotlib | B |
| Model dự đoán chính xác? | Actual vs Predicted + CI | matplotlib | C: Model |
| Residuals có bias? | 4-in-1 Diagnostic | matplotlib | C |
| Model sai ở giờ nào? | Polar Error-by-Hour | matplotlib | C |
| Model nào win ở horizon nào? | Bump Chart | matplotlib | C |

---

## Project Visualization Rules

| Rule | Chi tiết |
|------|---------|
| **1 chart = 1 question** | Title = câu hỏi đang trả lời. Không vẽ vì "đẹp" |
| **WHO Reference** | Mọi PM2.5 chart PHẢI có `axhline(y=15)` (WHO guideline 2021) |
| **Color System** | `YlOrRd` = PM2.5 severity, `RdBu_r` = correlation, `viridis` = counts |
| **Labels** | Title tiếng Việt, axis labels tiếng Anh (convention khoa học) |
| **Annotation** | Charts phải tự giải thích — thêm text cho key insights |
| **Export** | `.png` (300 dpi) + `.svg` (vector) |
| **Font** | Minimum font size 12, consistent rcParams |

---

## Style Configuration

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Project-wide style
PROJECT_STYLE = {
    "figure.figsize": (12, 7),
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.constrained_layout.use": True,
}
plt.rcParams.update(PROJECT_STYLE)
sns.set_palette("deep")

# WHO PM2.5 guideline
WHO_PM25_ANNUAL = 5     # µg/m³ (annual mean)
WHO_PM25_24H = 15       # µg/m³ (24-hour mean)

# Project color palette
COLORS = {
    "pm25_low": "#2ecc71",       # Good (≤WHO)
    "pm25_moderate": "#f39c12",  # Moderate
    "pm25_high": "#e74c3c",      # Unhealthy
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "accent": "#F18F01",
    "neutral": "#C73E1D",
}
```

---

## Phase A: Data Understanding Charts

### A1. PM2.5 Distribution — Violin + Strip

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Violin plot (linear scale)
sns.violinplot(y=df["pm25"], ax=axes[0], color=COLORS["primary"], inner="quartile")
axes[0].axhline(y=WHO_PM25_24H, color="red", linestyle="--", label=f"WHO 24h ({WHO_PM25_24H} µg/m³)")
axes[0].set_title("Phân phối PM2.5 — Linear Scale")
axes[0].legend()

# Log-scale histogram (vì skewed)
axes[1].hist(df["pm25"].clip(lower=0.1), bins=100, log=True, color=COLORS["primary"], edgecolor="black", alpha=0.7)
axes[1].axvline(x=WHO_PM25_24H, color="red", linestyle="--", label=f"WHO 24h")
axes[1].set_xlabel("PM2.5 (µg/m³)")
axes[1].set_ylabel("Count (log scale)")
axes[1].set_title("Phân phối PM2.5 — Log Scale")
axes[1].legend()

plt.savefig("research/eda/plots/a1_pm25_distribution.png", dpi=300, bbox_inches="tight")
plt.savefig("research/eda/plots/a1_pm25_distribution.svg", bbox_inches="tight")
```

### A2. Time Series Overview — Multi-panel

```python
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

# Panel 1: PM2.5
ax1.plot(df["ngay_tao"], df["pm25"], alpha=0.3, linewidth=0.5, color="gray", label="Raw")
ax1.plot(df["ngay_tao"], df["pm25"].rolling(window=168*30).mean(), linewidth=2, color=COLORS["primary"], label="7-day Rolling Mean")
ax1.axhline(y=WHO_PM25_24H, color="red", linestyle="--", alpha=0.7, label="WHO 24h Guideline")
ax1.set_ylabel("PM2.5 (µg/m³)")
ax1.set_title("PM2.5 theo thời gian — 3.1 năm dữ liệu")
ax1.legend()

# Panel 2: CO2
ax2.plot(df["ngay_tao"], df["co2"], alpha=0.3, linewidth=0.5, color="gray", label="Raw")
ax2.plot(df["ngay_tao"], df["co2"].rolling(window=168*30).mean(), linewidth=2, color=COLORS["secondary"], label="7-day Rolling Mean")
ax2.set_xlabel("Thời gian")
ax2.set_ylabel("CO2 (ppm)")
ax2.set_title("CO2 theo thời gian")
ax2.legend()

# Annotate spike events (PM2.5 > 100)
spikes = df[df["pm25"] > 100]
for _, row in spikes.head(3).iterrows():
    ax1.annotate(f"Spike: {row['pm25']:.0f}", xy=(row["ngay_tao"], row["pm25"]),
                 fontsize=8, color="red", arrowprops=dict(arrowstyle="->", color="red"))
```

### A3. Hour×Month Heatmap

```python
pivot = df.groupby([df["ngay_tao"].dt.month, df["ngay_tao"].dt.hour])["pm25"].mean().unstack()
pivot.index.name = "Tháng"
pivot.columns.name = "Giờ"

fig, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(pivot, cmap="YlOrRd", annot=True, fmt=".1f", linewidths=0.5, ax=ax,
            cbar_kws={"label": "PM2.5 trung bình (µg/m³)"})
ax.set_title("PM2.5 trung bình theo Giờ và Tháng")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Month")
```

---

## Phase B: Pattern Discovery Charts

### B1. Clustered Correlation Heatmap

```python
corr = df[["nhiet_do", "do_am", "diem_suong", "co2", "pm25"]].corr()

g = sns.clustermap(corr, cmap="RdBu_r", vmin=-1, vmax=1, annot=True, fmt=".2f",
                   linewidths=1, figsize=(8, 8),
                   cbar_kws={"label": "Pearson Correlation"})
g.fig.suptitle("Correlation Heatmap (Clustered)", y=1.02)
```

### B2. ACF/PACF Dual

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

hourly = df.set_index("ngay_tao")["pm25"].resample("1h").mean().dropna()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
plot_acf(hourly, lags=168, ax=ax1, alpha=0.05)
ax1.set_title("Autocorrelation Function (ACF) — PM2.5 Hourly")
ax1.axvline(x=24, color="red", linestyle=":", alpha=0.5, label="Lag 24 (daily)")
ax1.axvline(x=168, color="blue", linestyle=":", alpha=0.5, label="Lag 168 (weekly)")
ax1.legend()

plot_pacf(hourly, lags=168, ax=ax2, alpha=0.05, method="ywm")
ax2.set_title("Partial Autocorrelation Function (PACF)")
```

### B3. STL Decomposition

```python
from statsmodels.tsa.seasonal import STL

hourly = df.set_index("ngay_tao")["pm25"].resample("1h").mean().interpolate()
stl = STL(hourly, period=24, robust=True)
result = stl.fit()

fig = result.plot()
fig.set_size_inches(14, 10)
fig.suptitle("STL Decomposition — PM2.5 (period=24h)", fontsize=14, fontweight="bold")
```

---

## Phase C: Model Storytelling Charts

### C1. Actual vs Predicted + Confidence Bands

```python
fig, ax = plt.subplots(figsize=(14, 6))

# Show 2-week window for readability
window = slice("2025-04-01", "2025-04-14")
ax.plot(y_true[window], linewidth=2, color=COLORS["primary"], label="Actual")
ax.plot(y_pred[window], linewidth=2, color=COLORS["accent"], linestyle="--", label="Predicted")
ax.fill_between(y_pred[window].index,
                y_pred[window] - 1.96*std_pred, y_pred[window] + 1.96*std_pred,
                alpha=0.2, color=COLORS["accent"], label="95% CI")
ax.axhline(y=WHO_PM25_24H, color="red", linestyle="--", alpha=0.5, label="WHO Guideline")
ax.set_title("Actual vs Predicted PM2.5 — 2-Week Window")
ax.legend()

# Annotate largest error
max_error_idx = (y_true[window] - y_pred[window]).abs().idxmax()
ax.annotate(f"Max Error: {abs(y_true[max_error_idx]-y_pred[max_error_idx]):.1f}",
            xy=(max_error_idx, y_true[max_error_idx]), fontsize=9, color="red",
            arrowprops=dict(arrowstyle="->", color="red"))
```

### C2. Residual Analysis 4-in-1

```python
residuals = y_true - y_pred

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (1) Residuals vs Time
axes[0,0].scatter(range(len(residuals)), residuals, alpha=0.3, s=5, color=COLORS["primary"])
axes[0,0].axhline(y=0, color="red", linestyle="--")
axes[0,0].set_title("Residuals vs Time")

# (2) Residuals Histogram
axes[0,1].hist(residuals, bins=50, edgecolor="black", alpha=0.7, color=COLORS["primary"])
axes[0,1].set_title("Residuals Distribution")

# (3) Q-Q Plot
from scipy import stats
stats.probplot(residuals, plot=axes[1,0])
axes[1,0].set_title("Q-Q Plot")

# (4) Residuals vs Predicted
axes[1,1].scatter(y_pred, residuals, alpha=0.3, s=5, color=COLORS["primary"])
axes[1,1].axhline(y=0, color="red", linestyle="--")
axes[1,1].set_title("Residuals vs Predicted")

fig.suptitle("Residual Diagnostic — 4 Views", fontsize=14, fontweight="bold")
```

### C3. Error by Hour — Polar Chart

```python
import numpy as np

mae_by_hour = residuals.abs().groupby(residuals.index.hour).mean()
theta = np.linspace(0, 2*np.pi, 24, endpoint=False)

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))
bars = ax.bar(theta, mae_by_hour.values, width=2*np.pi/24, alpha=0.7, color=COLORS["primary"])

# Highlight high-error hours
for bar, mae in zip(bars, mae_by_hour.values):
    if mae > mae_by_hour.mean() * 1.5:
        bar.set_color(COLORS["pm25_high"])

ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_xticks(theta)
ax.set_xticklabels([f"{h}h" for h in range(24)])
ax.set_title("MAE theo giờ trong ngày (Polar)", pad=20, fontweight="bold")
```

---

## Data Storytelling Checklist

Trước khi present bất kỳ chart nào, kiểm tra:

- [ ] **Hook**: Title có gây chú ý? ("PM2.5 spike 40x" vs "PM2.5 Distribution")
- [ ] **Context**: Có baseline/reference line? (WHO guideline)
- [ ] **Annotation**: Key insights có text trên chart?
- [ ] **1 Question**: Chart chỉ trả lời 1 câu hỏi?
- [ ] **"So What"**: Kết luận rút ra là gì? Ghi dưới chart
- [ ] **Color**: Đúng palette? Colorblind-friendly?
- [ ] **Export**: Lưu cả .png (300dpi) + .svg?

---

## EDA Report Template

```markdown
# EDA Report — YYYYMMDD_HHMMSS

## 1. Data Overview
- Rows: {n_rows:,}
- Date range: {start} → {end}
- Missing: {missing_pct:.1f}%

## 2. Univariate Analysis
- PM2.5: median={median}, mean={mean}, std={std}
- Skewness: {skewness:.2f} (> 2 = heavily skewed)
![Distribution](plots/a1_pm25_distribution.png)

## 3. Temporal Patterns
![Time Series](plots/a2_time_series_overview.png)
![Heatmap](plots/a3_hourly_monthly_heatmap.png)

## 4. Stationarity
- ADF test: p={adf_p:.4f} (< 0.05 = stationary ✅)
- KPSS test: p={kpss_p:.4f} (> 0.05 = stationary ✅)

## 5. Autocorrelation
![ACF/PACF](plots/b2_acf_pacf.png)
- Significant lags: {significant_lags}

## 6. Correlations
![Heatmap](plots/b1_correlation_heatmap.png)
- Strongest correlation with PM2.5: {strongest_corr}

## 7. Decomposition
![STL](plots/b3_stl_decomposition.png)

## 8. Key Findings
1. ...
2. ...
3. ...

## 9. Recommendations for Feature Engineering
1. ...
2. ...
```

---

## Anti-patterns

| ❌ Don't | ✅ Do |
|----------|-------|
| `plt.show()` không `savefig()` trước | `plt.savefig(...)` rồi `plt.show()` |
| Dùng `jet` colormap | Dùng `viridis` hoặc `YlOrRd` |
| Default font size (8) | `rcParams` set ≥ 12 |
| Title: "Chart 1" | Title: "PM2.5 tăng 3x vào tháng 1-3" |
| Forget WHO reference line | Luôn thêm `axhline(y=15)` |
| Dùng `plt.plot()` interface | Dùng `fig, ax = plt.subplots()` OO |
