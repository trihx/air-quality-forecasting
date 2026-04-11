"""
Khó khăn do Đỉnh cực đoan (Erratic Spikes)
Mục tiêu: Cho thấy sự phân bố đuôi dài (Fat-Tailed) của PM2.5 và
cách các đỉnh ô nhiễm xuất hiện bất ngờ phá hủy dự báo của Machine Learning.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.dates import DateFormatter

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "dataset" / "interim" / "cleaned_hourly.csv"
OUTPUT_DIR = PROJECT_ROOT / "research" / "eda" / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Styling
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("talk")
PRIMARY_COLOR = "#007AFF"  # Blue
ERROR_COLOR = "#FF3B30"  # Red
WARNING_COLOR = "#FFCC00"  # Yellow
SAFE_COLOR = "#34C759"  # Green
TEXT_COLOR = "#1C1C1E"


def plot_distribution(df: pd.DataFrame):
    """Vẽ phân phối Fat-Tailed của PM2.5."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Histogram & KDE
    sns.histplot(df["pm25"], kde=True, bins=50, color=PRIMARY_COLOR, alpha=0.6, ax=ax)

    # Metrics
    mean_val = df["pm25"].mean()
    p95 = df["pm25"].quantile(0.95)
    max_val = df["pm25"].max()

    # Annotations
    ax.axvline(mean_val, color=TEXT_COLOR, linestyle="--", lw=2, label=f"Mean: {mean_val:.1f}")
    ax.axvline(p95, color=WARNING_COLOR, linestyle="-", lw=2, label=f"95th Percentile: {p95:.1f}")
    ax.axvline(max_val, color=ERROR_COLOR, linestyle="-", lw=2, label=f"Max: {max_val:.1f}")

    ax.set_title("Phân phối PM2.5: Bất đối xứng & Đuôi dài (Fat-Tailed)", fontweight="bold", pad=15)
    ax.set_xlabel("Nồng độ PM2.5 (µg/m³)")
    ax.set_ylabel("Tần suất (Cảm biến báo cáo)")
    ax.legend()

    # Text box explaining ML difficulty
    text_content = (
        "Thách thức ML:\n"
        "Mô hình thường tối ưu hóa theo Mean (MSE, MAE).\n"
        "Tuy nhiên các đỉnh > 60 µg/m³ (đuôi dài) chứa rủi ro y tế lớn\n"
        "nhưng bị coi là 'nhiễu' (outliers) và thường bị ML dự báo under-estimate."
    )
    props = dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor=WARNING_COLOR)
    ax.text(0.5, 0.4, text_content, transform=ax.transAxes, fontsize=12, verticalalignment="top", bbox=props)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02a_pm25_fat_tailed_distribution.png", dpi=300)
    plt.close()


def plot_spikes_timeline(df: pd.DataFrame):
    """Vẽ chuỗi thời gian 1 tháng kèm cảnh báo mức độ."""
    fig, ax = plt.subplots(figsize=(16, 6))

    # Chọn một tháng có nhiều biến động (Ví dụ: dữ liệu thường nhiễu vào mùa đông/xuân)
    # Tìm tháng có phương sai lớn nhất
    monthly_var = df["pm25"].resample("ME").var()
    if monthly_var.empty:
        return
    worst_month = monthly_var.idxmax()

    # Get exactly that month
    mask = (df.index.year == worst_month.year) & (df.index.month == worst_month.month)
    month_data = df[mask]
    if month_data.empty:
        month_data = df.iloc[: 24 * 30]  # Fallback

    ax.plot(month_data.index, month_data["pm25"], color=TEXT_COLOR, lw=1.5, zorder=2)

    # Add colored bands based on AQI approximations
    ax.axhspan(0, 15, facecolor=SAFE_COLOR, alpha=0.2, label="Tốt (≤15)")
    ax.axhspan(15, 35, facecolor=WARNING_COLOR, alpha=0.2, label="Trung bình (15-35)")
    ax.axhspan(35, 200, facecolor=ERROR_COLOR, alpha=0.2, label="Chỉ số xấu (>35)")

    # Highlight highest spike
    max_idx = month_data["pm25"].idxmax()
    max_val = month_data["pm25"].max()
    ax.annotate(
        "Đỉnh dị thường\n(Kẹt xe / Đốt cháy?)",
        xy=(max_idx, max_val),
        xytext=(max_idx, max_val + 20),
        arrowprops={"facecolor": ERROR_COLOR, "shrink": 0.05},
        horizontalalignment="center",
        color=ERROR_COLOR,
        fontweight="bold",
        zorder=3,
    )

    ax.set_title(
        f"Sự biến động khó lường của PM2.5 trong khoảng thời gian {worst_month.strftime('%m/%Y')}",
        fontweight="bold",
        pad=15,
    )
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.legend(loc="upper right")

    # Formatting date
    date_form = DateFormatter("%d-%m")
    ax.xaxis.set_major_formatter(date_form)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02b_pm25_erratic_spikes.png", dpi=300)
    plt.close()


def main():
    print(f"Loading data from {DATA_PATH}...")
    if not DATA_PATH.exists():
        print("Data not found!")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    df = df.dropna(subset=["pm25"])

    print("Generating distribution plot...")
    plot_distribution(df)

    print("Generating spikes timeline...")
    plot_spikes_timeline(df)

    print(f"SUCCESS! Charts saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
