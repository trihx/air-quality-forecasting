"""
Khoảng trống chất lượng (Data Quality Gaps)
Mục tiêu: Kể câu chuyện về sự không hoàn hảo của IoT Sensor. Mất điện, rớt mạng tạo ra
những khoảng trống dữ liệu dai dẳng. Tại sao Linear Interpolation thất bại.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_PATH = PROJECT_ROOT / "dataset" / "raw" / "final_dataset.csv"
JSON_REPORT = PROJECT_ROOT / "research" / "eda" / "gap_analysis_report.json"
OUTPUT_DIR = PROJECT_ROOT / "research" / "eda" / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# VTF: Centralized theme (Light mode for publication)
sys.path.insert(0, str(PROJECT_ROOT))
from src.viz.theme import apply_mpl_theme, annotation_bbox

apply_mpl_theme("light")
sns.set_context("talk")
PRIMARY_COLOR = "#007AFF"
ERROR_COLOR = "#FF3B30"


def plot_missing_barcode(df: pd.DataFrame):
    """Vẽ mã vạch (barcode) thể hiện các thời điểm bị mất dữ liệu."""
    fig, ax = plt.subplots(figsize=(16, 3))

    # 0 = missing, 1 = present
    missing_mask = df["pm25"].isna().astype(int)

    # Plot vertical lines for missing data
    missing_times = df[missing_mask == 1].index

    # We use a trick: scatter plot with large markers shaped like vertical lines
    ax.vlines(missing_times, ymin=0, ymax=1, color=ERROR_COLOR, alpha=0.5, linewidth=1)

    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_title("Bản đồ khoảng trống (Missing Data Barcode)", fontweight="bold", pad=10)
    ax.set_xlabel("Thời gian")

    # Text annotation
    missing_pct = missing_mask.sum() / len(missing_mask) * 100
    ax.text(
        0.01,
        0.5,
        f"Tổng số dữ liệu bị thiếu: {missing_pct:.1f}%",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        bbox=annotation_bbox("light"),
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04a_missing_barcode.png", dpi=300)
    plt.close()


def plot_recovery_bar(report_data: dict):
    """Vẽ biểu đồ các loại gap."""
    gap_dist = report_data.get("gap_distribution", {})
    if not gap_dist:
        return


    for _k, _v in gap_dist.items():
        # Clean labels and extract non-cumulative bins if possible,
        # but gap_dist is cumulative by default from analyze_gaps script.
        pass  # Will simplify and just pull directly mapping

    # Manual extract for story
    # Total missing = 5897 (example from local data knowledge if json empty)
    total_missing = report_data.get("hourly_missing", 5000)

    hours_recoverable = gap_dist.get("≤24h (daily)", {}).get("hours_recoverable", 0)
    if hours_recoverable == 0:
        return  # Fallback guard

    hours_unrecoverable = total_missing - hours_recoverable

    fig, ax = plt.subplots(figsize=(8, 6))

    categories = ["Có thể phục hồi\n(Gaps ≤ 24h)", "Không thể phục hồi\n(Gaps > 24h)"]
    values = [hours_recoverable, hours_unrecoverable]
    colors_bar = [PRIMARY_COLOR, ERROR_COLOR]

    bars = ax.bar(categories, values, color=colors_bar, alpha=0.8)

    # Add counts above bars
    for bar in bars:
        yval = bar.get_height()
        pct = yval / total_missing * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 20,
            f"{yval:,}h\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.set_title("Năng lực nội suy dữ liệu (Interpolation Recovery Limit)", fontweight="bold", pad=15)
    ax.set_ylabel("Số giờ bị mất")

    # Explanation
    ax.text(
        0.5,
        0.4,
        (
            "Khoảng trống > 24h phá vỡ tính liên tục của mùa (seasonality).\n"
            "Bắt buộc phải Drop thay vì FillNa để tránh Data Leakage/Ảo giác."
        ),
        transform=ax.transAxes,
        fontsize=11,
        ha="center",
        bbox=annotation_bbox("light"),
    )

    # Make top margin higher for text
    ax.set_ylim(0, max(values) * 1.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04b_recovery_limits.png", dpi=300)
    plt.close()


def main():
    print("Loading Gap Report...")
    if not JSON_REPORT.exists():
        print("Run analyze_gaps.py first!")
        sys.exit(1)

    with open(JSON_REPORT, encoding="utf-8") as f:
        report = json.load(f)

    print(f"Loading raw data for barcode from {RAW_PATH}...")
    df = pd.read_csv(RAW_PATH, parse_dates=["ngay_tao"])
    df = df.set_index("ngay_tao").resample("1h").mean()

    print("Generating barcode plot...")
    plot_missing_barcode(df)

    print("Generating recovery limits plot...")
    plot_recovery_bar(report)

    print(f"SUCCESS! Charts saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
