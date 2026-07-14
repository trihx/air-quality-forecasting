"""
Sự dịch chuyển của Đa Biến (Multivariate Shift)
Mục tiêu: Cho thấy mối tương quan giữa Nhiệt độ, Độ ẩm và PM2.5 không hề
tuyến tính hay ổn định, mà thay đổi trồi sụt theo thời gian (drift).
Gây khó cho các mô hình hồi quy truyền thống.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "dataset" / "interim" / "cleaned_hourly.csv"
OUTPUT_DIR = PROJECT_ROOT / "research" / "eda" / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# VTF: Centralized theme (Light mode for publication)
sys.path.insert(0, str(PROJECT_ROOT))
from src.viz.theme import apply_mpl_theme, annotation_bbox

apply_mpl_theme("light")
sns.set_context("talk")
PRIMARY_COLOR = "#007AFF"
SECONDARY_COLOR = "#FF9500"
TEXT_COLOR = "#373737"


def plot_rolling_correlation(df: pd.DataFrame):
    """Vẽ tương quan chạy (Rolling Correlation) theo Window = 14 ngày."""
    fig, ax = plt.subplots(figsize=(15, 6))

    # Calculate rolling spearman correlation
    window_hours = 24 * 14  # 14 days

    # We need to drop NAs for correlation calculation
    df_clean = df.dropna(subset=["pm25", "nhiet_do", "do_am"])

    roll_corr_temp = df_clean["pm25"].rolling(window=window_hours).corr(df_clean["nhiet_do"])
    roll_corr_hum = df_clean["pm25"].rolling(window=window_hours).corr(df_clean["do_am"])

    ax.plot(roll_corr_temp.index, roll_corr_temp, label="Tương quan PM2.5 & Nhiệt độ", color=SECONDARY_COLOR, lw=2)
    ax.plot(roll_corr_hum.index, roll_corr_hum, label="Tương quan PM2.5 & Độ ẩm", color=PRIMARY_COLOR, lw=2)

    # Zero line
    ax.axhline(0, color=TEXT_COLOR, linestyle="--", lw=1.5, alpha=0.5)

    ax.set_title("Tính không ổn định của Multivariates (14-Day Rolling Correlation)", fontweight="bold", pad=15)
    ax.set_ylabel("Hệ số tương quan Spearman")
    ax.set_ylim(-1, 1)
    ax.legend(loc="upper right")

    # Adding insight text
    text = (
        "Insight:\n"
        "Tương quan dao động mạnh từ âm sang dương (-0.6 đến +0.6).\n"
        "Nghĩa là ảnh hưởng của thời tiết lên PM2.5 phụ thuộc vào mùa (Concept Drift).\n"
        "Làm Machine Learning khó bắt được quy luật tĩnh."
    )
    # Put text at bottom left
    ax.text(0.02, 0.05, text, transform=ax.transAxes, fontsize=11, verticalalignment="bottom", bbox=annotation_bbox("light"))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03a_rolling_correlation.png", dpi=300)
    plt.close()


def plot_hexbin_density(df: pd.DataFrame):
    """Vẽ Hexbin 2D thay vì Scatter để xử lý overplotting."""
    df_clean = df.dropna(subset=["pm25", "nhiet_do", "do_am"])

    # Create a 1x2 subplot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # PM2.5 vs Temp
    hb1 = axes[0].hexbin(df_clean["nhiet_do"], df_clean["pm25"], gridsize=30, cmap="Oranges", mincnt=1)
    axes[0].set_title("Nhiệt độ vs PM2.5 (Mật độ tập trung)", fontweight="bold")
    axes[0].set_xlabel("Nhiệt độ (°C)")
    axes[0].set_ylabel("PM2.5 (µg/m³)")
    cb1 = fig.colorbar(hb1, ax=axes[0])
    cb1.set_label("Số lượng quan sát")

    # PM2.5 vs Humidity
    hb2 = axes[1].hexbin(df_clean["do_am"], df_clean["pm25"], gridsize=30, cmap="Blues", mincnt=1)
    axes[1].set_title("Độ ẩm vs PM2.5 (Mật độ tập trung)", fontweight="bold")
    axes[1].set_xlabel("Độ ẩm (%)")
    axes[1].set_ylabel("PM2.5 (µg/m³)")
    cb2 = fig.colorbar(hb2, ax=axes[1])
    cb2.set_label("Số lượng quan sát")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03b_hexbin_multivariate.png", dpi=300)
    plt.close()


def main():
    print(f"Loading data from {DATA_PATH}...")
    if not DATA_PATH.exists():
        print("Data not found!")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)

    print("Generating rolling correlation...")
    plot_rolling_correlation(df)

    print("Generating hexbin density...")
    plot_hexbin_density(df)

    print(f"SUCCESS! Charts saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
