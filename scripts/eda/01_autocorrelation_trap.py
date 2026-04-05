"""
Bẫy Tự Tương Quan (The Autocorrelation Trap)
Mục tiêu: Chứng minh tại sao Baseline (Persistence) lại rất mạnh ở h=1 và cực yếu ở h=24.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from matplotlib.gridspec import GridSpec

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "dataset" / "interim" / "cleaned_hourly.csv"
OUTPUT_DIR = PROJECT_ROOT / "research" / "eda" / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Styling for premium storytelling look (Apple HIG / modern data sci style)
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("talk")
PRIMARY_COLOR = "#007AFF" # Apple blue
SECONDARY_COLOR = "#FF3B30" # Apple red
TEXT_COLOR = "#1C1C1E"

def plot_autocorrelation_story(df: pd.DataFrame):
    """Vẽ ACF/PACF để thấy tín hiệu giảm mạnh sau 24h."""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # 1. Plot PM2.5 Time Series Sample (First 2 weeks)
    ax0 = fig.add_subplot(gs[0, :])
    sample_data = df['pm25'].iloc[:24*14]
    ax0.plot(sample_data.index, sample_data.values, color=PRIMARY_COLOR, lw=2)
    ax0.set_title("Biến động PM2.5 (Trích xuất 2 tuần mẫu)", fontweight='bold', pad=15, color=TEXT_COLOR)
    ax0.set_ylabel("PM2.5 (µg/m³)")
    
    # 2. ACF Plot
    ax1 = fig.add_subplot(gs[1, 0])
    plot_acf(df['pm25'], lags=72, ax=ax1, color=PRIMARY_COLOR, alpha=0.05, 
             title="Tự tương quan (ACF) - Lags up to 72 hours")
    
    # Emphasize specific lags
    ax1.axvline(x=24, color=SECONDARY_COLOR, linestyle='--', alpha=0.5, label='24h (Daily Seasonality)')
    ax1.axvline(x=1, color='green', linestyle=':', lw=3, label='1h (High Correlation Trap)')
    ax1.legend()
    
    # 3. PACF Plot
    ax2 = fig.add_subplot(gs[1, 1])
    plot_pacf(df['pm25'], lags=72, ax=ax2, color=PRIMARY_COLOR, method='ywm', alpha=0.05,
              title="Tự tương quan riêng phần (PACF)")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01a_autocorrelation_memory.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_horizon_scatter(df: pd.DataFrame):
    """Vẽ Scatter matrix giữa y_t và y_{t-h} để thấy sự xói mòn theo horizon."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Create lagged features
    df_lag = pd.DataFrame({'y_t': df['pm25']})
    df_lag['y_t_minus_1'] = df_lag['y_t'].shift(1)
    df_lag['y_t_minus_6'] = df_lag['y_t'].shift(6)
    df_lag['y_t_minus_24'] = df_lag['y_t'].shift(24)
    df_lag = df_lag.dropna()
    
    # Plot configurations
    settings = [
        (0, 'y_t_minus_1', 'Horizon = 1h (Rất chật, tuyến tính mạnh)', PRIMARY_COLOR),
        (1, 'y_t_minus_6', 'Horizon = 6h (Bắt đầu phân tán rải rác)', '#FF9500'),
        (2, 'y_t_minus_24', 'Horizon = 24h (Rời rạc, dạng đám mây)', SECONDARY_COLOR)
    ]
    
    for ax_idx, col, title, color in settings:
        ax = axes[ax_idx]
        corr = df_lag['y_t'].corr(df_lag[col])
        
        # Hexbin thay vì scatter thông thường để tránh overplotting
        hb = ax.hexbin(df_lag[col], df_lag['y_t'], gridsize=40, cmap="Blues", bins='log', mincnt=1)
        
        # Perfect prediction line
        max_val = max(df_lag['y_t'].max(), df_lag[col].max())
        ax.plot([0, max_val], [0, max_val], 'r--', lw=2, alpha=0.7)
        
        ax.set_title(f"{title}\nCorr = {corr:.3f}", fontweight='bold', pad=15)
        ax.set_xlabel(f"PM2.5 tại Thời điểm T - {col.split('_')[-1]}h")
        ax.set_ylabel("PM2.5 tại Thời điểm T")
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 150) # Capping
        ax.set_ylim(0, 150)
        
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01b_horizon_scatter_dispersion.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print(f"Loading data from {DATA_PATH}...")
    if not DATA_PATH.exists():
        print("Data not found! Please run make data or ensure cleaned_hourly.csv exists.")
        sys.exit(1)
        
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    
    print("Generating ACF/PACF plots...")
    plot_autocorrelation_story(df)
    
    print("Generating Horizon Scatter Disperion...")
    plot_horizon_scatter(df)
    
    print(f"SUCCESS! Charts saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
