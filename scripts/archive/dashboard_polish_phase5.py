import pandas as pd
import numpy as np
import json
from pathlib import Path
from src.data.loader import load_raw_data
from src.data.cleaner import clean_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EDA_DIR = PROJECT_ROOT / "research" / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)

df = clean_data(load_raw_data())

result = {}

# 1. Complexity Profile
eda_json_path = EDA_DIR / "eda_results.json"
if eda_json_path.exists():
    with open(eda_json_path, "r", encoding="utf-8") as f:
        eda_data = json.load(f)
    
    fc = eda_data.get("forecastability", {})
    stl = eda_data.get("stl", {})
    
    metrics = ['CoV', 'ApEn', 'Seasonality', 'Trend', 'NoiseRatio']
    
    cov_val = min(fc.get('cov', 0) / 2.0, 1.0)
    apen_val = min(fc.get('approximate_entropy', 0) / 2.0, 1.0)
    seas_val = min(stl.get('seasonal_strength', 0), 1.0)
    trend_val = min(stl.get('trend_strength', 0), 1.0)
    noise_val = min(stl.get('noise_ratio', 0), 1.0)
    
    result['complexity_radar'] = {
        'metrics': metrics,
        'values': [cov_val, apen_val, seas_val, trend_val, noise_val]
    }

# 2. Walk-Forward Stability chart
monthly_stats = df.resample('ME').agg({'pm25': ['mean', 'std']})
monthly_stats.columns = ['mean', 'std']
monthly_stats = monthly_stats.dropna()
result['walk_forward'] = {
    'dates': monthly_stats.index.astype(str).tolist(),
    'mean': monthly_stats['mean'].tolist(),
    'std': monthly_stats['std'].tolist()
}

# 3. Expanding Window Stats
df['expanding_mean'] = df['pm25'].expanding().mean()
df['expanding_std'] = df['pm25'].expanding().std()

plot_df = df.iloc[::24].copy() # Daily downsample
plot_df = plot_df.dropna()

result['expanding_window'] = {
    'dates': plot_df.index.astype(str).tolist(),
    'pm25_raw': plot_df['pm25'].tolist(),
    'expanding_mean': plot_df['expanding_mean'].tolist(),
    'expanding_std': plot_df['expanding_std'].tolist()
}

output_path = EDA_DIR / "phase5_dashboard_data.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(result, f)

print(f"Phase 5 data saved to {output_path}")
