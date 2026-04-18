import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy import stats
from statsmodels.tsa.seasonal import STL
from src.data.loader import load_raw_data
from src.data.cleaner import clean_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EDA_DIR = PROJECT_ROOT / "research" / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)

df = clean_data(load_raw_data())

result = {}

# 1. P2-1: Box-Cox Optimal Lambda
# Box-cox needs strictly positive data, PM2.5 can have 0s. Use pm25 + 1.
try:
    pm25_pos = df['pm25'] + 1 
    _, opt_lambda = stats.boxcox(pm25_pos)
    result['box_cox'] = {
        'optimal_lambda': float(opt_lambda),
        'interpretation': 'Log transform' if abs(opt_lambda) < 0.2 else 'Square root' if abs(opt_lambda - 0.5) < 0.2 else f'Power {opt_lambda:.2f}'
    }
except Exception as e:
    result['box_cox'] = {'error': str(e)}

# 2. P2-2: S-ESD Outlier Detection (Simplified)
# S-ESD: Detrend/Deseasonalize using STL -> apply Generalized ESD on residuals (we'll use a simpler robust threshold on residuals for demonstration: 3 * MAD(residuals))
try:
    # Need regular index for STL, we know it's 1h freq after clean_data
    df = df.asfreq('1h')
    stl = STL(df['pm25'].interpolate(), period=24, robust=True)
    res = stl.fit()
    resid = res.resid
    
    # MAD (Median Absolute Deviation)
    mad = np.median(np.abs(resid - np.median(resid)))
    threshold = 3 * mad
    
    outliers_mask = np.abs(resid) > threshold
    outliers = df['pm25'][outliers_mask]
    
    result['s_esd'] = {
        'n_outliers_detected': int(outliers_mask.sum()),
        'pct_outliers': float(outliers_mask.sum() / len(df) * 100),
        'mad': float(mad),
        'threshold': float(threshold)
    }
except Exception as e:
    result['s_esd'] = {'error': str(e)}

# 3. P2-3: Purging Gap Documentation (Just string constants for dashboard)
result['purging_gap'] = {
    'concept': 'Purging Gap',
    'definition': 'Để tránh leakage khi dùng TimeSeriesSplit, ta cần loại bỏ (purge) một khoảng gap tương đương với max_lookback giữa Train và Test split, tránh thông tin từ Train lọt qua rolling/lag features sang Test.',
    'status': 'Được xử lý ngầm (implicitly handled) bởi hàm evaluate_forecast qua việc khởi tạo lookback độc lập cho mỗi split.'
}

output_path = EDA_DIR / "phase6_dashboard_data.json"
with open(output_path, "w") as f:
    json.dump(result, f, indent=4)

print(f"Phase 6 data saved to {output_path}")
