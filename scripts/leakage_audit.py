"""Quick leakage audit — verifies specific leakage hypotheses."""

import sys
import numpy as np
import pandas as pd

def p(msg):
    """Print with immediate flush."""
    print(msg, flush=True)

p("=" * 60)
p("DATA LEAKAGE AUDIT")
p("=" * 60)

p("\n⏳ Loading marts data...")
# Only load needed columns for speed
needed = [
    "pm25", "pm25_lag_1h", "pm25_diff_1h", "pm25_diff_24h",
    "pm25_pct_change_1h", "pm25_pct_change_24h",
    "co2_pm25_ratio", "pm25_aqi_cat",
    "nhiet_do", "co2", "pm25_ewm_12h_mean",
]
df = pd.read_csv(
    "dataset/processed/marts_features.csv",
    index_col=0,
    parse_dates=True,
    usecols=lambda c: c in needed or c == "ngay_tao",
)
t = df["pm25"]
p(f"✅ Loaded: {len(df):,} rows, {len(df.columns)} cols")

# 1. DIFF LEAKAGE
p("\n--- [1/6] Checking diff_1h leakage ---")
d = df["pm25_diff_1h"]
l = df["pm25_lag_1h"]
recon = d + l
valid = recon.notna()
m = bool(np.allclose(recon[valid], t[valid], rtol=1e-10))
p(f"  pm25 = diff_1h + lag_1h? => {m}")
if m:
    p("  🔴 LEAKAGE CONFIRMED: diff_1h encodes current target!")
else:
    p("  🟢 OK")

# 2. PCT_CHANGE LEAKAGE
p("\n--- [2/6] Checking pct_change_1h leakage ---")
pc = df["pm25_pct_change_1h"]
recon2 = l * (1 + pc)
valid2 = recon2.notna() & (l != 0)
m2 = bool(np.allclose(recon2[valid2], t[valid2], rtol=1e-8))
p(f"  pm25 = lag_1h * (1+pct_change)? => {m2}")
if m2:
    p("  🔴 LEAKAGE CONFIRMED: pct_change_1h encodes current target!")
else:
    p("  🟢 OK")

# 3. co2_pm25_ratio
p("\n--- [3/6] Checking co2_pm25_ratio ---")
if "co2_pm25_ratio" in df.columns:
    expected = df["co2"] / t.replace(0, float("nan"))
    valid3 = expected.notna() & df["co2_pm25_ratio"].notna()
    m3 = bool(np.allclose(df["co2_pm25_ratio"][valid3], expected[valid3], rtol=1e-8))
    p(f"  co2_pm25_ratio = co2/pm25[t]? => {m3}")
    if m3:
        p("  🔴 LEAKAGE CONFIRMED: uses current target!")
    else:
        p("  🟢 OK")
else:
    p("  ⏭️ Column not found")

# 4. pm25_aqi_cat
p("\n--- [4/6] Checking pm25_aqi_cat ---")
if "pm25_aqi_cat" in df.columns:
    corr_aqi = float(df["pm25_aqi_cat"].corr(t))
    p(f"  corr(pm25_aqi_cat, pm25) = {corr_aqi:.4f}")
    if abs(corr_aqi) > 0.9:
        p("  🔴 LEAKAGE: high corr, uses pm25[t] for binning")
    else:
        p("  🟢 OK")
else:
    p("  ⏭️ Column not found")

# 5. Key correlations
p("\n--- [5/6] Key feature correlations ---")
for c in ["pm25_lag_1h", "pm25_diff_1h", "pm25_pct_change_1h",
          "co2_pm25_ratio", "pm25_aqi_cat", "nhiet_do", "co2"]:
    if c in df.columns:
        corr = float(df[c].corr(t))
        flag = " ⚠️" if abs(corr) > 0.95 else ""
        p(f"  {c:35s} corr={corr:.4f}{flag}")

# 6. Autocorrelation
p("\n--- [6/6] PM2.5 autocorrelation ---")
for lag in [1, 2, 3, 6, 12, 24]:
    p(f"  lag={lag:3d}h: {t.autocorr(lag):.4f}")

p("\n" + "=" * 60)
p("✅ AUDIT COMPLETE")
p("=" * 60)
