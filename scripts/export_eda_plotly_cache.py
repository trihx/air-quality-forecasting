import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
import scipy.stats as stats
import scipy.signal as signal
from statsmodels.tsa.stattools import acf, pacf, grangercausalitytests, ccf
from statsmodels.tsa.seasonal import STL

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "dataset"
CACHE_DIR = PROJECT_ROOT / "research" / "eda" / "plotly_cache"

CACHE_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    print("Loading data...")
    # Raw data for missing analysis
    raw = pd.read_csv(DATA_DIR / "raw" / "final_dataset.csv", usecols=["ngay_tao", "pm25"], parse_dates=["ngay_tao"])
    raw.set_index("ngay_tao", inplace=True)
    raw_h = raw.resample('1h').mean()
    
    # Imputed / Clean data
    df_clean = pd.read_csv(DATA_DIR / "interim" / "cleaned_hourly.csv", parse_dates=["ngay_tao"])
    df_clean.set_index("ngay_tao", inplace=True)
    
    return raw_h, df_clean

def export_missing_barcode(raw_h, df_clean):
    print("Exporting missing barcode...")
    # Take last 2000 hours to match current UI logic
    raw_sample = raw_h.iloc[-2000:]
    
    data = {
        "index": raw_sample.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
        "is_missing": raw_sample["pm25"].isna().tolist()
    }
    
    # Specific imputation comparison window (2022-03-15 to 2022-03-18)
    try:
        w_raw = raw_h.loc['2022-03-15 12:00:00':'2022-03-18 12:00:00', "pm25"]
        idx = w_raw.index.intersection(df_clean.index)
        w_raw = w_raw.loc[idx]
        w_imp = df_clean.loc[idx, "pm25"]
        data["imputation_comp"] = {
            "index": idx.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            "raw": w_raw.where(w_raw.notna(), None).tolist(),
            "imp": w_imp.tolist()
        }
    except KeyError as e:
        print(f"Skipping imputation_comp: {e}")
        
    with open(CACHE_DIR / "missing_barcode.json", "w") as f:
        json.dump(data, f)

def export_distributions(df_clean):
    print("Exporting distributions...")
    data = {}
    features = ["pm25", "nhiet_do", "do_am", "diem_suong", "co2"]
    
    for f in features:
        if f in df_clean.columns:
            vals = df_clean[f].dropna().values
            # Gaussian KDE
            kde = stats.gaussian_kde(vals)
            x_grid = np.linspace(vals.min(), vals.max(), 100)
            pdf = kde(x_grid)
            
            data[f] = {
                "x": x_grid.tolist(),
                "pdf": pdf.tolist(),
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals))
            }
            if f == "pm25":
                data[f]["percentiles"] = {
                    "95th": float(np.percentile(vals, 95)),
                    "99th": float(np.percentile(vals, 99)),
                    "max": float(np.max(vals))
                }
                
    with open(CACHE_DIR / "distributions.json", "w") as f:
        json.dump(data, f)

def export_qq_plot(df_clean):
    print("Exporting QQ Plot...")
    vals = df_clean["pm25"].dropna().values
    
    # Raw QQ
    osm, osr = stats.probplot(vals, dist="norm")
    
    # Log-transformed QQ
    vals_log = np.log1p(vals)
    osm_log, osr_log = stats.probplot(vals_log, dist="norm")
    
    data = {
        "raw": {"theo": osm[0].tolist(), "sample": osm[1].tolist()},
        "log": {"theo": osm_log[0].tolist(), "sample": osm_log[1].tolist()}
    }
    with open(CACHE_DIR / "qq_plot.json", "w") as f:
        json.dump(data, f)

def export_acf_pacf(df_clean):
    print("Exporting ACF/PACF...")
    vals = df_clean["pm25"].dropna().values
    nlags = 100
    
    acf_vals, acf_conf = acf(vals, nlags=nlags, alpha=0.05)
    pacf_vals, pacf_conf = pacf(vals, nlags=nlags, alpha=0.05)
    
    data = {
        "acf": acf_vals.tolist(),
        "acf_conf_lower": (acf_conf[:, 0] - acf_vals).tolist(),
        "acf_conf_upper": (acf_conf[:, 1] - acf_vals).tolist(),
        "pacf": pacf_vals.tolist(),
        "pacf_conf_lower": (pacf_conf[:, 0] - pacf_vals).tolist(),
        "pacf_conf_upper": (pacf_conf[:, 1] - pacf_vals).tolist()
    }
    with open(CACHE_DIR / "acf_pacf.json", "w") as f:
        json.dump(data, f)

def export_stl(df_clean):
    print("Exporting STL...")
    # Fill any remaining NaNs just in case
    ts = df_clean["pm25"].interpolate(method='linear')
    res = STL(ts, period=24).fit()
    
    # To save space, maybe downsample to every 4 hours for the full view
    # But STL results are best viewed in full, so we export full, but limit precision
    data = {
        "index": ts.index.strftime('%Y-%m-%d %H').tolist(),
        "original": np.round(ts.values, 2).tolist(),
        "trend": np.round(res.trend.values, 2).tolist(),
        "seasonal": np.round(res.seasonal.values, 2).tolist(),
        "resid": np.round(res.resid.values, 2).tolist()
    }
    with open(CACHE_DIR / "stl.json", "w") as f:
        json.dump(data, f)

def export_psd(df_clean):
    print("Exporting PSD...")
    vals = df_clean["pm25"].interpolate(method='linear').values
    freqs, pxx = signal.welch(vals, fs=1.0, nperseg=24*30) # 1 month window
    
    data = {
        "freqs": freqs.tolist(),
        "power": pxx.tolist()
    }
    with open(CACHE_DIR / "psd.json", "w") as f:
        json.dump(data, f)

def export_correlations(df_clean):
    print("Exporting correlations...")
    data = {}
    
    # Concept drift (rolling correlation 60 days)
    if "nhiet_do" in df_clean.columns:
        roll_corr = df_clean["pm25"].rolling(window=24*60).corr(df_clean["nhiet_do"])
        data["rolling_corr_pm25_nhiet_do"] = {
            "index": roll_corr.dropna().index.strftime('%Y-%m-%d').tolist(),
            "corr": np.round(roll_corr.dropna().values, 3).tolist()
        }
        
        # Cross correlation
        ccf_vals = ccf(df_clean["pm25"].values, df_clean["nhiet_do"].values)[:48]
        data["cross_corr_nhiet_do"] = {
            "lags": list(range(48)),
            "corr": np.round(ccf_vals, 3).tolist()
        }
        
    # Granger Causality (p-values)
    print("Computing Granger causality p-values...")
    gc_pvalues = {}
    maxlag = 24
    for exog in ["nhiet_do", "do_am", "co2"]:
        if exog in df_clean.columns:
            # Drop na for the test
            df_gc = df_clean[["pm25", exog]].dropna()
            # Granger causality: Does exog cause pm25?
            # Data must be 2D array, first column is the predicted variable (pm25), second is the predictor
            try:
                gc_res = grangercausalitytests(df_gc[["pm25", exog]], maxlag=maxlag, verbose=False)
                pvals = [gc_res[i+1][0]['ssr_ftest'][1] for i in range(maxlag)]
                gc_pvalues[exog] = np.round(pvals, 6).tolist()
            except Exception as e:
                print(f"Failed GC for {exog}: {e}")
                
    if gc_pvalues:
        data["granger_causality"] = {
            "lags": list(range(1, maxlag + 1)),
            "pvalues": gc_pvalues
        }
    
    with open(CACHE_DIR / "correlations.json", "w") as f:
        json.dump(data, f)

def main():
    raw_h, df_clean = load_data()
    export_missing_barcode(raw_h, df_clean)
    export_distributions(df_clean)
    export_qq_plot(df_clean)
    export_acf_pacf(df_clean)
    export_stl(df_clean)
    export_psd(df_clean)
    export_correlations(df_clean)
    print("Done! All EDA plotly cache JSONs generated.")

if __name__ == "__main__":
    main()
