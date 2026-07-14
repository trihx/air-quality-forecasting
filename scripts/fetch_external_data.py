"""
Script: Fetch external data from Open-Meteo APIs for missing periods in the IoT dataset.
Location: Can Tho area (10.290796, 105.767080)
Sources:
  - Historical Weather API: temperature_2m, relative_humidity_2m, dew_point_2m
  - Air Quality API: pm2_5 (CAMS Global, only available for recent data ~2024+)

Output: dataset/external/open_meteo_missing_periods.csv
Columns match: nhiet_do, do_am, diem_suong, co2, pm25, ngay_tao, source
"""

import pandas as pd
import requests
import time
import sys
import io
from pathlib import Path
from datetime import datetime, timedelta

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ── Config ──
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_PATH = PROJECT_ROOT / "dataset" / "raw" / "final_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "external"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "open_meteo_missing_periods.csv"

LAT = 10.290796
LON = 105.767080
TIMEZONE = "Asia/Ho_Chi_Minh"

# Open-Meteo API endpoints
WEATHER_API = "https://archive-api.open-meteo.com/v1/archive"
AQ_API = "https://air-quality-api.open-meteo.com/v1/air-quality"

# ── Step 1: Load existing dataset and find missing hours ──
print("[1/5] Loading existing dataset...", flush=True)
df = pd.read_csv(DATASET_PATH, parse_dates=["ngay_tao"], usecols=["ngay_tao", "pm25"])
df = df.dropna(subset=["ngay_tao", "pm25"])
df.set_index("ngay_tao", inplace=True)
df = df.sort_index()

# Resample to hourly
hourly = df.resample("1h").mean()
print(f"  Total hourly slots: {len(hourly)}", flush=True)
print(f"  Missing PM2.5 hours: {hourly.pm25.isna().sum()}", flush=True)
print(f"  Date range: {hourly.index.min()} -> {hourly.index.max()}", flush=True)

# Find contiguous missing blocks (only gaps >= 24h to focus on significant gaps)
is_missing = hourly["pm25"].isna()
blocks = []
start = None
for ts, missing in is_missing.items():
    if missing and start is None:
        start = ts
    elif not missing and start is not None:
        gap_hours = (ts - start).total_seconds() / 3600
        if gap_hours >= 24:  # Only fetch for gaps >= 1 day
            blocks.append((start, ts - pd.Timedelta(hours=1)))
        start = None
if start is not None:
    gap_hours = (hourly.index[-1] - start).total_seconds() / 3600
    if gap_hours >= 24:
        blocks.append((start, hourly.index[-1]))

print(f"\n[2/5] Found {len(blocks)} significant gaps (>= 24h) to fill.", flush=True)
total_missing_hours = sum((e - s).total_seconds() / 3600 + 1 for s, e in blocks)
print(f"  Total hours to fetch: {total_missing_hours:.0f}", flush=True)


# ── Step 2: Fetch weather data from Open-Meteo in chunks ──
def fetch_weather_chunk(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch historical weather data for a date range."""
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,dew_point_2m",
        "timezone": TIMEZONE,
    }
    resp = requests.get(WEATHER_API, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    hourly_data = data.get("hourly", {})
    if not hourly_data.get("time"):
        return pd.DataFrame()

    result = pd.DataFrame({
        "ngay_tao": pd.to_datetime(hourly_data["time"]),
        "nhiet_do": hourly_data.get("temperature_2m"),
        "do_am": hourly_data.get("relative_humidity_2m"),
        "diem_suong": hourly_data.get("dew_point_2m"),
    })
    return result


def fetch_aq_chunk(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch air quality data (PM2.5) for a date range using CAMS Global."""
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "pm2_5",
        "timezone": TIMEZONE,
        "domains": "cams_global",
    }
    try:
        resp = requests.get(AQ_API, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        hourly_data = data.get("hourly", {})
        if not hourly_data.get("time"):
            return pd.DataFrame()

        result = pd.DataFrame({
            "ngay_tao": pd.to_datetime(hourly_data["time"]),
            "pm25_external": hourly_data.get("pm2_5"),
        })
        return result
    except Exception as e:
        print(f"    [WARN] AQ API failed: {e}", flush=True)
        return pd.DataFrame()


# ── Step 3: Iterate through gaps and fetch data ──
print("\n[3/5] Fetching data from Open-Meteo APIs...", flush=True)
all_weather = []
all_aq = []

for i, (gap_start, gap_end) in enumerate(blocks):
    start_str = gap_start.strftime("%Y-%m-%d")
    end_str = gap_end.strftime("%Y-%m-%d")
    gap_hours = (gap_end - gap_start).total_seconds() / 3600 + 1
    print(f"  Gap {i+1}/{len(blocks)}: {start_str} -> {end_str} ({gap_hours:.0f}h = {gap_hours/24:.1f} days)", flush=True)

    # Open-Meteo allows up to ~365 days per request, split large ranges
    chunk_start = gap_start
    while chunk_start <= gap_end:
        chunk_end = min(chunk_start + timedelta(days=89), gap_end)
        cs = chunk_start.strftime("%Y-%m-%d")
        ce = chunk_end.strftime("%Y-%m-%d")

        # Weather data
        try:
            wx = fetch_weather_chunk(cs, ce)
            if not wx.empty:
                all_weather.append(wx)
                print(f"    Weather OK: {cs} -> {ce} ({len(wx)} rows)", flush=True)
        except Exception as e:
            print(f"    [ERR] Weather failed {cs}->{ce}: {e}", flush=True)

        # Air Quality (PM2.5) - might return nulls for historical data
        try:
            aq = fetch_aq_chunk(cs, ce)
            if not aq.empty:
                all_aq.append(aq)
                non_null = aq["pm25_external"].notna().sum()
                print(f"    AQ OK: {cs} -> {ce} ({non_null}/{len(aq)} non-null PM2.5)", flush=True)
        except Exception as e:
            print(f"    [WARN] AQ skipped: {e}", flush=True)

        chunk_start = chunk_end + timedelta(days=1)

    # Rate limit: be nice to the free API
    time.sleep(0.5)


# ── Step 4: Merge and format ──
print("\n[4/5] Merging and formatting data...", flush=True)

if all_weather:
    df_weather = pd.concat(all_weather, ignore_index=True)
else:
    df_weather = pd.DataFrame(columns=["ngay_tao", "nhiet_do", "do_am", "diem_suong"])

if all_aq:
    df_aq = pd.concat(all_aq, ignore_index=True)
    df_merged = df_weather.merge(df_aq, on="ngay_tao", how="left")
else:
    df_merged = df_weather.copy()
    df_merged["pm25_external"] = None

# Round to match IoT sensor precision
df_merged["nhiet_do"] = df_merged["nhiet_do"].round(1)
df_merged["do_am"] = df_merged["do_am"].round(1)
df_merged["diem_suong"] = df_merged["diem_suong"].round(1)

# Add columns to match the original dataset schema
df_merged["co2"] = None  # CO2 not available from Open-Meteo
df_merged["pm25"] = df_merged["pm25_external"]  # May be null for historical
df_merged["source"] = "open_meteo"  # Tag for traceability

# Reorder columns to match original + source tag
df_final = df_merged[["nhiet_do", "do_am", "diem_suong", "co2", "pm25", "ngay_tao", "source"]].copy()

# Drop duplicates (in case of overlapping chunks)
df_final = df_final.drop_duplicates(subset=["ngay_tao"]).sort_values("ngay_tao").reset_index(drop=True)

# ── Step 5: Save ──
print("\n[5/5] Saving to CSV...", flush=True)
df_final.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

# Summary
pm25_available = df_final["pm25"].notna().sum()
pm25_total = len(df_final)
weather_available = df_final["nhiet_do"].notna().sum()

print(f"\n{'='*60}")
print(f"DONE! Output: {OUTPUT_PATH}")
print(f"  Total rows: {len(df_final)}")
print(f"  Weather data (nhiet_do/do_am/diem_suong): {weather_available}/{pm25_total} rows")
print(f"  PM2.5 data (from CAMS Global): {pm25_available}/{pm25_total} rows")
print(f"  CO2: Not available (sensor-specific)")
print(f"  Date range: {df_final['ngay_tao'].min()} -> {df_final['ngay_tao'].max()}")
print(f"{'='*60}")
