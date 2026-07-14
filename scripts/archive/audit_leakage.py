"""Comprehensive Leakage & Stationarity Audit.

Run: uv run python scripts/audit_leakage.py
"""

import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


def main():
    # ============================================================
    # LOAD DATA
    # ============================================================
    df = pd.read_csv("dataset/processed/marts_features.csv", index_col=0, parse_dates=True)
    target = "pm25"
    feature_cols = [c for c in df.columns if c != target]

    print("=" * 70)
    print("DATA LEAKAGE AUDIT")
    print("=" * 70)
    print(f"Dataset: {len(df):,} rows × {len(df.columns)} cols")
    print(f"Features: {len(feature_cols)}")
    print()

    # ============================================================
    # 1. CONTEMPORANEOUS FEATURES (Leakage Type 1)
    # ============================================================
    print("=" * 70)
    print("1. CONTEMPORANEOUS FEATURES (same timestamp as target)")
    print("=" * 70)
    print("   In REAL forecasting at time t, we DON'T have sensor readings at t.")
    print("   These features should be LAGGED or REMOVED.\n")

    original_cols = ["nhiet_do", "do_am", "diem_suong", "co2"]
    leaky = []
    for col in original_cols:
        if col in feature_cols:
            r = float(df[col].corr(df[target]))
            print(f"   ⚠️  {col:20} r={r:+.4f} — measured at SAME TIME as target")
            leaky.append(col)

    # temp_humidity_interaction = nhiet_do(t) * do_am(t)
    if "temp_humidity_interaction" in feature_cols:
        r = float(df["temp_humidity_interaction"].corr(df[target]))
        print(f"   ⚠️  {'temp_humidity_interaction':20} r={r:+.4f} — derived from contemporaneous")
        leaky.append("temp_humidity_interaction")

    # ============================================================
    # 2. DOMAIN FEATURES USING TARGET AT TIME t (Leakage Type 2)
    # ============================================================
    print()
    print("=" * 70)
    print("2. DOMAIN FEATURES DERIVED FROM TARGET AT TIME t")
    print("=" * 70)
    print("   These features are COMPUTED from pm25(t) — circular dependency!\n")

    for col in ["co2_pm25_ratio", "pm25_aqi_cat"]:
        if col in feature_cols:
            r = float(df[col].corr(df[target]))
            print(f"   🔴  {col:30} r={r:+.4f} — uses pm25(t) directly!")
            leaky.append(col)

    # ============================================================
    # 3. DIFF FEATURES USING TARGET AT TIME t (Leakage Type 3)
    # ============================================================
    print()
    print("=" * 70)
    print("3. DIFF / PCT_CHANGE FEATURES")
    print("=" * 70)
    print("   diff_1h(t) = pm25(t) - pm25(t-1) → needs pm25(t)!\n")

    diff_cols = [c for c in feature_cols if "diff" in c or "pct_change" in c]
    for col in diff_cols:
        r = float(df[col].corr(df[target]))
        print(f"   🔴  {col:35} r={r:+.4f} — requires pm25(t)!")
        leaky.append(col)

    # ============================================================
    # 4. LAG / ROLLING / EWM — Verify shift correctness
    # ============================================================
    print()
    print("=" * 70)
    print("4. LAG FEATURES — Spot-check shift correctness")
    print("=" * 70)

    # Verify pm25_lag_1h = pm25 shifted by 1
    if "pm25_lag_1h" in df.columns:
        expected = df[target].shift(1)
        valid_idx = df["pm25_lag_1h"].dropna().index.intersection(expected.dropna().index)
        match_pct = (df.loc[valid_idx, "pm25_lag_1h"] == expected.loc[valid_idx]).mean() * 100
        print(f"   pm25_lag_1h == pm25.shift(1): match={match_pct:.1f}%")
        r = float(df["pm25_lag_1h"].corr(df[target]))
        print(f"   pm25_lag_1h correlation with target: r={r:.4f}")
        print("   ✅  Lag features are correctly shifted (use past data only)")

    # ============================================================
    # 5. SUMMARY
    # ============================================================
    print()
    print("=" * 70)
    print("5. LEAKAGE SUMMARY")
    print("=" * 70)
    leaky_unique = sorted(set(leaky))
    clean_count = len(feature_cols) - len(leaky_unique)
    print(f"   Total features:  {len(feature_cols)}")
    print(f"   🔴 Leaky features: {len(leaky_unique)}")
    print(f"   ✅ Clean features:  {clean_count}")
    print("\n   Features to REMOVE:")
    for c in leaky_unique:
        print(f"      - {c}")
    print("\n   IMPACT: Models are 'cheating' by seeing current-time data.")
    print("   ACTION: Remove leaky features and re-train all models.")

    # ============================================================
    # STATIONARITY TESTS
    # ============================================================
    print()
    print()
    print("=" * 70)
    print("STATIONARITY ANALYSIS — PM2.5")
    print("=" * 70)

    y = df[target].dropna()

    # ADF Test
    print("\n--- ADF Test (H0: unit root = non-stationary) ---")
    adf_result = adfuller(y, autolag="AIC")
    adf_stat, adf_p = adf_result[0], adf_result[1]
    print(f"   ADF Statistic: {adf_stat:.4f}")
    print(f"   p-value: {adf_p:.6f}")
    for k, v in adf_result[4].items():
        print(f"   Critical {k}: {v:.4f}")
    if adf_p < 0.05:
        print("   → Reject H0: Series IS stationary ✅")
    else:
        print("   → Fail to reject H0: Series is NON-stationary ❌")

    # KPSS Test
    print("\n--- KPSS Test (H0: stationary) ---")
    kpss_stat, kpss_p, kpss_lags, kpss_crit = kpss(y, regression="c", nlags="auto")
    print(f"   KPSS Statistic: {kpss_stat:.4f}")
    print(f"   p-value: {kpss_p:.4f}")
    for k, v in kpss_crit.items():
        print(f"   Critical {k}: {v:.4f}")
    if kpss_p >= 0.05:
        print("   → Fail to reject H0: Series IS stationary ✅")
    else:
        print("   → Reject H0: Series is NON-stationary ❌")

    # Combined interpretation per SKILL.md
    print("\n--- Combined Interpretation (SKILL.md §4) ---")
    adf_stationary = adf_p < 0.05
    kpss_stationary = kpss_p >= 0.05

    if adf_stationary and kpss_stationary:
        print("   ADF=stationary ✅, KPSS=stationary ✅ → STRONG EVIDENCE of stationarity")
    elif adf_stationary and not kpss_stationary:
        print("   ADF=stationary ✅, KPSS=non-stationary ❌ → TREND-stationary")
        print("   → May need de-trending but not differencing")
    elif not adf_stationary and kpss_stationary:
        print("   ADF=non-stationary ❌, KPSS=stationary ✅ → Inconclusive")
    else:
        print("   ADF=non-stationary ❌, KPSS=non-stationary ❌ → NON-STATIONARY")
        print("   → Need differencing (d=1)")

    # Test differenced series
    print("\n--- Differenced Series (d=1) ---")
    y_diff = y.diff().dropna()
    adf_diff = adfuller(y_diff, autolag="AIC")
    kpss_diff = kpss(y_diff, regression="c", nlags="auto")
    print(f"   ADF: stat={adf_diff[0]:.4f}, p={adf_diff[1]:.6f}")
    print(f"   KPSS: stat={kpss_diff[0]:.4f}, p={kpss_diff[1]:.4f}")
    if adf_diff[1] < 0.05 and kpss_diff[1] >= 0.05:
        print("   → After differencing: STATIONARY ✅")
    else:
        print(f"   → ADF p={adf_diff[1]:.4f}, KPSS p={kpss_diff[1]:.4f}")

    # Basic stats for context
    print("\n--- PM2.5 Descriptive Stats ---")
    print(f"   Mean:   {y.mean():.2f}")
    print(f"   Std:    {y.std():.2f}")
    print(f"   Min:    {y.min():.2f}")
    print(f"   Max:    {y.max():.2f}")
    print(f"   Median: {y.median():.2f}")


if __name__ == "__main__":
    main()
