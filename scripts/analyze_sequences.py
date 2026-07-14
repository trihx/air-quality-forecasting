"""Quick analysis: Why 30m train sequences ≈ 1h train sequences."""
import pandas as pd

print("=" * 70)
print("ANALYSIS: Why Train Sequences at 30m ≈ 1h?")
print("=" * 70)

df_30m = pd.read_csv("dataset/processed/marts_features_30m.csv", index_col=0, parse_dates=True)
seg_sizes = df_30m.groupby("segment_id").size()

lookback = 144  # 72h * 2 steps/h
horizon_1h = 2
min_window = lookback + horizon_1h  # 146

print(f"\n--- 30m Fair Dataset ---")
print(f"Total rows: {len(df_30m):,}")
print(f"Segments: {df_30m.segment_id.nunique()}")
print(f"Lookback: {lookback} steps | Horizon 1h: {horizon_1h} steps")
print(f"Min window needed: {min_window} steps")
print(f"\nSegment length distribution:")
print(f"  min={seg_sizes.min()}, median={seg_sizes.median():.0f}, max={seg_sizes.max()}")

usable = (seg_sizes >= min_window).sum()
unusable = (seg_sizes < min_window).sum()
print(f"\nUsable segments (>= {min_window} steps): {usable}")
print(f"TOO SHORT segments (< {min_window} steps): {unusable}")
print(f"\n*** ROOT CAUSE: {unusable}/{len(seg_sizes)} segments are SHORTER than lookback! ***")
print(f"*** These segments contribute ZERO training sequences ***")

total_seqs = 0
print(f"\nPer-segment sequence count:")
for sid, slen in seg_sizes.sort_values(ascending=False).items():
    n = max(0, slen - min_window + 1)
    total_seqs += n
    if n > 0:
        print(f"  Seg {sid}: {slen} steps -> {n} sequences")
    
print(f"\nTOTAL valid sequences: {total_seqs}")
print(f"  80% train: {int(total_seqs * 0.8):,}")
print(f"  10% test:  {int(total_seqs * 0.1):,}")

# Compare with BASE dataset
print(f"\n--- 30m BASE (Expert) Dataset ---")
df_base = pd.read_csv("dataset/processed/marts_features_30m_base.csv", index_col=0, parse_dates=True)
seg_base = df_base.groupby("segment_id").size()
print(f"Total rows: {len(df_base):,}")
print(f"Segment lengths: min={seg_base.min()}, median={seg_base.median():.0f}, max={seg_base.max()}")

usable_b = (seg_base >= min_window).sum()
total_base = sum(max(0, slen - min_window + 1) for slen in seg_base)
print(f"Usable segments: {usable_b}")
print(f"TOTAL valid sequences: {total_base}")
print(f"  80% train: {int(total_base * 0.8):,}")

print(f"\n--- CONCLUSION ---")
print(f"30m Fair:   {total_seqs:,} sequences (median seg {seg_sizes.median():.0f} < lookback {lookback})")
print(f"30m Expert: {total_base:,} sequences (median seg {seg_base.median():.0f} vs lookback {lookback})")
gain_pct = (total_base / max(total_seqs, 1) - 1) * 100
print(f"Expert gain: +{total_base - total_seqs:,} sequences (+{gain_pct:.0f}%)")
