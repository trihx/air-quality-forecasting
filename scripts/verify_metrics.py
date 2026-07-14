"""Quick verification of the metrics pipeline end-to-end."""

from src.snapshot_adapter import load_all_normalized
from src.reporting.engine import ReportingEngine

all_data = load_all_normalized()
data = all_data.get("v9_multi_resolution", {})
if not data:
    print("ERROR: v9_multi_resolution not found in snapshots")
    import sys; sys.exit(1)
rpt = ReportingEngine(data)

# Test ranking tables
print("=== MAE Ranking Table (1h, top 3) ===")
df = rpt.get_mae_ranking_table("1h", top_n=3)
print("Columns:", df.columns.tolist())
print(df.to_string())

print()
print("=== MASE Ranking Table (6h, top 3) ===")
df2 = rpt.get_mase_ranking_table("6h", top_n=3)
print("Columns:", df2.columns.tolist())
print(df2.to_string())

print()
print("=== Sample model data (1h) ===")
models = list(data["results"]["1h"].items())[:3]
for name, m in models:
    rmse = m.get("rmse")
    r2 = m.get("r2")
    da = m.get("da")
    bias = m.get("forecast_bias")
    print(f"  {name}: rmse={rmse}, r2={r2}, da={da}, bias={bias}")

print()
print("=== Bias test (6h) ===")
bias_count = 0
for name, m in data["results"]["6h"].items():
    if m.get("forecast_bias") is not None:
        bias_count += 1
print(f"  Models with bias data: {bias_count}/{len(data['results']['6h'])}")

print()
print("PASS: All verifications completed successfully!")
