from src.snapshot_adapter import load_all_normalized

data = load_all_normalized().get('v9_multi_resolution', {})

for h in ['1h', '6h', '24h']:
    print(f"Testing {h}")
    h_results_for_rank = data.get("results", {}).get(h, {})
    all_models_for_h = []
    for model_name, metrics in h_results_for_rank.items():
        if model_name.startswith("Persistence"):
            continue
        all_models_for_h.append({
            "model": model_name,
            "mae": metrics.get("mae", float("inf")),
            "mase": metrics.get("mase", float("inf")),
            "rmse": metrics.get("rmse"),
            "r2": metrics.get("r2"),
            "da": metrics.get("da"),
        })
        
    top_models = sorted(all_models_for_h, key=lambda x: x.get("mase", float("inf")))[:5]
    for row in top_models:
        try:
            rmse_display = f"{row['rmse']:.2f}" if row.get('rmse') else "—"
            r2_display = f"{row['r2']:.3f}" if row.get('r2') is not None else "—"
            da_display = f"{row['da']:.1f}%" if row.get('da') is not None else "—"
        except Exception as e:
            print(f"Error on model {row['model']}: {type(e).__name__}: {e}")
            print(f"row: {row}")
