"""Generate manifest.json with MD5 hashes for all model weights and key data files."""
import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def md5_file(filepath: Path) -> str:
    """Compute MD5 hash of a file."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    models_dir = PROJECT_ROOT / "models" / "exported"
    dataset_dir = PROJECT_ROOT / "dataset"

    # Model weights
    model_entries = []
    model_patterns = ["*.pt", "*.txt"]
    for pattern in model_patterns:
        for f in sorted(models_dir.glob(pattern)):
            # Determine model name and horizon from filename
            name = f.stem
            parts = name.split("_")

            if "quantile" in name:
                model_name = "GRU_Quantile"
                horizon = int(parts[-1].replace("h", ""))
                fmt = "TorchScript"
            elif name.startswith("gru"):
                model_name = "GRU"
                horizon = int(parts[-1].replace("h", ""))
                fmt = "TorchScript"
            elif name.startswith("lgbm"):
                model_name = "LightGBM"
                horizon = int(parts[-1].replace("h", ""))
                fmt = "Native (.txt)"
            else:
                model_name = name
                horizon = 0
                fmt = f.suffix

            md5 = md5_file(f)
            model_entries.append({
                "model": model_name,
                "format": fmt,
                "horizon": horizon,
                "filename": f.name,
                "expected_md5": md5,
            })
            print(f"  Model: {f.name} → {md5}")

    # Data files
    data_entries = []
    data_files = [
        dataset_dir / "raw" / "final_dataset.csv",
    ]

    # Add processed files if they exist
    processed_candidates = [
        dataset_dir / "processed" / "cleaned_hourly.csv",
        dataset_dir / "processed" / "hybrid_imputed.csv",
        dataset_dir / "processed" / "marts_features.csv",
    ]
    for f in processed_candidates:
        if f.exists():
            data_files.append(f)

    # Config/feature files
    config_candidates = list(models_dir.glob("*_features.json"))
    config_candidates += list(models_dir.glob("*_config.json"))
    config_candidates += list(models_dir.glob("scalers_*.json"))
    data_files.extend(sorted(config_candidates))

    for f in sorted(data_files):
        if f.exists():
            md5 = md5_file(f)
            rel_path = str(f.relative_to(PROJECT_ROOT))
            data_entries.append({
                "path": rel_path,
                "expected_md5": md5,
            })
            print(f"  Data:  {rel_path} → {md5}")

    # Build manifest
    manifest = {
        "version": "v8_scientific_audit",
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "models": model_entries,
        "data_files": data_entries,
    }

    output_path = models_dir / "manifest.json"
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✅ Manifest saved: {output_path}")
    print(f"   Models: {len(model_entries)}")
    print(f"   Data files: {len(data_entries)}")


if __name__ == "__main__":
    main()
