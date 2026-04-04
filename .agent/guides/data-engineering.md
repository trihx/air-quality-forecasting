# Data Engineering — Pipeline, Quality, Lineage

> **Nguồn**: Skill `data-engineer` + `dbt-transformation-patterns` — adapted cho dự án PM2.5
>
> Hướng dẫn tổ chức data pipeline theo layers, data quality checks, data contracts, và lineage tracking.

---

## 1. Data Transformation Layers (dbt-inspired)

Tổ chức data transformations theo **3 lớp**, lấy cảm hứng từ dbt:

```
dataset/raw/          → Staging Layer    (src/data/loader.py)
  │ Load & Validate — 1:1 với source, KHÔNG transform logic
  ↓
dataset/interim/      → Intermediate Layer (src/data/cleaner.py + preprocessor.py)
  │ Clean, Handle missing, Outliers, Resample
  ↓
dataset/processed/    → Marts Layer      (src/features/builder.py)
  │ Feature engineering, Train-ready
  ↓
  Sẵn sàng cho training
```

### Layer Definitions

| Layer | Thư mục | Module | Quy tắc |
|-------|---------|--------|---------|
| **Staging** | `dataset/raw/` | `src/data/loader.py` | 1:1 mapping với source. Chỉ validate format, KHÔNG áp dụng business logic |
| **Intermediate** | `dataset/interim/` | `src/data/cleaner.py`, `preprocessor.py` | Business logic: interpolation, outlier handling, resampling |
| **Marts** | `dataset/processed/` | `src/features/builder.py` | Feature-rich, specific cho model type. Train-ready format |

### Naming Convention

```
dataset/raw/final_dataset.csv                  # Raw — immutable
dataset/interim/cleaned_YYYYMMDD.parquet       # Sau cleaning
dataset/interim/resampled_1h_YYYYMMDD.parquet  # Sau resampling
dataset/processed/features_v1_YYYYMMDD.parquet # Features train-ready
```

### Quy tắc Layer

1. **Raw data KHÔNG BAO GIỜ bị thay đổi** (immutable)
2. Mỗi layer chỉ đọc từ layer trước nó (staging → intermediate → marts)
3. **KHÔNG** skip layers (raw → marts trực tiếp)
4. Mỗi transformation **PHẢI** log input/output shape & stats
5. Checkpoint data giữa các layers (parquet format, có timestamp)

---

## 2. Data Quality Checks

### Approach: Great Expectations → Custom Validation

**Giai đoạn 1**: Dùng Great Expectations cho structured validations
**Giai đoạn 2**: So sánh với custom validation, chọn approach tối ưu

### Great Expectations Setup

```python
# src/data/quality.py
import great_expectations as gx

def create_data_context():
    """Khởi tạo Great Expectations context cho project."""
    context = gx.get_context()
    return context

def validate_raw_data(df, context):
    """Validate raw data tại Staging layer."""
    datasource = context.sources.add_or_update_pandas("raw_data")
    data_asset = datasource.add_dataframe_asset(name="raw_pm25")
    
    batch_request = data_asset.build_batch_request(dataframe=df)
    
    # Expectations cho raw data
    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name="raw_data_suite"
    )
    
    # Column existence
    validator.expect_table_columns_to_match_set(
        column_set=["ngay_tao", "nhiet_do", "do_am", "diem_suong", "co2", "pm25"]
    )
    
    # Not null checks
    validator.expect_column_values_to_not_be_null("ngay_tao")
    validator.expect_column_values_to_not_be_null("pm25", mostly=0.95)
    
    # Range checks
    validator.expect_column_values_to_be_between("pm25", min_value=0, max_value=500)
    validator.expect_column_values_to_be_between("nhiet_do", min_value=-20, max_value=60)
    validator.expect_column_values_to_be_between("do_am", min_value=0, max_value=100)
    
    # Type checks
    validator.expect_column_values_to_be_dateutil_parseable("ngay_tao")
    
    results = validator.validate()
    return results

def validate_processed_data(df, context):
    """Validate processed data tại Marts layer."""
    datasource = context.sources.add_or_update_pandas("processed_data")
    data_asset = datasource.add_dataframe_asset(name="processed_pm25")
    
    batch_request = data_asset.build_batch_request(dataframe=df)
    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name="processed_data_suite"
    )
    
    # Sau cleaning, KHÔNG có NaN
    for col in df.columns:
        validator.expect_column_values_to_not_be_null(col)
    
    # Frequency check (resampled to 1h)
    validator.expect_column_values_to_be_unique("ngay_tao")
    
    results = validator.validate()
    return results
```

### Quality Check Matrix

| Layer | Check | Severity | Action on Fail |
|-------|-------|----------|---------------|
| **Staging** | Columns exist | 🔴 Critical | STOP pipeline |
| **Staging** | Schema match (dtypes) | 🔴 Critical | STOP pipeline |
| **Staging** | PM2.5 range [0, 500] | 🟡 Warning | Log, clip values |
| **Staging** | Null rate < 5% | 🟡 Warning | Log, proceed |
| **Staging** | Null rate > 20% | 🔴 Critical | STOP, investigate source |
| **Intermediate** | No NaN after interpolation | 🔴 Critical | Check interpolation logic |
| **Intermediate** | Frequency = 1h | 🔴 Critical | Check resampling |
| **Intermediate** | Row count > 1000 | 🟡 Warning | Possible data loss |
| **Marts** | No NaN in features | 🔴 Critical | Check feature engineering |
| **Marts** | No Inf values | 🔴 Critical | Check rolling/ratio features |
| **Marts** | No future data leakage | 🔴 Critical | Verify lag features only use past |

### Custom Validation (Lightweight Alternative)

```python
# src/data/validators.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class ValidationResult:
    passed: bool
    check_name: str
    details: str
    severity: str  # "critical" | "warning" | "info"

def validate_no_future_leakage(
    df, 
    target_col: str, 
    feature_cols: list,
    datetime_col: str
) -> ValidationResult:
    """Kiểm tra: features KHÔNG sử dụng dữ liệu tương lai."""
    # Với mỗi lag feature, verify giá trị tại t 
    # KHÔNG lấy từ t+1, t+2, ...
    for col in feature_cols:
        if "lag" in col:
            lag_value = int(col.split("_")[-1])
            # Spot check: feature tại index i = target tại index i-lag
            sample_idx = min(lag_value + 10, len(df) - 1)
            expected = df[target_col].iloc[sample_idx - lag_value]
            actual = df[col].iloc[sample_idx]
            if abs(expected - actual) > 1e-6:
                return ValidationResult(
                    passed=False,
                    check_name="no_future_leakage",
                    details=f"Lag feature {col} tại index {sample_idx}: "
                            f"expected {expected}, got {actual}",
                    severity="critical"
                )
    
    return ValidationResult(
        passed=True,
        check_name="no_future_leakage",
        details="All lag features verified — no future data leakage",
        severity="info"
    )
```

### So sánh: Great Expectations vs Custom

| Tiêu chí | Great Expectations | Custom Validation |
|----------|-------------------|-------------------|
| **Setup** | Heavier (config, stores) | Minimal (pure Python) |
| **Built-in checks** | 300+ expectations | Tự viết |
| **Data docs** | Auto-generate HTML reports | Manual |
| **CI/CD integration** | Built-in checkpoints | Custom script |
| **Domain-specific** | Generic → cần customize | Hoàn toàn flexible |
| **Tốt cho** | Standard data validation | Time series-specific checks (leakage, drift) |

**Đề xuất**: Dùng Great Expectations cho standard checks (null, range, type), custom cho domain-specific checks (leakage, temporal consistency).

---

## 3. Data Contract & SLA

### Data Assets & Owners

| Data Asset | Owner | Format | Freshness SLA | Quality SLA |
|-----------|-------|--------|--------------|------------|
| `dataset/raw/final_dataset.csv` | Data Source (IoT Sensors) | CSV | Daily update | >95% completeness |
| `dataset/interim/cleaned_*.parquet` | Data Pipeline (`src/data/`) | Parquet | After each pipeline run | No NaN, validated ranges |
| `dataset/processed/features_*.parquet` | Feature Pipeline (`src/features/`) | Parquet | After each pipeline run | 100% no-null, no-inf, no leakage |
| `models/*.joblib` | Training Pipeline | Joblib | After each experiment | MASE < 1.0 on test set |
| `research/runs/*/metrics.csv` | Experiment Logger | CSV | After each run | All metrics computed |

### Contract Violations

Khi SLA bị vi phạm:

```
1. Log violation vào LESSONS_LEARNED.md
2. Alert owner (ghi rõ trong contract)
3. Pipeline DỪNG tại violation point
4. KHÔNG propagate bad data downstream
```

---

## 4. Data Lineage (Truy vết Nguồn gốc)

### Full Lineage Graph

```
                        ┌──────────────────────┐
                        │   IoT Sensor (PM2.5) │
                        └──────────┬───────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  dataset/raw/final_dataset  │
                    │  (Immutable, ~210K records)  │
                    └──────────────┬──────────────┘
                                   │
               ┌───────────────────▼───────────────────┐
               │   STAGING: src/data/loader.py          │
               │   Load CSV + Schema Validation         │
               │   + Great Expectations raw_data_suite  │
               └───────────────────┬───────────────────┘
                                   │
          ┌────────────────────────▼────────────────────────┐
          │   INTERMEDIATE: src/data/cleaner.py              │
          │   Sort → Dedup → Missing Values → Outliers       │
          │   + src/data/preprocessor.py: Resample → Segment │
          │   → dataset/interim/cleaned_*.parquet             │
          └────────────────────────┬────────────────────────┘
                                   │
     ┌─────────────────────────────▼─────────────────────────────┐
     │   MARTS: src/features/builder.py                           │
     │   Lag + Rolling + Calendar + Domain Features               │
     │   + No-leakage validation                                  │
     │   → dataset/processed/features_*.parquet                   │
     └─────────────────────────────┬─────────────────────────────┘
                                   │
          ┌────────────────────────▼────────────────────────┐
          │   SPLIT: src/data/splitter.py                    │
          │   Train (80%) / Val (10%) / Test (10%)           │
          │   Temporal order, NO shuffle                     │
          └────────────────────────┬────────────────────────┘
                                   │
               ┌───────────────────▼───────────────────┐
               │   TRAINING: src/pipelines/train.py     │
               │   Scaler fit ONLY on train              │
               │   → models/*.joblib                     │
               │   → research/runs/*/                    │
               └───────────────────┬───────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   PREDICTION & DASHBOARD     │
                    │   src/pipelines/predict.py    │
                    │   app.py (Streamlit)          │
                    └─────────────────────────────┘
```

### Lineage Logging

```python
# src/utils/lineage.py
import json
from datetime import datetime
from pathlib import Path

def log_transformation(
    step_name: str,
    input_path: str,
    output_path: str,
    input_shape: tuple,
    output_shape: tuple,
    transformations: list[str],
    params: dict = None,
):
    """Log data transformation step cho lineage tracking."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "step": step_name,
        "input": {
            "path": input_path,
            "shape": list(input_shape),
        },
        "output": {
            "path": output_path,
            "shape": list(output_shape),
        },
        "transformations": transformations,
        "params": params or {},
        "row_diff": output_shape[0] - input_shape[0],
        "col_diff": output_shape[1] - input_shape[1],
    }
    
    # Append to lineage log
    lineage_file = Path("research/lineage_log.jsonl")
    lineage_file.parent.mkdir(parents=True, exist_ok=True)
    with open(lineage_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    
    return log_entry
```

---

## 5. Data Versioning (Lightweight)

> Thay vì DVC full, dùng checksums + metadata cho dataset versioning.

```python
# src/utils/data_version.py
import hashlib
from pathlib import Path

def compute_checksum(file_path: str) -> str:
    """Compute SHA-256 checksum cho data file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def create_data_manifest(data_dir: str = "dataset") -> dict:
    """Tạo manifest với checksums cho tất cả data files."""
    manifest = {}
    for path in Path(data_dir).rglob("*"):
        if path.is_file() and not path.name.startswith("."):
            manifest[str(path)] = {
                "checksum": compute_checksum(str(path)),
                "size_bytes": path.stat().st_size,
                "modified": path.stat().st_mtime,
            }
    return manifest
```

**Workflow:**
1. Sau mỗi pipeline run → generate manifest
2. Lưu manifest vào `research/runs/YYYYMMDD_HHMMSS/data_manifest.json`
3. So sánh manifests giữa các runs → detect data changes
