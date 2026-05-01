# Logging Guide — loguru

> **Quyết định**: Dùng `loguru` thay vì `logging` stdlib.
> **Lý do**: Zero config, color output, structured JSON trivial, file rotation built-in. Phù hợp ML research pipeline.
> **Khi nào reconsider**: Scale lên production serving cần OTEL → migrate sang `structlog`.

---

## Setup Pattern

```python
from loguru import logger
import sys

def setup_logging(log_dir: str = "research/runs", level: str = "INFO"):
    """Cấu hình logging cho toàn project."""
    # Remove default handler
    logger.remove()

    # Console: colorful, concise
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{module}</cyan>:<cyan>{function}</cyan> | {message}",
        colorize=True,
    )

    # File: structured, rotated
    logger.add(
        f"{log_dir}/{{time:YYYYMMDD}}.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {module}:{function}:{line} | {message}",
        rotation="10 MB",
        retention="30 days",
        compression="gz",
    )

    return logger
```

---

## Log Levels cho Data Pipeline

| Level | Khi nào dùng | Ví dụ |
|-------|-------------|-------|
| **DEBUG** | Data shapes, dtypes, sample values | `logger.debug(f"Loaded: {df.shape}, dtypes: {df.dtypes.to_dict()}")` |
| **INFO** | Stage completion, timing, row counts | `logger.info(f"Cleaning done: {n_before}→{n_after} rows ({elapsed:.1f}s)")` |
| **WARNING** | Data quality issues | `logger.warning(f"High null rate: {col} has {pct:.1f}% missing")` |
| **ERROR** | Pipeline failures, validation errors | `logger.error(f"Validation failed: {errors}")` |
| **CRITICAL** | Data corruption, security violations | `logger.critical(f"Data leakage detected in {feature}")` |

---

## Structured Logging cho Experiments

```python
# JSON logging cho experiment tracking
logger.add(
    "research/runs/experiments.jsonl",
    serialize=True,  # ← Auto JSON format
    level="INFO",
    filter=lambda record: "experiment" in record["extra"],
)

# Usage
logger.bind(experiment=True).info(
    "Model training complete",
    model="XGBoost",
    mae=12.5,
    rmse=18.3,
    mase=0.85,
    duration_seconds=45.2,
)
```

---

## Context Variables

```python
# Bind context cho toàn bộ pipeline run
with logger.contextualize(run_id="20260329_120000", model="XGBoost"):
    logger.info("Starting training")       # Auto-kèm run_id + model
    logger.info("Training complete")
```

---

## Intercept stdlib logging

```python
# Bắt logs từ sklearn, xgboost, statsmodels
import logging

class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Get corresponding loguru level
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        logger.opt(depth=6, exception=record.exc_info).log(level, record.getMessage())

logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
```

---

## Timing Decorator

```python
import time
from functools import wraps

def log_time(func):
    """Decorator: log thời gian chạy của function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"{func.__name__} completed in {elapsed:.2f}s")
        return result
    return wrapper

# Usage
@log_time
def train_model(X, y):
    ...
```

---

## Templates theo Module Type

### Data Module (src/data/)
```python
from loguru import logger

def load_data(path: str) -> pd.DataFrame:
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path, parse_dates=["ngay_tao"])
    logger.debug(f"Shape: {df.shape}, Memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    logger.debug(f"Date range: {df['ngay_tao'].min()} → {df['ngay_tao'].max()}")
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df
```

### Model Module (src/models/)
```python
from loguru import logger

class XGBoostModel(BaseModel):
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        logger.info(f"Training {self.name}: X={X_train.shape}, y={y_train.shape}")
        logger.debug(f"Params: {self.params}")
        # ... training ...
        logger.info(f"Training complete: MAE={mae:.4f}, RMSE={rmse:.4f}")
```

### Pipeline Module (src/pipelines/)
```python
from loguru import logger

@log_time
def run_pipeline(config):
    logger.info("="*60)
    logger.info(f"Pipeline started | Model: {config['model_name']}")
    logger.info("="*60)

    # Stage 1
    logger.info("[1/5] Loading data...")
    # Stage 2
    logger.info("[2/5] Preprocessing...")
    # ... etc
```

---

## Anti-patterns

```python
# ❌ print() cho debugging
print(f"Shape: {df.shape}")

# ✅ logger cho debugging
logger.debug(f"Shape: {df.shape}")

# ❌ Silent failures
try:
    result = process(data)
except Exception:
    pass

# ✅ Logged failures
try:
    result = process(data)
except ValueError as e:
    logger.error(f"Processing failed: {e}", exc_info=True)
    raise
```

---

## Experiment Tracking

### Cấu trúc mỗi Run

```
research/runs/YYYYMMDD_HHMMSS/
├── config.json            # Toàn bộ tham số
├── metrics.csv            # MAE, RMSE, R², MAPE, MASE
├── predictions.csv        # y_true, y_pred
├── training_log.csv       # Loss/metric theo epoch (DL)
├── plots/
│   ├── actual_vs_predicted.png
│   ├── residuals.png
│   └── feature_importance.png
└── model.joblib           # hoặc model.pt
```

### Auto-Logging Directory Mẫu

```python
from datetime import datetime
from pathlib import Path

def create_run_directory() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"research/runs/{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)
    return run_dir
```
