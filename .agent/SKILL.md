---
name: Time Series Forecasting & Prediction
description: Quy trình toàn diện cho dự án dự báo chuỗi thời gian PM2.5 - bao gồm data pipeline, ML/DL, ensemble, Optuna, context engineering, unit testing, và self-correcting rules.
---

# 🧠 Time Series Forecasting & Prediction Skill

> **Mục đích**: Tài liệu này là kim chỉ nam cho toàn bộ dự án dự báo chuỗi thời gian.
> AI agent **PHẢI** đọc file này và các file trong `.agent/memory/` **TRƯỚC** khi thực hiện bất kỳ thay đổi nào.

---

## 1. QUY TẮC VÀNG (GOLDEN RULES)

> [!CAUTION]
> Các quy tắc này là **BẮT BUỘC**. Vi phạm bất kỳ quy tắc nào sẽ dẫn đến kết quả không đáng tin cậy.

### 1.1 Gate-keeping & Approval Process

**TRƯỚC KHI thực hiện bất kỳ thay đổi nào**, agent PHẢI:

```
┌──────────────────────────────────────────────────────┐
│ 1. Đọc .agent/SKILL.md (file này)                   │
│ 2. Đọc .agent/memory/CONTEXT.md                     │
│ 3. Đọc .agent/memory/LESSONS_LEARNED.md              │
│ 4. Đọc .agent/memory/RUNS_LOG.md                     │
│ 5. Đọc .agent/memory/DECISIONS.md                    │
│ 6. Kiểm tra .agent/memory/TODO.md                    │
│ 7. Xác nhận đầy đủ context                          │
│ 8. Trình bày kế hoạch cho user duyệt                │
│ 9. Chỉ implement SAU KHI user approve               │
└──────────────────────────────────────────────────────┘
```

**SAU KHI implement**, agent PHẢI:

```
┌──────────────────────────────────────────────────────┐
│ 1. Chạy lint/validate (xem Section 1.10)             │
│ 2. Chạy unit tests                                   │
│ 3. Cập nhật .agent/memory/CONTEXT.md                 │
│ 4. Ghi kết quả vào .agent/memory/RUNS_LOG.md         │
│ 5. Nếu có lỗi → .agent/memory/LESSONS_LEARNED.md     │
│ 6. Nếu có quyết định → .agent/memory/DECISIONS.md    │
│ 7. Cập nhật .agent/memory/TODO.md                    │
└──────────────────────────────────────────────────────┘
```

### 1.2 Nguyên tắc Trung thực Dữ liệu
- **KHÔNG BAO GIỜ** tưởng tượng hoặc bịa đặt kết quả. Mọi số liệu phải đến từ code chạy thực tế.
- **LUÔN LUÔN** chạy code và báo cáo kết quả **ĐÚNG** như output thực.
- Nếu kết quả không như mong đợi → ghi nhận thực tế, phân tích nguyên nhân, KHÔNG chỉnh sửa số liệu.

### 1.3 Nguyên tắc Tham chiếu (Reference-First)
- Trước khi implement → **ĐỌC LẠI** SKILL.md này và các tài liệu liên quan.
- Tham chiếu `docs/` cho kiến thức domain và academic foundations.
- Kiểm tra `.agent/memory/LESSONS_LEARNED.md` để tránh lặp lại lỗi cũ.

### 1.4 Nguyên tắc Khai báo Tham số
- **TẤT CẢ** tham số phải khai báo ở **ĐẦU FILE** trong block `CONFIG`.
- Mỗi tham số **PHẢI** có chú thích giải thích ý nghĩa, đơn vị, và phạm vi hợp lệ.
- **KHÔNG** hard-code tham số bên trong thân hàm.

```python
# ============================================================
# CONFIGURATION - Điều chỉnh tham số tại đây
# ============================================================

# --- Data Parameters ---
DATA_PATH = "dataset/final_dataset.csv"          # Đường dẫn file dữ liệu gốc
TARGET_COL = "pm25"                               # Cột mục tiêu cần dự báo
DATETIME_COL = "ngay_tao"                         # Cột thời gian
FEATURE_COLS = ["nhiet_do", "do_am", "diem_suong", "co2"]  # Các cột đặc trưng

# --- Preprocessing Parameters ---
RESAMPLE_FREQ = "1h"                # Tần suất resampling (1h = 1 giờ)
INTERPOLATION_METHOD = "linear"     # Phương pháp nội suy: linear, cubic, time
MAX_GAP_INTERPOLATE = "2h"         # Khoảng trống tối đa cho phép nội suy
OUTLIER_METHOD = "iqr"              # Phương pháp phát hiện outlier: iqr, zscore
OUTLIER_THRESHOLD = 1.5             # Hệ số IQR (1.5 = mild, 3.0 = extreme)

# --- Train/Test Split ---
TRAIN_RATIO = 0.8                   # Tỷ lệ dữ liệu huấn luyện (80%)
VALIDATION_RATIO = 0.1              # Tỷ lệ dữ liệu validation (10%)
TEST_RATIO = 0.1                    # Tỷ lệ dữ liệu kiểm tra (10%)
SHUFFLE = False                     # KHÔNG shuffle cho time series!

# --- Model Parameters ---
RANDOM_STATE = 42                   # Seed cho reproducibility
N_JOBS = -1                         # Số CPU cores (-1 = tất cả)
CV_SPLITS = 5                       # Số fold cho TimeSeriesSplit

# --- Forecasting Horizons ---
FORECAST_HORIZONS = [1, 6, 24]      # Dự báo trước 1h, 6h, 24h
LOOKBACK_WINDOW = 168               # Cửa sổ nhìn lại (168h = 7 ngày)

# --- Deep Learning Parameters ---
BATCH_SIZE = 32                     # Kích thước batch
EPOCHS = 100                        # Số epoch tối đa
LEARNING_RATE = 0.001               # Tốc độ học khởi tạo
EARLY_STOPPING_PATIENCE = 10        # Dừng sớm sau N epoch không cải thiện
HIDDEN_DIM = 64                     # Số neuron ẩn (LSTM/GRU)
NUM_LAYERS = 2                      # Số lớp ẩn
DROPOUT = 0.2                       # Tỷ lệ dropout (chống overfitting)
```

### 1.5 Nguyên tắc Lưu trữ (Persistence)
- **MỌI** lần chạy (run), tối ưu, fine-tune **PHẢI** được lưu lại.
- Lưu vào `research/runs/YYYYMMDD_HHMMSS/` với đầy đủ: config, metrics, model artifacts.
- Cập nhật `.agent/memory/RUNS_LOG.md` ngay sau mỗi lần chạy.

### 1.6 Nguyên tắc Tự sửa lỗi (Self-Correcting)
- Khi gặp lỗi → **TÌM NGUYÊN NHÂN GỐC RỄ** (root cause), không chỉ fix triệu chứng.
- Ghi lại vào `.agent/memory/LESSONS_LEARNED.md`.
- Cập nhật SKILL.md nếu cần bổ sung quy tắc mới để phòng tránh.

### 1.7 Nguyên tắc Bảo mật (Security Rules)

> [!WARNING]
> Code trong dự án **PHẢI** tuân thủ các quy tắc bảo mật dưới đây.

#### a) Input Path Validation
- **TẤT CẢ** đường dẫn file phải được validate trước khi sử dụng.
- Dùng `pathlib.Path.resolve()` để chống path traversal attacks.
- Kiểm tra file tồn tại trước khi đọc.

```python
from pathlib import Path

def validate_path(path: str, must_exist: bool = True) -> Path:
    """Validate và chuẩn hóa đường dẫn file."""
    p = Path(path).resolve()
    project_root = Path(__file__).parent.parent.resolve()
    if not str(p).startswith(str(project_root)):
        raise ValueError(f"Path '{path}' nằm ngoài thư mục dự án")
    if must_exist and not p.exists():
        raise FileNotFoundError(f"File không tồn tại: {p}")
    return p
```

#### b) Safe Model Deserialization
- **KHÔNG** dùng `joblib.load()` hoặc `pickle.load()` trực tiếp trên file không tin cậy.
- Chỉ load model từ thư mục `models/` trong dự án.
- Dùng `skops.io` hoặc kiểm tra nguồn gốc file trước khi load.

```python
import joblib

def safe_load_model(model_path: str):
    """Load model an toàn — chỉ từ thư mục models/ của dự án."""
    path = validate_path(model_path, must_exist=True)
    models_dir = Path(__file__).parent.parent.resolve() / "models"
    if not str(path).startswith(str(models_dir)):
        raise ValueError(f"Chỉ được load model từ {models_dir}")
    return joblib.load(path)
```

#### c) Exception Handling
- **KHÔNG** dùng bare `except:` — luôn chỉ rõ exception cụ thể.
- Log toàn bộ traceback để debug.

```python
# ❌ SAI
try:
    model.fit(X, y)
except:
    pass

# ✅ ĐÚNG
try:
    model.fit(X, y)
except (ValueError, RuntimeError) as e:
    logger.error(f"Model training failed: {e}", exc_info=True)
    raise
```

#### d) Warnings Suppression
- **KHÔNG** dùng `warnings.filterwarnings('ignore')` global.
- Chỉ suppress warnings cụ thể trong context manager.

```python
# ❌ SAI
import warnings
warnings.filterwarnings('ignore')

# ✅ ĐÚNG
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
    model.fit(X, y)
```

#### e) Secrets & Environment
- **KHÔNG** commit `.env`, API keys, credentials vào git.
- Kiểm tra `.gitignore` bao gồm: `.env`, `.env.*`, `*.key`
- Dùng `python-dotenv` hoặc biến môi trường cho secrets.

### 1.8 Nguyên tắc Lập kế hoạch (Concise Planning)

> Mọi kế hoạch implement **PHẢI** tuân theo Plan Template chuẩn: Approach → Scope → Action Items (6-10 bước, verb-first) → Open Questions (≤3).
> Mỗi action item phải **atomic**, **concrete** (ghi rõ file/module), và có ít nhất 1 bước **validation**.

📖 **Chi tiết**: [`.agent/guides/concise-planning.md`](file:///Users/trihx/Desktop/time-series-forecasting/.agent/guides/concise-planning.md)

### 1.9 Nguyên tắc Kaizen (Cải tiến Liên tục)

> **Triết lý**: Nhiều cải tiến nhỏ > 1 thay đổi lớn. Phòng lỗi từ thiết kế.

**4 Trụ cột**:

| Trụ cột | Áp dụng |
|---------|--------|
| **Cải tiến liên tục** | 3 vòng: Make it work → Make it clear → Make it efficient |
| **Poka-Yoke** | Validate tại boundary, 4 lớp phòng thủ (type → validation → guards → error boundaries) |
| **Chuẩn hóa** | Follow BaseModel interface, naming convention, error handling pattern |
| **Just-In-Time** | YAGNI + Rule of Three (abstract chỉ khi ≥3 cases giống nhau) |

📖 **Chi tiết**: [`.agent/guides/kaizen.md`](file:///Users/trihx/Desktop/time-series-forecasting/.agent/guides/kaizen.md)

### 1.10 Nguyên tắc Code Quality (Lint & Validate)

> [!CAUTION]
> **BẮT BUỘC** chạy lint/validate SAU MỖI thay đổi code. Code không qua lint = chưa hoàn thành.

**Quality Loop**: Ruff Check → Ruff Format → Bandit → MyPy → Pytest → ✅ Commit

| Công cụ | Lệnh nhanh |
|---------|--------|
| Ruff (lint) | `uv run ruff check src/ --fix` |
| Ruff (format) | `uv run ruff format src/` |
| Bandit (security) | `uv run bandit -r src/ -ll` |
| MyPy (types) | `uv run mypy src/` |
| Pytest (tests) | `uv run pytest tests/ -v --cov=src` |

📖 **Chi tiết + config**: [`.agent/guides/lint-and-validate.md`](file:///Users/trihx/Desktop/time-series-forecasting/.agent/guides/lint-and-validate.md)
🔄 **Workflow**: Chạy `/lint` để thực hiện toàn bộ quality loop.

---

## 2. KIẾN TRÚC DỰ ÁN (PROJECT ARCHITECTURE)

### 2.1 Cấu trúc Thư mục Chuẩn

```
time-series-forecasting/
│
├── .agent/                           # 🤖 Antigravity Skill System
│   ├── SKILL.md                      #   File này - Kim chỉ nam dự án
│   ├── memory/                       #   Bộ nhớ dự án (context engineering)
│   │   ├── CONTEXT.md
│   │   ├── RUNS_LOG.md
│   │   ├── LESSONS_LEARNED.md
│   │   ├── DECISIONS.md
│   │   └── TODO.md
│   ├── guides/                       #   Hướng dẫn chi tiết (tách file)
│   │   ├── concise-planning.md       #     Template lập kế hoạch
│   │   ├── kaizen.md                 #     Cải tiến liên tục 4 trụ cột
│   │   ├── lint-and-validate.md      #     Code quality tools & config
│   │   ├── systematic-debugging.md   #     Debug 4 pha có hệ thống
│   │   ├── analytics-experiment-design.md # Experiment design cho ML
│   │   ├── data-engineering.md       #     Data pipeline, quality, lineage
│   │   └── embedding-strategies.md   #     TS embeddings (placeholder)
│   └── workflows/                    #   Workflows tự động
│       ├── run-experiment.md
│       ├── refactor.md
│       └── lint.md                   #   Quality check workflow
│
├── configs/                          # ⚙️ Cấu hình tập trung
│   ├── base_config.yaml              #   Cấu hình mặc định
│   └── model_configs/                #   Cấu hình từng model
│       ├── random_forest.yaml
│       ├── xgboost.yaml
│       ├── lstm.yaml
│       └── transformer.yaml
│
├── dataset/                          # 📊 Dữ liệu
│   ├── raw/                          #   Dữ liệu gốc, bất biến (immutable)
│   │   └── final_dataset.csv
│   ├── interim/                      #   Dữ liệu trung gian (sau cleaning)
│   └── processed/                    #   Dữ liệu sạch, sẵn sàng train
│
├── docs/                             # 📚 Tài liệu tham khảo (31 sách)
│
├── src/                              # 🔧 Source code chính
│   ├── __init__.py
│   ├── data/                         #   Load, clean, preprocess, split
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── cleaner.py
│   │   ├── preprocessor.py
│   │   └── splitter.py
│   ├── features/                     #   Feature engineering
│   │   ├── __init__.py
│   │   ├── temporal.py               #     Lag, rolling, ewm
│   │   ├── calendar.py               #     Hour, day, month, holiday
│   │   └── builder.py                #     Pipeline tổng hợp
│   ├── models/                       #   Định nghĩa model
│   │   ├── __init__.py
│   │   ├── base_model.py             #     Abstract base class
│   │   ├── statistical/              #     ARIMA, SARIMA, Prophet
│   │   ├── ml/                       #     RF, XGBoost, LightGBM
│   │   ├── dl/                       #     LSTM, GRU, Transformer
│   │   └── ensemble/                 #     Voting, Stacking, Blending
│   ├── evaluation/                   #   Metrics & reporting
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── visualizer.py
│   │   └── reporter.py
│   ├── pipelines/                    #   Pipeline end-to-end
│   │   ├── __init__.py
│   │   ├── train_pipeline.py
│   │   └── predict_pipeline.py
│   └── utils/                        #   Tiện ích
│       ├── __init__.py
│       ├── logger.py
│       ├── config_loader.py
│       └── reproducibility.py
│
├── notebooks/                        # 📓 Jupyter (EDA, prototyping)
│
├── models/                           # 💾 Trained models (serialized)
│   └── model_registry.json
│
├── research/                         # 🔬 Kết quả nghiên cứu
│   ├── runs/                         #   Mỗi lần chạy = 1 thư mục
│   │   └── YYYYMMDD_HHMMSS/
│   │       ├── config.json
│   │       ├── metrics.csv
│   │       ├── predictions.csv
│   │       └── plots/
│   └── experiments/                  #   Optuna studies
│       └── optuna.db
│
├── tests/                            # 🧪 Unit & Integration Tests
│   ├── conftest.py
│   ├── unit/
│   ├── integration/
│   └── validation/
│
├── app.py                            # 🌐 Streamlit Dashboard
├── main.py
├── pyproject.toml                    # Dependencies (managed by uv)
├── uv.lock                           # Lock file (auto-generated)
└── README.md
```

### 2.2 Nguyên tắc Kiến trúc

| Nguyên tắc | Mô tả |
|------------|--------|
| **Single Responsibility** | Mỗi module/file chỉ làm MỘT việc |
| **Immutable Raw Data** | Dữ liệu gốc trong `dataset/raw/` KHÔNG bao giờ bị thay đổi |
| **Config-Driven** | Mọi tham số đều từ config, không hard-code |
| **Reproducible** | Set seed, log config → bất kỳ ai cũng chạy lại được |
| **Separation of Concerns** | Notebooks = thử nghiệm, `src/` = production code |
| **DRY** | Code dùng chung → `src/utils/` |

### 2.3 Base Model Interface

Mọi model **PHẢI** kế thừa từ `BaseModel`:

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

class BaseModel(ABC):
    """Abstract base class cho tất cả models trong dự án."""

    def __init__(self, name: str, params: Dict[str, Any]):
        self.name = name
        self.params = params
        self.model = None
        self.is_fitted = False

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None) -> Dict[str, float]:
        """Huấn luyện model. Trả về dict metrics trên validation set."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Dự báo. Trả về array giá trị dự báo."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Lưu model ra file."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Nạp model từ file."""
        pass

    def get_params(self) -> Dict[str, Any]:
        return self.params.copy()
```

---

## 3. QUY TRÌNH DỮ LIỆU (DATA PIPELINE)

### 3.1 Thu thập & Validate

```python
def load_and_validate(path: str) -> pd.DataFrame:
    """
    Load CSV và validate cơ bản.
    1. Kiểm tra file tồn tại
    2. Load CSV với parse_dates
    3. Validate: columns tồn tại, dtypes đúng
    4. Validate: không có giá trị âm cho PM2.5
    5. Validate: datetime range hợp lý
    6. Log thống kê cơ bản
    """
```

### 3.2 Data Cleaning

Thứ tự xử lý **BẮT BUỘC** (thay đổi thứ tự ảnh hưởng kết quả):

| Bước | Thao tác | Chi tiết |
|------|----------|----------|
| 1 | **Sort** theo thời gian | `df.sort_values(DATETIME_COL)` |
| 2 | **Remove duplicates** | Giữ bản ghi đầu tiên |
| 3 | **Handle missing values** | Chiến lược nội suy 3 cấp |
| 4 | **Detect outliers** | IQR hoặc Z-score, **clip** không xóa |
| 5 | **Resample** | Đưa về tần suất cố định |
| 6 | **Segmentation** | Chia segment dựa trên gaps |

### 3.3 Chiến lược Nội suy

| Khoảng trống | Phương pháp | Lý do |
|-------------|-------------|-------|
| < 30 phút | Linear interpolation | Gap nhỏ, biến đổi tuyến tính |
| 30 phút – 2 giờ | Cubic spline | Cần smooth hơn linear |
| > 2 giờ | **KHÔNG nội suy** → NaN → Segment mới | Dữ liệu không đáng tin cậy |

### 3.4 Train/Validation/Test Split

> [!WARNING]
> **KHÔNG BAO GIỜ** dùng `shuffle=True` cho time series. Chia theo thứ tự thời gian.

```
|<------ Train (80%) ------>|<-- Val (10%) -->|<-- Test (10%) -->|
```

Cross-Validation: `TimeSeriesSplit` (expanding window).

### 3.5 Data Transformation Layers

> Tổ chức data transformations theo **3 lớp** (dbt-inspired):

```
dataset/raw/     → Staging      (src/data/loader.py)       → Load, validate format
dataset/interim/ → Intermediate  (src/data/cleaner.py)      → Clean, normalize, resample
dataset/processed/ → Marts       (src/features/builder.py)  → Feature-rich, train-ready
```

**Quy tắc**: Raw = immutable | Mỗi layer chỉ đọc layer trước | KHÔNG skip layers | Log shape/stats giữa layers.

### 3.6 Data Quality Checks

> Dùng **Great Expectations** cho standard validations, **custom validators** cho domain-specific checks (leakage, temporal consistency).

| Layer | Check | Severity | Action |
|-------|-------|----------|--------|
| Staging | Columns exist + Schema match | 🔴 Critical | STOP pipeline |
| Staging | PM2.5 range [0, 500] | 🟡 Warning | Log, clip |
| Intermediate | No NaN after interpolation | 🔴 Critical | Check logic |
| Marts | No future data leakage | 🔴 Critical | Verify lag features |

### 3.7 Data Contract, Lineage & Versioning

- **Data Contract**: Mỗi data asset có owner, freshness SLA, quality SLA
- **Lineage**: Truy vết IoT Sensor → Raw → Staging → Intermediate → Marts → Training → Predictions
- **Versioning**: SHA-256 checksums + manifest cho mỗi run

📖 **Chi tiết + code mẫu**: [`.agent/guides/data-engineering.md`](file:///Users/trihx/Desktop/time-series-forecasting/.agent/guides/data-engineering.md)

---

## 4. PHÂN TÍCH KHÁM PHÁ (EDA)

### Checklist EDA Bắt buộc

- [ ] **Univariate Analysis**: Phân phối từng biến (violin + strip, boxplot, KDE)
- [ ] **Temporal Analysis**: Patterns theo giờ, ngày, tuần, tháng (heatmap)
- [ ] **Stationarity Tests**: ADF Test, KPSS Test
- [ ] **Autocorrelation**: ACF, PACF plots → xác định lag quan trọng
- [ ] **Multivariate Analysis**: Clustered correlation heatmap, rolling correlation
- [ ] **Data Quality**: Missing values heatmap, gap analysis
- [ ] **Decomposition**: STL decomposition (Trend, Seasonal, Residual)
- [ ] **Granger Causality**: Quan hệ nhân quả giữa các biến

Output: `research/eda/YYYYMMDD_HHMMSS/eda_report.md` + `plots/`

### Interpretation Guide

| Test | p < 0.05 nghĩa là | Action |
|------|-------------------|--------|
| **ADF** | Stationary ✅ | Có thể dùng ARMA |
| **KPSS** | Non-stationary ❌ | Cần difference |
| **Ljung-Box** | Residuals correlated ❌ | Thêm AR/MA terms |
| **Granger** | X giúp predict Y ✅ | Include X as feature |

> **Lưu ý**: ADF và KPSS có null hypothesis **NGƯỢC NHAU**. Chạy **CẢ HAI** để confirm.

📖 **Chi tiết charts & templates**: [`.agent/guides/visualization-storytelling.md`](file:///Users/trihx/Desktop/time-series-forecasting/.agent/guides/visualization-storytelling.md)

---

## 5. FEATURE ENGINEERING

### 5.1 Temporal Features

```python
LAG_FEATURES = [1, 2, 3, 6, 12, 24, 48, 168]      # hours (dựa trên ACF/PACF)
ROLLING_WINDOWS = [3, 6, 12, 24, 48, 168]           # hours
ROLLING_FUNCS = ['mean', 'std', 'min', 'max']
EWM_SPANS = [12, 24, 48]                             # hours
```

### 5.2 Calendar Features

```python
CALENDAR_FEATURES = ['hour', 'day_of_week', 'day_of_month', 'month',
                     'is_weekend', 'is_rush_hour', 'season']
```

### 5.3 Domain-Specific (Air Quality)

- Chỉ số AQI breakpoint categories
- Tỷ lệ CO2/PM2.5
- Rate of change (đạo hàm bậc 1 PM2.5)

> [!TIP]
> Sau khi tạo features, **LUÔN** kiểm tra multicollinearity (VIF > 10 → loại bỏ) và feature importance (SHAP).

---

## 6. MACHINE LEARNING MODELS

### 6.1 Progression Strategy (Đơn giản → Phức tạp)

```
Level 0: Naive Baselines  → Persistence, Seasonal Naive, Mean
Level 1: Statistical       → ARIMA, SARIMA, SARIMAX, ETS, Prophet
Level 2: Linear ML         → Ridge, Lasso, ElasticNet
Level 3: Tree-based ML     → Random Forest, XGBoost, LightGBM, CatBoost
Level 4: Deep Learning     → LSTM, GRU, CNN-LSTM, Transformer
Level 5: Ensemble          → Voting, Stacking, Blending
Level 6: AutoML            → Optuna + Best model
```

> [!IMPORTANT]
> **LUÔN** bắt đầu từ Level 0 (Naive Baseline). Model phức tạp phải **chứng minh** tốt hơn baseline.

### 6.2 Scaling Strategy

| Algorithm | Scaling? | Lý do |
|-----------|----------|-------|
| RF / XGBoost / LightGBM | ❌ No | Tree-based, invariant to scale |
| Ridge / Lasso / ElasticNet | ✅ Yes | Regularization penalizes large coefficients |
| LSTM / GRU / Transformer | ✅ Yes | Gradient-based, activation functions |
| ARIMA / SARIMA | ❌ No | Works on original scale |

### 6.3 Walk-Forward Validation

> [!WARNING]
> **KHÔNG dùng K-Fold cho time series. PHẢI dùng Walk-Forward (TimeSeriesSplit).**

- **Expanding Window** (mặc định): Train trên toàn bộ history → test 1 tuần
- **Sliding Window**: Khi data distribution thay đổi theo mùa

📖 **Chi tiết + code**: [`.agent/guides/model-training.md`](file:///Users/trihx/Desktop/time-series-forecasting/.agent/guides/model-training.md)

### 6.4 Model Configs (YAML)

```yaml
# configs/model_configs/xgboost.yaml
model_name: "XGBoost"
scaling_required: false  # ← PHẢI khai báo!
params:
  n_estimators: 1000
  max_depth: 6
  learning_rate: 0.01
  subsample: 0.8
  colsample_bytree: 0.8
  min_child_weight: 3
  reg_alpha: 0.1
  reg_lambda: 1.0
  early_stopping_rounds: 50
```

---

## 7. DEEP LEARNING MODELS

### 7.1 LSTM / GRU Config

```python
DL_CONFIG = {
    "hidden_dim": 64,              # Kích thước hidden state
    "num_layers": 2,               # Số lớp stacked
    "dropout": 0.2,                # Dropout giữa các layers
    "bidirectional": False,
    "sequence_length": 168,        # 7 ngày lookback
}
```

### 7.2 Transformer Config

```python
TRANSFORMER_CONFIG = {
    "d_model": 64,                 # Embedding dimension
    "nhead": 4,                    # Attention heads
    "num_encoder_layers": 3,
    "dim_feedforward": 256,
    "dropout": 0.1,
}
```

### 7.3 Quy tắc Deep Learning

1. **Normalize/Scale** dữ liệu trước khi đưa vào model (`MinMaxScaler` hoặc `StandardScaler`).
2. **LUÔN** dùng Early Stopping (patience ≥ 10).
3. **Log** training/validation loss mỗi epoch → `training_log.csv`.
4. **Inverse transform** predictions trước khi tính metrics.
5. **Set seed** cho torch, numpy, random → dùng `set_seed(42)` helper.
6. **LR Scheduler**: Dùng `ReduceLROnPlateau` (LSTM/GRU) hoặc `CosineAnnealingWarmRestarts` (Transformer).
7. **Gradient Clipping**: `max_norm=1.0` để tránh exploding gradients.
8. **Data Windowing**: Dùng `TimeSeriesDataset` class — KHÔNG tự viết sliding window mỗi lần.

📖 **Training loop template + TimeSeriesDataset code**: [`.agent/guides/model-training.md`](file:///Users/trihx/Desktop/time-series-forecasting/.agent/guides/model-training.md)

---

## 8. PHƯƠNG PHÁP NÂNG CAO

### 8.1 Ensemble Methods

| Phương pháp | Mô tả | Khi nào dùng |
|------------|--------|-------------|
| **Simple Average** | Trung bình dự báo | Baseline ensemble |
| **Weighted Average** | Trung bình có trọng số | Biết model nào tốt hơn |
| **Stacking** | Meta-learner (Ridge) kết hợp | ≥3 models đa dạng |
| **Blending** | Stacking với holdout set | Nhanh hơn Stacking |

### 8.2 Optuna Hyperparameter Optimization

```python
import optuna

study = optuna.create_study(
    direction='minimize',
    study_name='xgboost_pm25',
    storage='sqlite:///research/experiments/optuna.db',  # PHẢI persist!
    load_if_exists=True,                                  # Resume được
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner()
)
study.optimize(objective, n_trials=100, show_progress_bar=True)
```

> [!IMPORTANT]
> Optuna study **PHẢI** persist vào SQLite để resume và phân tích sau.

### 8.3 Explainable AI (SHAP)

Sau mỗi model tốt nhất → chạy SHAP để giải thích:
1. Summary plot (feature importance tổng thể)
2. Dependence plot (ảnh hưởng từng feature)
3. Force plot (giải thích 1 dự báo cụ thể)

### 8.4 Time Series Embedding & Similarity Search

> 📌 **PLACEHOLDER** — Triển khai SAU khi có baseline models hoạt động.
> Use cases dự kiến: Similar Day Finding, Anomaly Detection, Regime Classification.
> Approach: Statistical features (baseline) → Learned embeddings (nếu cần).

📖 **Chi tiết**: [`.agent/guides/embedding-strategies.md`](file:///Users/trihx/Desktop/time-series-forecasting/.agent/guides/embedding-strategies.md)

---

## 9. ĐÁNH GIÁ & METRICS

| Metric | Ý nghĩa | Vai trò |
|--------|---------|---------|
| **MAE** | Sai số tuyệt đối trung bình | **Primary metric** |
| **RMSE** | Phạt nặng outlier errors | Secondary |
| **MAPE** | % sai số | Interpretability |
| **MASE** | So với naive baseline | **BẮT BUỘC** benchmark |
| **R²** | % variance giải thích | Overall fit |

**Quy tắc**:
- MASE < 1.0 ✅ (tốt hơn naive) | MASE ≥ 1.0 ❌ (cần cải thiện)
- **Multi-Horizon**: PHẢI đánh giá TỪNG horizon (1h, 6h, 24h) riêng biệt
- **Confidence Intervals**: Mọi forecast PHẢI kèm **95% CI** (bootstrap hoặc built-in)
- **Diebold-Mariano Test**: Khi MAE difference < 10% giữa 2 models → PHẢI chạy DM test (p < 0.05)

> [!WARNING]
> **MAPE**: Undefined khi y=0. PM2.5 có min=0 → dùng `sMAPE` hoặc skip khi y < 1.0.

📖 **Chi tiết code**: [`.agent/guides/evaluation-metrics.md`](file:///Users/trihx/Desktop/time-series-forecasting/.agent/guides/evaluation-metrics.md)

---

## 10. EXPERIMENT TRACKING

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

### Auto-Logging

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

---

## 11. UNIT TESTING

> [!IMPORTANT]
> **MỌI** chức năng mới hoặc bug fix **PHẢI** có unit test đi kèm.

### 11.1 Test Pyramid

- **70% Unit Tests**: Từng hàm (loader, cleaner, metrics, features)
- **20% Integration Tests**: Pipeline end-to-end
- **10% Model Validation**: Performance checks

### 11.2 Checklist Test Bắt buộc

**Data Tests**: Load đúng format, missing values xử lý đúng, outlier detection, resampling frequency, NO data leakage, scaler fit chỉ trên train.

**Feature Tests**: Lag/rolling tạo đúng giá trị, calendar features đúng, không NaN bất ngờ.

**Model Tests**: Fit không exception, predict đúng shape, save/load consistent, metrics tính đúng.

**Pipeline Tests**: End-to-end không lỗi, output đúng format, run directory đúng cấu trúc.

### 11.3 Pytest

```python
# tests/conftest.py - Shared fixtures
@pytest.fixture
def sample_timeseries():
    np.random.seed(42)
    n = 1000
    dates = pd.date_range('2024-01-01', periods=n, freq='h')
    return pd.DataFrame({
        'ngay_tao': dates,
        'nhiet_do': 25 + 5 * np.sin(np.arange(n) * 2*np.pi/24) + np.random.randn(n),
        'do_am': 60 + 10 * np.cos(np.arange(n) * 2*np.pi/24) + np.random.randn(n)*2,
        'diem_suong': 20 + 3 * np.sin(np.arange(n) * 2*np.pi/24) + np.random.randn(n),
        'co2': 400 + 50 * np.random.randn(n),
        'pm25': np.abs(15 + 10*np.sin(np.arange(n)*2*np.pi/24) + 5*np.random.randn(n)),
    })
```

```bash
pytest tests/ -v --cov=src --cov-report=html
```

---

## 12. XỬ LÝ LỖI & SELF-CORRECTING

### 12.1 Quy trình

```
Phát hiện lỗi → Ghi triệu chứng → Root Cause Analysis → Khắc phục
    → Viết regression test → Cập nhật LESSONS_LEARNED.md
    → Cập nhật SKILL.md nếu cần quy tắc mới
```

### 12.2 Phương pháp Debug Có Hệ thống (4 Pha)

> [!CAUTION]
> **LUẬT SẮT**: KHÔNG sửa code khi chưa tìm ra root cause.

| Pha | Mục đích | Tiêu chí hoàn thành |
|-----|---------|---------------------|
| **1. Root Cause** | Đọc errors, reproduce, trace data flow | Hiểu WHAT và WHY |
| **2. Pattern** | Tìm working code tương tự, so sánh | Xác định khác biệt |
| **3. Hypothesis** | Phát biểu giả thuyết, test tối thiểu | Confirm hoặc giả thuyết mới |
| **4. Fix** | Viết failing test, sửa 1 chỗ, verify | Bug resolved, tests pass |

**Red Flags**: "Thử sửa X xem sao" | "Fix nhanh rồi tìm sau" | "Sửa nhiều chỗ cùng lúc" | ≥3 fix thất bại → xem xét architecture.

📖 **Chi tiết đầy đủ**: [`.agent/guides/systematic-debugging.md`](file:///Users/trihx/Desktop/time-series-forecasting/.agent/guides/systematic-debugging.md)

### 12.3 Lỗi Thường Gặp Time Series

| Lỗi | Triệu chứng | Phòng tránh |
|-----|-------------|-------------|
| **Data Leakage** | Test accuracy cao bất thường | Feature chỉ dùng past data |
| **Look-ahead Bias** | Tốt trên test, xấu trên live | Scaler fit CHỈ trên train |
| **Temporal Mismatch** | Predictions lệch bước | Unit test cho lag features |
| **Overfitting** | Train loss ↓, val loss ↑ | Early stopping, dropout |

---

## 13. TÀI LIỆU THAM KHẢO

### Sách chính trong `docs/`

| Sách | Chủ đề |
|------|--------|
| Modern Time Series Forecasting with Python | ML/DL, PyTorch |
| Deep Learning for Time Series Cookbook | LSTM, CNN, Transformer |
| Time Series Forecasting in Python (Peixeiro) | ARIMA → DL, end-to-end |
| Applied Time Series Analysis (Huang) | Lý thuyết + Python |
| Time Series Analysis with R Examples (Shumway-Stoffer) | Nền tảng thống kê |
| Air Pollution Modeling | Domain knowledge |

### Academic References

| Metric | Paper |
|--------|-------|
| MASE | Hyndman & Koehler (2006) |
| RMSE vs MAE | Willmott & Matsuura (2005) |
| CRPS | Gneiting & Raftery (2007) |

## 14. ANALYTICS & EXPERIMENT DESIGN

### 14.1 Measurement Readiness Index

> Trước khi rút kết luận từ experiment, đánh giá Signal Quality:

| Tiêu chí | Weight | Kiểm tra |
|----------|--------|--------|
| Decision Alignment | 25% | Kết quả ảnh hưởng quyết định gì? |
| Metric Clarity | 20% | MAE/RMSE/MASE phù hợp? |
| Data Accuracy | 20% | Dữ liệu sạch? Không leakage? |
| Reproducibility | 15% | Seed fixed? Config logged? |
| Baseline Comparison | 10% | So với naive? MASE < 1.0? |
| Documentation | 10% | RUNS_LOG.md đầy đủ? |

**Verdict**: 85-100 ✅ Tin cậy | 70-84 ⚠️ Cần kiểm tra | <70 ❌ Không tin

### 14.2 Experiment Design (Hypothesis Lock)

> Khi so sánh models, **PHẢI** lock hypothesis + freeze primary metric TRƯỚC KHI chạy.

**Checklist**: Hypothesis locked → Primary metric frozen → Guardrails set → Same data split → Seed fixed → Baseline included.

**Anti-patterns**: ❌ "Chạy 10 models pick best" (p-hacking) | ❌ "Đổi metric sau khi xem" (HARBing) | ❌ "Tốt hơn 0.01%" (noise).

📖 **Chi tiết đầy đủ**: [`.agent/guides/analytics-experiment-design.md`](file:///Users/trihx/Desktop/time-series-forecasting/.agent/guides/analytics-experiment-design.md)

---

## PHỤ LỤC: QUICK COMMANDS

> Dự án sử dụng **uv** để quản lý Python environment và dependencies.
> File `pyproject.toml` chứa metadata + dependencies, `uv.lock` là lock file tự động.

```bash
# === Setup ===
uv sync                                    # Cài đặt dependencies
uv add <package-name>                      # Thêm dependency
uv add --dev <package-name>                # Thêm dev dependency

# === Lint & Quality ===
uv run ruff check src/ --fix               # Lint + auto-fix
uv run ruff format src/                    # Format code
uv run bandit -r src/ -ll                  # Security scan
uv run mypy src/                           # Type check

# === Testing ===
uv run pytest tests/ -v --cov=src          # Tests + coverage
uv run pytest tests/ -v --cov=src --cov-report=html  # HTML report

# === Training & Experiment ===
uv run python -m src.pipelines.train_pipeline --config configs/base_config.yaml
uv run python scripts/run_experiment.py --model xgboost --optimize --n-trials 100

# === Dashboard ===
uv run streamlit run app.py
```

## PHỤ LỤC: GUIDES REFERENCE

| Guide | Nội dung | Khi nào đọc |
|-------|---------|------------|
| [`concise-planning.md`](file:///Users/trihx/Desktop/time-series-forecasting/.agent/guides/concise-planning.md) | Template lập kế hoạch | Trước khi implement feature |
| [`kaizen.md`](file:///Users/trihx/Desktop/time-series-forecasting/.agent/guides/kaizen.md) | 4 trụ cột cải tiến | Khi refactor, review code |
| [`lint-and-validate.md`](file:///Users/trihx/Desktop/time-series-forecasting/.agent/guides/lint-and-validate.md) | Tools config & quality loop | Sau mỗi code change |
| [`systematic-debugging.md`](file:///Users/trihx/Desktop/time-series-forecasting/.agent/guides/systematic-debugging.md) | Debug 4 pha | Khi gặp bug/failure |
| [`analytics-experiment-design.md`](file:///Users/trihx/Desktop/time-series-forecasting/.agent/guides/analytics-experiment-design.md) | Experiment design ML | Trước khi so sánh models |
| [`data-engineering.md`](file:///Users/trihx/Desktop/time-series-forecasting/.agent/guides/data-engineering.md) | Pipeline, quality, lineage | Khi xây/sửa data pipeline |
| [`embedding-strategies.md`](file:///Users/trihx/Desktop/time-series-forecasting/.agent/guides/embedding-strategies.md) | TS embeddings (placeholder) | Sau khi có baseline |
| [`logging.md`](file:///Users/trihx/Desktop/time-series-forecasting/.agent/guides/logging.md) | loguru patterns, levels, timing | Khi viết code mới |
| [`visualization-storytelling.md`](file:///Users/trihx/Desktop/time-series-forecasting/.agent/guides/visualization-storytelling.md) | 12 chart templates, style, EDA report | Khi vẽ charts/EDA |
| [`model-training.md`](file:///Users/trihx/Desktop/time-series-forecasting/.agent/guides/model-training.md) | Scaling, walk-forward, DL loop, windowing | Khi train models |
| [`evaluation-metrics.md`](file:///Users/trihx/Desktop/time-series-forecasting/.agent/guides/evaluation-metrics.md) | Multi-horizon, DM test, CI, ensemble | Khi evaluate/compare models |
