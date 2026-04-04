# Model Training Guide — ML & Deep Learning

> Bổ sung chi tiết kỹ thuật cho SKILL.md §6-§7. Code templates và best practices.

---

## 1. Scaling Strategy Decision Table

> [!IMPORTANT]
> **Tree-based models KHÔNG cần scaling. Linear/Neural models CẦN scaling.**

| Algorithm | Scaling Required? | Scaler Recommendation | Lý do |
|-----------|-------------------|----------------------|-------|
| Random Forest | ❌ No | — | Decision trees split on thresholds, invariant to scale |
| XGBoost / LightGBM / CatBoost | ❌ No | — | Tree-based, same reason |
| Ridge / Lasso / ElasticNet | ✅ Yes | `StandardScaler` | Regularization penalizes large coefficients |
| SVR | ✅ Yes | `StandardScaler` | Kernel distance-based |
| LSTM / GRU / Transformer | ✅ Yes | `MinMaxScaler(0,1)` or `StandardScaler` | Gradient-based optimization, sigmoid/tanh activations |
| ARIMA / SARIMA | ❌ No | — | Works on original scale |

### sklearn Pipeline Pattern

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

# Anti data-leakage: scaler INSIDE pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Ridge(alpha=1.0)),
])

# Time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_val, y_val)
```

---

## 2. Walk-Forward Validation

> [!IMPORTANT]
> **KHÔNG dùng K-Fold cho time series. PHẢI dùng Walk-Forward.**

### Expanding Window (khuyên dùng cho project này)

```
Train: [====]           → Test: [==]
Train: [======]         → Test: [==]
Train: [========]       → Test: [==]
Train: [==========]     → Test: [==]
```

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5, test_size=24*7)  # test = 1 week
for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    logger.info(f"Fold {fold+1}: train={len(train_idx)}, test={len(test_idx)}")
    # Train trên toàn bộ history → test trên 1 tuần tiếp theo
```

### Sliding Window (khi data distribution thay đổi theo mùa)

```
Train: [====]           → Test: [==]
   Train: [====]        → Test: [==]
      Train: [====]     → Test: [==]
         Train: [====]  → Test: [==]
```

```python
# Custom sliding window
TRAIN_SIZE = 24 * 30 * 6  # 6 months
TEST_SIZE = 24 * 7        # 1 week
STEP = 24 * 7             # slide 1 week

for start in range(0, len(X) - TRAIN_SIZE - TEST_SIZE, STEP):
    train_end = start + TRAIN_SIZE
    test_end = train_end + TEST_SIZE
    X_train, y_train = X[start:train_end], y[start:train_end]
    X_test, y_test = X[train_end:test_end], y[train_end:test_end]
```

### Khi nào chọn gì?

| Trường hợp | Dùng | Lý do |
|------------|------|-------|
| Dữ liệu stationary, trend ổn định | Expanding Window | Nhiều data hơn = model tốt hơn |
| Distribution thay đổi theo mùa | Sliding Window | Recent data quan trọng hơn old data |
| **Project này** (3 năm PM2.5) | **Expanding Window** | Climate patterns ổn định, cần đủ data cho seasonal patterns |

---

## 3. Feature Selection Pipeline (4 bước)

```python
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np

def feature_selection_pipeline(X, y, feature_names):
    """4-step feature selection."""

    # Step 1: Remove high-correlation features (|r| > 0.95)
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    drop_corr = [col for col in upper.columns if any(upper[col] > 0.95)]
    logger.info(f"Step 1 — Dropped (corr>0.95): {drop_corr}")
    X = X.drop(columns=drop_corr)

    # Step 2: VIF check (> 10 = multicollinearity)
    vif_data = pd.DataFrame({
        "Feature": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    })
    drop_vif = vif_data[vif_data["VIF"] > 10]["Feature"].tolist()
    logger.info(f"Step 2 — VIF > 10: {drop_vif}")
    X = X.drop(columns=drop_vif)

    # Step 3: Statistical filter (F-test)
    selector = SelectKBest(f_regression, k=min(20, X.shape[1]))
    selector.fit(X, y)
    selected = X.columns[selector.get_support()].tolist()
    logger.info(f"Step 3 — F-test selected: {len(selected)} features")

    # Step 4: Tree-based importance (RF)
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X[selected], y)
    importances = pd.Series(rf.feature_importances_, index=selected).sort_values(ascending=False)
    final = importances[importances > 0.01].index.tolist()
    logger.info(f"Step 4 — RF importance > 1%: {final}")

    return final
```

---

## 4. PyTorch Training Loop Template

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from loguru import logger
import numpy as np
import time

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    lr: float = 0.001,
    patience: int = 10,
    device: str = "cpu",
    run_dir: str = "research/runs/",
):
    """Standard PyTorch training loop with early stopping."""

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "lr": []}

    for epoch in range(epochs):
        start = time.perf_counter()

        # === Training ===
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred.squeeze(), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            train_losses.append(loss.item())

        # === Validation ===
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                loss = criterion(y_pred.squeeze(), y_batch)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.perf_counter() - start

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        # LR Scheduler
        scheduler.step(val_loss)

        logger.info(
            f"Epoch {epoch+1:03d}/{epochs} | "
            f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
            f"LR: {current_lr:.2e} | {elapsed:.1f}s"
        )

        # === Early Stopping ===
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"{run_dir}/best_model.pt")
            logger.info(f"  ✓ New best model saved (val_loss={val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.warning(f"  Early stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(torch.load(f"{run_dir}/best_model.pt"))
    logger.info(f"Training complete. Best val_loss: {best_val_loss:.6f}")

    return model, history
```

---

## 5. Time Series Dataset (Data Windowing)

```python
import torch
from torch.utils.data import Dataset
import numpy as np

class TimeSeriesDataset(Dataset):
    """Convert time series → sliding windows for LSTM/Transformer."""

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 168):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        # X: [idx : idx+seq_len] → y: [idx+seq_len]
        x = self.X[idx : idx + self.seq_len]
        y = self.y[idx + self.seq_len]
        return x, y


def create_sequences(df, target_col, feature_cols, seq_len=168, horizon=1):
    """Create train/val/test DataLoaders with proper temporal split."""
    features = df[feature_cols].values
    target = df[target_col].values

    # Temporal split (NO shuffle)
    n = len(features)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    # Scale (fit on train only)
    from sklearn.preprocessing import StandardScaler
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    features[:train_end] = scaler_X.fit_transform(features[:train_end])
    features[train_end:] = scaler_X.transform(features[train_end:])
    target[:train_end] = scaler_y.fit_transform(target[:train_end].reshape(-1, 1)).flatten()
    target[train_end:] = scaler_y.transform(target[train_end:].reshape(-1, 1)).flatten()

    train_ds = TimeSeriesDataset(features[:train_end], target[:train_end], seq_len)
    val_ds = TimeSeriesDataset(features[train_end:val_end], target[train_end:val_end], seq_len)
    test_ds = TimeSeriesDataset(features[val_end:], target[val_end:], seq_len)

    return train_ds, val_ds, test_ds, scaler_X, scaler_y
```

---

## 6. Reproducibility Checklist

```python
import torch
import numpy as np
import random

def set_seed(seed: int = 42):
    """Set seed cho tất cả random generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```
