# Nguyên tắc Kaizen — Cải tiến Liên tục

> **Nguồn**: Skill `kaizen` — 4 trụ cột, adapted cho dự án Time Series Forecasting PM2.5
>
> **Triết lý**: Nhiều cải tiến nhỏ tích lũy thành thay đổi lớn. Phòng lỗi từ thiết kế, không phải từ patch.

---

## 4 Trụ cột Kaizen

### Trụ cột 1: Cải tiến Liên tục (Continuous Improvement)

> Mỗi lần commit = code tốt hơn trước

**Nguyên tắc:**

- Thay đổi nhỏ nhất có thể cải thiện chất lượng → làm ngay
- Một cải tiến tại một thời điểm
- Verify mỗi thay đổi trước khi tiếp
- Tạo momentum từ small wins

**Luôn để code tốt hơn khi rời đi:**

- Fix lỗi nhỏ khi gặp (trong scope)
- Cập nhật comments lỗi thời
- Xóa dead code khi thấy
- Refactor nhẹ trong khi làm việc

**Quy trình 3 Vòng (Iterative Refinement):**

```
Vòng 1: Làm cho CHẠY (make it work)
  → Code chạy đúng, có test cơ bản ✅
  
Vòng 2: Làm cho RÕ (make it clear)
  → Refactor, đặt tên tốt, thêm docs ✅
  
Vòng 3: Làm cho TỐT (make it efficient)
  → Optimize CHỈ KHI đo được bottleneck ✅

⚠️ KHÔNG làm cả 3 cùng lúc!
```

**Khi refactor:**

- Fix one code smell tại một thời điểm
- Commit sau mỗi cải tiến
- Giữ tests passing xuyên suốt
- Dừng khi "good enough" (diminishing returns)

---

### Trụ cột 2: Poka-Yoke (Chống lỗi bằng Thiết kế)

> Thiết kế để lỗi KHÔNG THỂ xảy ra, thay vì fix sau khi lỗi xảy ra

**4 lớp phòng thủ (Defense in Depth):**

```
Lớp 1: Type System (compile/analysis time)
  → Type hints, NewType, Protocol

Lớp 2: Validation (runtime, tại boundary)
  → validate_path(), validate_pm25()

Lớp 3: Guards (preconditions)
  → Early returns, assert trong dev

Lớp 4: Error Boundaries (graceful degradation)
  → try/except cụ thể, fallback
```

**Áp dụng cho Python Data Science:**

```python
from typing import NewType
from pathlib import Path

# ✅ Validate tại boundary, an toàn bên trong
PositiveFloat = NewType('PositiveFloat', float)

def validate_pm25(value: float) -> PositiveFloat:
    """Validate PM2.5 tại boundary — gọi 1 lần khi nhận dữ liệu."""
    if value < 0:
        raise ValueError(f"PM2.5 không thể âm: {value}")
    if value > 1000:
        raise ValueError(f"PM2.5 bất thường (>1000 µg/m³): {value}")
    return PositiveFloat(value)

# ❌ KHÔNG kiểm tra lại sâu bên trong pipeline
# Sau khi validate, giá trị được trust trong toàn bộ xử lý

# ✅ Fail fast tại config
def load_config(config_path: str) -> dict:
    """Load và validate config — fail tại startup, không phải khi chạy."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config không tồn tại: {path}")
    
    config = yaml.safe_load(path.read_text())
    
    required_keys = ["model_name", "params", "data_path"]
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise ValueError(f"Config thiếu keys: {missing}")
    
    return config
```

**Quy tắc Poka-Yoke:**

| Tình huống | Ứng dụng |
|-----------|---------|
| API/Function inputs | Validate tại boundary, trust bên trong |
| Configuration | Validate toàn bộ tại startup, fail sớm |
| Data pipeline | Validate shape/type giữa mỗi stage |
| Model loading | Kiểm tra source trước khi deserialize |

---

### Trụ cột 3: Chuẩn hóa (Standardized Work)

> Follow pattern đã có, không sáng tạo khi không cần thiết

**Áp dụng trong dự án:**

- **Tuân thủ BaseModel interface** — Mọi model mới PHẢI kế thừa `BaseModel` (SKILL.md Section 2.3)
- **Naming convention** — Tuân theo convention trong project (snake_case, module names)
- **Error handling pattern** — Luôn dùng `except SpecificException`, luôn log
- **Config pattern** — Mọi tham số từ config files, không hard-code

**Trước khi viết pattern mới:**

1. Tìm xem project đã có code tương tự chưa
2. Nếu có → follow pattern đó
3. Nếu pattern mới tốt hơn → thảo luận, ghi vào `DECISIONS.md`, rồi áp dụng nhất quán

**Automate standards:**

- Linters (Ruff) enforce style → xem `lint-and-validate.md`
- Type checks (MyPy) enforce contracts
- Tests verify behavior
- `pyproject.toml` là single source of truth cho tools config

---

### Trụ cột 4: Just-In-Time (Chỉ Build Cái Cần)

> YAGNI — You Aren't Gonna Need It

**Nguyên tắc YAGNI:**

- Implement CHỈ requirement hiện tại
- KHÔNG code "đề phòng" hay "có thể cần sau này"
- Xóa speculative code
- Thêm complexity chỉ khi được yêu cầu hoặc đo được bottleneck

**Rule of Three (Quy tắc Trước khi Abstract):**

| Số lần gặp | Hành động |
|-----------|----------|
| **1 case** | Viết code trực tiếp, cụ thể |
| **2 cases tương tự** | Chấp nhận duplication nhẹ |
| **3+ cases giống nhau** | LÚC NÀY MỚI abstract thành utils/base class |

```python
# ✅ ĐÚNG: 1 model type → code trực tiếp
def train_random_forest(X, y, params):
    model = RandomForestRegressor(**params)
    model.fit(X, y)
    return model

# ❌ SAI: Chưa có 3 model types mà đã build framework
class ModelFactory:
    _registry = {}
    @classmethod
    def register(cls, name): ...
    @classmethod
    def create(cls, name, **kwargs): ...
    # 100 dòng code cho "tương lai"
```

**Khi nào ĐƯỢC thêm complexity:**

- Requirement hiện tại yêu cầu
- Pain point đã xác định qua sử dụng thực tế
- Performance issue đã đo được (profile trước, optimize sau)
- Pattern đã xuất hiện 3+ lần

---

## Red Flags

| Vi phạm | Dấu hiệu |
|---------|----------|
| **Continuous Improvement** | "Refactor sau" (sẽ không bao giờ xảy ra) |
| **Poka-Yoke** | "User nên cẩn thận hơn khi dùng API này" |
| **Standardized Work** | "Tôi thích cách khác hơn" (bỏ qua convention) |
| **Just-In-Time** | "Có thể cần someday" (build cho tương lai) |

---

## Tổng kết

**Kaizen LÀ:**

- Cải tiến nhỏ, liên tục
- Phòng lỗi bằng thiết kế
- Follow pattern đã chứng minh
- Build chỉ cái cần ngay

**Kaizen KHÔNG phải:**

- Hoàn hảo từ lần đầu
- Massive refactoring projects
- Abstraction thông minh nhưng không ai cần
- Optimize trước khi đo

> **Mindset**: Good enough today, better tomorrow. Repeat.
