# 📘 Hướng Dẫn Triển Khai Dự Án Mới — Personal Engineering Playbook

> **Tác giả:** Anh Trí (trihx) + AI Agent
> **Mục đích:** Hướng dẫn chi tiết cách áp dụng hệ thống kiến thức (Global Playbook) cho BẤT KỲ dự án mới nào, kể cả trên máy tính khác.
> **Cập nhật:** 2026-04-01

---

## 📋 Mục Lục

1. [Tổng Quan Hệ Thống](#1-tổng-quan-hệ-thống)
2. [Cài Đặt Máy Tính Mới](#2-cài-đặt-máy-tính-mới)
3. [Khởi Tạo Dự Án Mới](#3-khởi-tạo-dự-án-mới)
4. [Tùy Chỉnh & Cá Nhân Hóa](#4-tùy-chỉnh--cá-nhân-hóa)
5. [Quy Trình Làm Việc Hàng Ngày](#5-quy-trình-làm-việc-hàng-ngày)
6. [Bảo Trì & Cập Nhật Playbook](#6-bảo-trì--cập-nhật-playbook)
7. [Checklist Nhanh](#7-checklist-nhanh)
8. [FAQ & Xử Lý Sự Cố](#8-faq--xử-lý-sự-cố)

---

## 1. Tổng Quan Hệ Thống

### Kiến trúc 3 tầng kiến thức

```
🌍 TẦNG 1: Global Playbook (portable — mang theo mọi nơi)
│   Vị trí: ~/.gemini/AGENTS.md
│   Nội dung: Quy tắc cá nhân, patterns, testing, debug, phong cách
│   Áp dụng: TẤT CẢ dự án, TẤT CẢ máy tính
│
├── 🏠 TẦNG 2: Project AGENTS.md (riêng từng dự án)
│   │   Vị trí: <project-root>/AGENTS.md
│   │   Nội dung: Tech stack, commands, architecture, gotchas riêng dự án
│   │
│   └── 📝 TẦNG 3: Project Memory (tích lũy theo thời gian)
│       Vị trí: <project-root>/docs/
│       ├── PROJECT_MEMORY.md    # HOT — agent đọc mỗi phiên (≤150 dòng)
│       ├── DECISIONS_LOG.md     # COLD — tra cứu khi cần
│       └── CHANGELOG.md        # Nhật ký thay đổi
```

### Nguyên tắc hoạt động

| Khi agent bắt đầu phiên | Nó đọc gì? | Mục đích |
|--------------------------|-----------|----------|
| Bước 1 | `~/.gemini/AGENTS.md` (Global) | Hiểu phong cách, quy tắc chung |
| Bước 2 | `./AGENTS.md` (Project) | Hiểu tech stack, commands riêng dự án |
| Bước 3 | `docs/PROJECT_MEMORY.md` | Nhớ context, gotchas, decisions đã làm |
| Khi cần | `docs/DECISIONS_LOG.md` | Tra cứu chi tiết bug/quyết định cũ |

---

## 2. Cài Đặt Máy Tính Mới

### 2.1 Phần mềm bắt buộc

| Phần mềm | Mục đích | Cài đặt |
|-----------|---------|---------|
| **Python 3.11+** | Runtime chính | [python.org](https://www.python.org/downloads/) |
| **UV** | Package manager (thay pip) | `powershell -c "irm https://astral.sh/uv/install.ps1 \| iex"` |
| **Git** | Version control | [git-scm.com](https://git-scm.com/downloads) |
| **Antigravity** | AI coding agent | [antigravity.google](https://antigravity.google) |

### 2.2 Phần mềm tùy chọn (theo nhu cầu dự án)

| Phần mềm | Khi nào cần | Cài đặt |
|-----------|------------|---------|
| **PostgreSQL 17** | Dự án cần DB quan hệ | [postgresql.org](https://www.postgresql.org/download/) |
| **LM Studio** | Dự án dùng Local LLM | [lmstudio.ai](https://lmstudio.ai/) |
| **NVIDIA CUDA** | Dự án dùng GPU (ML/DL) | [developer.nvidia.com/cuda](https://developer.nvidia.com/cuda-downloads) |
| **Node.js** | Dự án có frontend web | [nodejs.org](https://nodejs.org/) |
| **Ngrok** | Cần tunnel công khai | [ngrok.com](https://ngrok.com/download) |

### 2.3 Cài đặt Global Playbook

Đây là bước **quan trọng nhất** — chỉ cần làm **1 lần** trên mỗi máy tính mới.

#### Bước 1: Tạo thư mục Antigravity (nếu chưa có)

```powershell
# Windows PowerShell
New-Item -ItemType Directory -Path "$env:USERPROFILE\.gemini" -Force
```

```bash
# macOS / Linux
mkdir -p ~/.gemini
```

#### Bước 2: Copy Global Playbook

**Cách 1: Copy thủ công** (nhanh nhất)
- Copy file `~/.gemini/AGENTS.md` từ máy cũ sang máy mới
- Vị trí: `C:\Users\<username>\.gemini\AGENTS.md` (Windows)

**Cách 2: Dùng Git** (khuyến nghị cho đồng bộ nhiều máy)
```powershell
# Tạo repo riêng cho playbook
cd $env:USERPROFILE
git init .gemini-playbook
# Sau đó symlink hoặc copy AGENTS.md vào ~/.gemini/
```

**Cách 3: Từ dự án RAG_bot** (nếu có sẵn)
```powershell
# Copy từ RAG_bot docs
Copy-Item "D:\01-Repos\RAG_bot\docs\PLAYBOOK_SETUP_GUIDE.md" "$env:USERPROFILE\.gemini\"
# Global Playbook đã có sẵn tại ~/.gemini/AGENTS.md
```

#### Bước 3: Verify

```powershell
# Kiểm tra file tồn tại
Get-Content "$env:USERPROFILE\.gemini\AGENTS.md" | Select-Object -First 3
# Output mong đợi:
# # 🧬 Personal Engineering Playbook — Global Agent Rules
```

#### Bước 4: Cấu hình Antigravity đọc Global Playbook

Antigravity tự động đọc `~/.gemini/AGENTS.md` như **user-level rules**. Kiểm tra:
1. Mở Antigravity → Settings → User Rules
2. Đảm bảo file `~/.gemini/AGENTS.md` được liệt kê
3. Nếu chưa có → thêm đường dẫn vào User Rules

> ⚠️ **LƯU Ý:** Nếu trước đó anh đã set User Rules trỏ vào `AGENTS.md` của dự án cụ thể (VD: RAG_bot), hãy đổi sang trỏ vào Global Playbook thay vì project-specific file.

### 2.4 Cấu hình môi trường chung

```powershell
# Thiết lập Git (1 lần trên máy mới)
git config --global user.name "Anh Trí"
git config --global user.email "your-email@example.com"

# Đảm bảo UV trong PATH
uv --version   # Nếu lỗi → restart terminal hoặc thêm vào PATH
```

---

## 3. Khởi Tạo Dự Án Mới

### 3.1 Tạo structure cơ bản

```powershell
# Tạo thư mục dự án
mkdir D:\01-Repos\TenDuAn
cd D:\01-Repos\TenDuAn

# Khởi tạo Git
git init

# Khởi tạo Python project với UV
uv init
```

### 3.2 Tạo Project AGENTS.md

Tạo file `AGENTS.md` ở root dự án. Dưới đây là **template** — chỉnh sửa theo dự án:

```markdown
# <TÊN DỰ ÁN> — Agent Rules

> Quy tắc riêng cho dự án này. Quy tắc chung xem `~/.gemini/AGENTS.md`.
> Đọc file này + `docs/PROJECT_MEMORY.md` TRƯỚC khi thay đổi code.

---

## 🔒 Quy Tắc Dự Án

1. **Đọc `docs/PROJECT_MEMORY.md`** trước khi implement.
2. **Chạy test:** `uv run pytest tests/ -v --tb=short`
3. **Chạy app:** `uv run main.py` (hoặc `uv run streamlit run app.py`)

---

## 🏗️ Tech Stack

| Thành phần | Công nghệ |
|-----------|----------|
| Ngôn ngữ | Python 3.11 |
| Framework | (Flask / FastAPI / Streamlit / ...) |
| Database | (PostgreSQL / SQLite / ...) |
| ML/DL | (scikit-learn / PyTorch / ...) |

---

## 📁 Cấu Trúc Thư Mục

```
TenDuAn/
├── main.py / app.py         # Entry point
├── config.py                # Config tập trung
├── src/
│   ├── services/            # Business logic
│   ├── domain/              # DTOs & interfaces
│   └── infrastructure/      # DB, external APIs
├── tests/
│   └── unit/                # Unit tests
├── data/                    # Data files
├── docs/                    # Documentation
│   ├── PROJECT_MEMORY.md
│   ├── DECISIONS_LOG.md
│   └── CHANGELOG.md
└── .agent/workflows/        # Agent workflows
```

---

## 🔄 Workflows

| Workflow | Mô tả |
|----------|-------|
| `/test` | Chạy test suite |
| `/update-memory` | Cập nhật PROJECT_MEMORY |

---

## ⚠️ Known Gotchas

(Thêm dần khi phát hiện vấn đề riêng dự án)
```

### 3.3 Tạo docs skeleton

```powershell
# Tạo thư mục docs
mkdir docs

# Tạo PROJECT_MEMORY.md (HOT memory)
```

Template cho `docs/PROJECT_MEMORY.md`:

```markdown
# 🧠 <TÊN DỰ ÁN> — Bộ Nhớ Dự Án (HOT Memory)

> **Cập nhật:** YYYY-MM-DD | **Giới hạn:** ≤150 dòng
> **Chi tiết:** xem `docs/DECISIONS_LOG.md`

---

## 1. Kiến Trúc Hiện Tại

| Ver | Quyết định | Kỹ thuật chính |
|-----|-----------|----------------|
| 0.1 | (Quyết định đầu tiên) | (Kỹ thuật dùng) |

---

## 2. ⚠️ Gotchas — PHẢI NHỚ

(Thêm dần khi phát hiện)

---

## 3. Hardware & Constraints

| Thành phần | Thông số |
|-----------|---------|
| GPU | (nếu có) |
| RAM | ... |

---

## 4. Roadmap

- [ ] Feature 1
- [ ] Feature 2

---

> 📌 Quy tắc: ≤150 dòng. Chi tiết → `DECISIONS_LOG.md`
```

### 3.4 Tạo workflows cơ bản

```powershell
# Tạo thư mục workflows
mkdir -p .agent\workflows
```

Template cho `.agent/workflows/test.md`:
```markdown
---
description: Chạy toàn bộ unit tests
---
// turbo-all

1. Chạy pytest:
   ```bash
   uv run pytest tests/ -v --tb=short
   ```
2. Kiểm tra kết quả: đảm bảo tất cả tests PASSED.
```

### 3.5 Tạo file .env và .gitignore

```powershell
# Tạo .env
echo "# Environment variables" > .env

# Tạo .gitignore
```

Nội dung `.gitignore` khuyến nghị:
```
# Python
__pycache__/
*.py[cod]
.venv/

# Environment
.env

# Data
data/*.db
data/*.sqlite
*.faiss

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db
```

---

## 4. Tùy Chỉnh & Cá Nhân Hóa

### 4.1 Chỉnh sửa Global Playbook

File `~/.gemini/AGENTS.md` có **6 vị trí** đánh dấu `<!-- 📝 BỔ SUNG -->` để anh mở rộng:

| Vị trí | Section | Khi nào bổ sung |
|--------|---------|----------------|
| 1 | Patterns Phổ Biến | Khi tìm thấy pattern mới dùng được nhiều dự án |
| 2 | Mock Patterns | Khi gặp mock pattern mới (ML, pandas, Streamlit) |
| 3 | Frontend ↔ Backend | Khi tích lũy patterns cho Streamlit, Next.js, etc. |
| 4 | Debug Known Traps | Khi phát hiện bẫy mới (NaN, timezone, dtype) |
| 5 | Workflows | Khi có workflow template mới (/eda, /deploy, /lint) |
| 6 | Tech Stack | Khi thêm công cụ mới vào stack |

### 4.2 Cách bổ sung an toàn

```markdown
<!-- TRƯỚC -->
### Patterns Phổ Biến
- **Tiered Fallback:** `Primary → Secondary → Tertiary`
- **Factory Method:** Centralized selection
<!-- 📝 BỔ SUNG: Thêm patterns mới -->

<!-- SAU (thêm 1 dòng) -->
### Patterns Phổ Biến
- **Tiered Fallback:** `Primary → Secondary → Tertiary`
- **Factory Method:** Centralized selection
- **Pipeline Pattern:** `Load → Clean → Transform → Model → Evaluate`
<!-- 📝 BỔ SUNG: Thêm patterns mới -->
```

> ⚠️ **QUY TẮC:** Global Playbook ≤ 200 dòng. Nếu phình → nén hoặc tách module.

### 4.3 Tùy chỉnh theo loại dự án

| Loại dự án | Cần thêm gì vào Project AGENTS.md? |
|-----------|-------------------------------------|
| **Web App (Flask/FastAPI)** | API endpoints, route naming, authentication pattern |
| **Data Science (pandas/ML)** | EDA workflow, data pipeline rules, notebook conventions |
| **Streamlit Dashboard** | Page routing, caching rules (`st.cache_data`), component naming |
| **Telegram/Chat Bot** | Handler patterns, command routing, message formatting |
| **ML/DL Training** | Experiment tracking, model versioning, GPU memory management |

### 4.4 Đồng bộ Playbook giữa nhiều máy

**Cách 1: Git repo riêng** (khuyến nghị)
```powershell
# Máy 1: Push
cd $env:USERPROFILE\.gemini
git add AGENTS.md
git commit -m "update playbook"
git push

# Máy 2: Pull
cd $env:USERPROFILE\.gemini
git pull
```

**Cách 2: Cloud sync** (OneDrive / Google Drive)
- Lưu `AGENTS.md` trong thư mục đồng bộ
- Tạo symlink từ `~/.gemini/AGENTS.md` → file trong cloud

```powershell
# Windows: Tạo symlink
New-Item -ItemType SymbolicLink `
  -Path "$env:USERPROFILE\.gemini\AGENTS.md" `
  -Target "D:\OneDrive\dev-config\AGENTS.md"
```

**Cách 3: Copy thủ công** (đơn giản nhất)
- USB / email / chat — copy file khi cần

---

## 5. Quy Trình Làm Việc Hàng Ngày

### 5.1 Bắt đầu phiên làm việc

```
1. Mở Antigravity → open project
2. Agent tự động đọc:
   ├── ~/.gemini/AGENTS.md (Global)
   ├── ./AGENTS.md (Project)
   └── docs/PROJECT_MEMORY.md (nếu có)
3. Bắt đầu làm việc — agent đã có đầy đủ context
```

### 5.2 Trong phiên làm việc

| Hành động | Nên làm |
|----------|---------|
| Implement feature mới | Agent tự lập plan → anh review → execute |
| Phát hiện bug | Agent ghi vào PROJECT_MEMORY gotchas |
| Học bài học mới | Bài học portable → Global Playbook. Riêng dự án → PROJECT_MEMORY |
| Hoàn thành feature | Chạy `/test` → `/update-memory` → commit |

### 5.3 Kết thúc phiên

```
1. Chạy /test — đảm bảo không regression
2. Chạy /update-memory — lưu context cho phiên sau
3. Commit code + docs cùng lúc
4. (Tùy chọn) Review PROJECT_MEMORY — xóa gotchas hết relevant
```

---

## 6. Bảo Trì & Cập Nhật Playbook

### 6.1 Khi nào cập nhật Global Playbook?

| Tình huống | Hành động |
|-----------|----------|
| Học pattern mới dùng được nhiều dự án | ✅ Thêm vào Global |
| Tìm bẫy debug mới phổ biến | ✅ Thêm vào Global |
| Đổi tool mặc định (VD: pandas → polars) | ✅ Cập nhật Tech Stack |
| Bug chỉ xảy ra 1 dự án | ❌ Giữ trong PROJECT_MEMORY |
| Quyết định kiến trúc riêng 1 dự án | ❌ Giữ trong DECISIONS_LOG |

### 6.2 Review định kỳ

**Mỗi tháng** (hoặc sau khi hoàn thành 1 milestone lớn):
1. Đọc lại Global Playbook → xóa/cập nhật phần lỗi thời
2. Đọc lại PROJECT_MEMORY → nén nếu > 150 dòng
3. Kiểm tra DECISIONS_LOG → có gì nên "promote" lên PROJECT_MEMORY không?

### 6.3 Migration checklist khi chuyển máy

- [ ] Cài Python, UV, Git
- [ ] Copy `~/.gemini/AGENTS.md` (Global Playbook)
- [ ] Clone các project repos
- [ ] Cài Antigravity
- [ ] Verify: mở 1 project → agent đọc đúng cả Global + Project rules
- [ ] (Tùy chọn) Cài PostgreSQL, LM Studio, CUDA theo nhu cầu

---

## 7. Checklist Nhanh

### ✅ Dự án mới — Khởi tạo trong 5 phút

```
[ ] 1. mkdir project && cd project
[ ] 2. git init && uv init
[ ] 3. Tạo AGENTS.md (copy template từ Section 3.2)
[ ] 4. mkdir docs && tạo PROJECT_MEMORY.md (template Section 3.3)
[ ] 5. mkdir -p .agent/workflows && tạo test.md
[ ] 6. Tạo .env + .gitignore
[ ] 7. Mở Antigravity → bắt đầu làm việc
```

### ✅ Máy tính mới — Thiết lập trong 15 phút

```
[ ] 1. Cài Python, UV, Git
[ ] 2. Cài Antigravity
[ ] 3. Copy ~/.gemini/AGENTS.md (Global Playbook)
[ ] 4. Verify cấu hình Antigravity user rules
[ ] 5. Clone project repos
[ ] 6. (Tùy chọn) Cài tools theo dự án cần
```

---

## 8. FAQ & Xử Lý Sự Cố

### Q: Agent không đọc Global Playbook?
**A:** Kiểm tra:
1. File tồn tại tại `~/.gemini/AGENTS.md`?
2. Antigravity Settings → User Rules có trỏ đúng không?
3. File có bị rỗng (0 bytes) không?

### Q: Agent đọc rules dự án cũ thay vì dự án hiện tại?
**A:** Antigravity user rules chỉ nên trỏ vào Global Playbook (`~/.gemini/AGENTS.md`), KHÔNG trỏ vào `AGENTS.md` của 1 dự án cụ thể. Project-specific rules được tự động đọc từ `./AGENTS.md` trong workspace.

### Q: Global Playbook quá dài (>200 dòng)?
**A:** Nén hoặc tách thành modules:
```
~/.gemini/
├── AGENTS.md              # Core rules (≤200 dòng)
├── patterns/
│   ├── python_async.md    # Async patterns chi tiết
│   └── ml_pipeline.md     # ML patterns chi tiết
```
Agent có thể đọc sub-files khi cần bằng lệnh `@import` hoặc tham chiếu.

### Q: Dự án dùng ngôn ngữ khác (Node.js, Rust, etc.)?
**A:** Global Playbook vẫn áp dụng được (testing discipline, debug methodology, documentation standards). Chỉ cần:
- Bỏ qua phần Python-specific (mock patterns, UV)
- Thêm rules riêng vào Project AGENTS.md

### Q: Làm sao biết bài học nào nên vào Global vs Project?
**A:** Hỏi: *"Nếu anh làm dự án hoàn toàn khác, bài học này còn hữu ích không?"*
- **Có** → Global Playbook
- **Không** → PROJECT_MEMORY của dự án đó

### Q: Có thể dùng chung cho team không?
**A:** Có! Chia thành:
- `~/.gemini/AGENTS.md` → Rules cá nhân (mỗi dev 1 file)
- `./AGENTS.md` → Rules dự án (commit vào repo, tất cả dev dùng chung)

---

## 📎 Phụ Lục: Danh Sách File Tham Chiếu

| File | Vị trí | Vai trò |
|------|--------|---------|
| **Global Playbook** | `~/.gemini/AGENTS.md` | Rules áp dụng mọi dự án |
| **Project Rules** | `<project>/AGENTS.md` | Rules riêng dự án |
| **HOT Memory** | `<project>/docs/PROJECT_MEMORY.md` | Context nhanh ≤150 dòng |
| **COLD Archive** | `<project>/docs/DECISIONS_LOG.md` | Chi tiết quyết định & bugs |
| **Changelog** | `<project>/docs/CHANGELOG.md` | Nhật ký thay đổi |
| **Workflows** | `<project>/.agent/workflows/*.md` | Tự động hóa tác vụ |

---

> 📌 **Ghi nhớ:** Mục tiêu cuối cùng là khi bắt đầu bất kỳ dự án nào, trên bất kỳ máy nào, agent đã sẵn sàng làm việc đúng phong cách của anh mà **không cần dạy lại bất cứ điều gì**.
