# 🔄 Hướng dẫn Đồng bộ Cấu hình Global Antigravity qua GitHub

Tài liệu này hướng dẫn chi tiết từng bước để đồng bộ hóa **Global Rules (`AGENTS.md`)** và **Global Skills** của tài khoản Antigravity giữa nhiều máy tính khác nhau thông qua một kho lưu trữ (repository) riêng tư trên GitHub.

---

## 📂 1. Cấu trúc Thư mục cấu hình Global

Trên macOS, Antigravity lưu trữ các thiết lập dùng chung cho mọi dự án tại thư mục ẩn `~/.gemini` (tương đương `/Users/YOUR_USERNAME/.gemini`). Cấu trúc gồm:
* `~/.gemini/AGENTS.md` — File chứa các quy tắc global (System Instructions, phong cách giao tiếp, background cá nhân).
* `~/.gemini/config/skills/` — Thư mục chứa các Skill global mà anh có thể gọi ở mọi dự án.

---

## 🛠️ 2. Các bước thiết lập trên Máy tính hiện tại (Máy nguồn)

### Bước 2.1: Tạo Private Repository trên GitHub
1. Truy cập [github.com/new](https://github.com/new).
2. Đặt tên repository (ví dụ: `antigravity_skill`).
3. **Bắt buộc chọn chế độ Private** để bảo vệ thông tin cá nhân và cấu hình rules.
4. Click **Create repository** (không chọn tạo sẵn README, .gitignore hay License).

### Bước 2.2: Tạo Personal Access Token (PAT) trên GitHub
*Từ năm 2021, GitHub yêu cầu dùng Token thay cho mật khẩu thường khi thao tác qua giao thức HTTPS.*

1. Click vào ảnh đại diện của anh ở góc trên bên phải GitHub → Chọn **Settings**.
2. Cuộn xuống dưới cùng bên trái → Chọn **Developer settings**.
3. Chọn **Personal access tokens** → **Tokens (classic)**.
4. Click **Generate new token** → Chọn **Generate new token (classic)**.
5. Cấu hình:
   * **Note:** Đặt tên gợi nhớ (ví dụ: `mac-antigravity-sync`).
   * **Expiration:** Chọn thời gian hết hạn (nên chọn *No expiration* nếu dùng cá nhân lâu dài).
   * **Scopes:** Tích chọn quyền **`repo`** (Full control of private repositories).
6. Click **Generate token** ở dưới cùng.
7. **Sao chép và lưu lại mã Token được hiển thị** (Mã này chỉ xuất hiện một lần duy nhất).

### Bước 2.3: Khởi tạo Git local và Push dữ liệu lên GitHub
Mở Terminal trên máy hiện tại và chạy các lệnh sau:

```bash
# 1. Di chuyển vào thư mục cấu hình .gemini
cd ~/.gemini

# 2. Khởi tạo Git repository
git init

# 3. Tạo file .gitignore để tránh đẩy file rác của hệ thống lên GitHub
cat > .gitignore << 'EOF'
.DS_Store
*.log
.uv_cache/
.venv/
EOF

# 4. Thêm các file rules, skills và file gitignore vào danh sách theo dõi
git add AGENTS.md config/skills/ .gitignore

# 5. Commit dữ liệu
git commit -m "Initial commit of global agents rules and skills"

# 6. Liên kết với kho lưu trữ GitHub bằng giao thức HTTPS
# (Thay "trihx" bằng tên tài khoản GitHub của anh nếu có thay đổi)
git remote add origin https://github.com/trihx/antigravity_skill.git
git branch -M main

# 7. Đẩy mã nguồn lên nhánh chính
git push -u origin main
```

*Khi Terminal yêu cầu xác thực:*
* **Username:** Nhập tên tài khoản GitHub của anh (`trihx`).
* **Password:** Nhập **Personal Access Token (PAT)** đã copy ở Bước 2.2 (không dùng mật khẩu tài khoản thường).

---

## 💻 3. Các bước Đồng bộ trên Máy tính mới

Để nạp toàn bộ cấu hình đã lưu trên GitHub sang máy tính mới:

### Bước 3.1: Clone cấu hình từ GitHub về máy mới
Mở Terminal trên máy tính mới và chạy các lệnh:

```bash
# 1. Di chuyển về thư mục Home của User
cd ~

# 2. Tải cấu hình từ GitHub về dưới dạng thư mục ẩn .gemini
git clone https://github.com/trihx/antigravity_skill.git .gemini
```
*Nhập Username GitHub và PAT đã tạo khi Git yêu cầu.*

### Bước 3.2: Xác minh
Sau khi clone thành công, anh chỉ cần khởi động lại session chat của Antigravity. IDE sẽ tự động quét thư mục `~/.gemini` mới được tải về và áp dụng ngay lập tức các global rules cũng như global skills của anh.

---

## 🔄 4. Quy trình làm việc hàng ngày (Giữa các máy)

Khi anh chỉnh sửa rules hoặc thêm skill mới trên một máy tính và muốn cập nhật sang các máy khác:

### Tại máy có thay đổi (ví dụ: máy A):
Chạy lệnh trong Terminal:
```bash
cd ~/.gemini
git add -A
git commit -m "Update agent rules or add new skills"
git push origin main
```

### Tại máy khác cần đồng bộ (ví dụ: máy B):
Chạy lệnh trong Terminal để cập nhật bản mới nhất từ GitHub về:
```bash
cd ~/.gemini
git pull origin main
```

---

## ⚙️ 5. Hướng dẫn Cập nhật System Instructions (Quy tắc Global)

Khi anh muốn thay đổi phong cách giao tiếp, cập nhật thông tin nghề nghiệp, hoặc bổ sung các quy tắc lập trình global mới cho Antigravity:

### Bước 5.1: Chỉnh sửa file rules cá nhân
1. Trên máy tính của anh, mở file **`~/.gemini/AGENTS.md`** bằng bất kỳ trình soạn thảo mã nguồn nào (VS Code, Cursor, hoặc ngay trong Antigravity IDE).
2. Chỉnh sửa, thêm bớt nội dung rules hoặc thông tin background cá nhân của anh.
3. Lưu file lại. Cấu hình mới sẽ có hiệu lực ngay trong session làm việc tiếp theo của Antigravity.

### Bước 5.2: Đẩy cập nhật lên GitHub
Để đồng bộ các thay đổi rules này sang máy tính khác, anh chạy lệnh sau tại Terminal của máy vừa sửa:
```bash
cd ~/.gemini
git add AGENTS.md
git commit -m "Update system instructions and global rules"
git push origin main
```

### Bước 5.3: Cập nhật rules mới trên máy khác
Trên máy tính còn lại, anh chỉ cần kéo bản cập nhật mới nhất từ GitHub về:
```bash
cd ~/.gemini
git pull origin main
```

---

## 💡 6. Quản lý & Đồng bộ Skill từ Cộng đồng (ví dụ: agentic-awesome-skills)

Khi anh muốn sử dụng kho dữ liệu khỏng lồ gồm hơn 1.960+ skill từ repo cộng đồng [sickn33/agentic-awesome-skills](https://github.com/sickn33/agentic-awesome-skills) trên nhiều máy tính khác nhau:

### ⚠️ Khuyến nghị quan trọng về Git
**KHÔNG commit toàn bộ 1.960+ skill lên repository GitHub cá nhân (`antigravity_skill`).**
* Lý do: Số lượng file quá lớn sẽ làm Git bị quá tải, khiến quá trình `push/pull` cực kỳ chậm và dễ gây ra xung đột (conflict) dữ liệu khi tác giả cập nhật bản mới.
* Giải pháp tối ưu: Tách biệt hoàn toàn. Chỉ đồng bộ cấu hình/rules cá nhân qua GitHub, còn các skill cộng đồng sẽ được cài đặt và cập nhật độc lập trên từng máy qua công cụ `npx` chính thức của tác giả.

### Bước 6.1: Cài đặt và Cập nhật bản mới nhất từ Cộng đồng
Trên mỗi máy tính (máy A, máy B, máy C...), chạy lệnh sau trong Terminal để tải hoặc cập nhật phiên bản stable mới nhất của 1.960+ skill từ tác giả:
```bash
npx agentic-awesome-skills --force
```
*(Lệnh này sẽ tự động tải các file skill về thư mục local `~/.agents/skills` trên máy).*

### Bước 6.2: Thiết lập liên kết ảo (Symbolic Link) với Antigravity
Để Antigravity tự động nhận diện toàn bộ các skill này mà không cần nhân đôi dung lượng ổ cứng, anh chạy các lệnh sau:
```bash
# 1. Xóa thư mục skills trống ở .gemini (nếu có)
rm -rf ~/.gemini/config/skills

# 2. Tạo liên kết ảo trỏ đến thư mục chứa 1.960+ skill vừa tải ở Bước 6.1
ln -s ~/.agents/skills ~/.gemini/config/skills
```

### Bước 6.3: Quy trình cập nhật khi có phiên bản mới của tác giả
Khi tác giả cập nhật các skill mới trên GitHub của họ, anh chỉ cần chạy lại lệnh sau trên máy bất kỳ để cập nhật bản mới nhất:
```bash
npx agentic-awesome-skills --force
```
Các liên kết ảo đã thiết lập ở Bước 6.2 sẽ tự động nhận diện phiên bản mới mà không cần bất kỳ cấu hình hay Git commit nào khác. Cấu hình cá nhân của anh vẫn hoàn toàn sạch sẽ và an toàn.

---

## 🚀 7. Quy trình Tạo Dự Án Mới (Automation-First Bootstrap)

Khi anh bắt đầu một dự án mới, chỉ cần nói với Antigravity: **"Khởi tạo dự án mới"** hoặc **"Bootstrap dự án"**. Agent sẽ tự động:

### Bước 7.1: Agent hỏi thông tin cơ bản
- Tech stack (Python/JS/cả hai?)
- Mục đích dự án (ML, Web app, API, Dashboard?)
- Database (PostgreSQL, MongoDB, SQLite?)

### Bước 7.2: Agent tự động tạo scaffolding

```
<project-name>/
├── .agents/
│   ├── AGENTS.md                    ← L0 Pointer (~40 dòng, inject tự động)
│   ├── workflows/                   ← lint.md, refactor.md (tùy dự án)
│   └── skills/<project-skill>/
│       ├── SKILL.md                 ← Router + Golden Rules + Gotchas
│       └── guides/                  ← Hướng dẫn chuyên sâu (Lazy-Load)
├── docs/
│   ├── MEMORY_HOT.md                ← L1 Hot Memory (~20 dòng ban đầu)
│   ├── LESSONS_LEARNED.md           ← L2 Bug Patterns (bảng rỗng)
│   └── DECISIONS_LOG.md             ← L3 Archive (rỗng)
├── scripts/
│   └── utilities/
│       └── update_memory.py         ← Script tự động dọn dẹp bộ nhớ
├── src/                             ← Source code chính
├── tests/                           ← Unit & Integration Tests
├── .env                             ← Environment variables (secrets)
├── .gitignore                       ← Chuẩn cho tech stack
├── .pre-commit-config.yaml          ← Auto lint/format trước mỗi commit
├── Makefile                         ← Automation hub (install, dev, test, check, clean)
└── pyproject.toml                   ← Dependencies (uv)
```

### Bước 7.3: Agent kích hoạt automation
```bash
# Agent tự động chạy:
pre-commit install         # Kích hoạt lint hooks
make install               # Cài đặt dependencies
```

### Bước 7.4: Agent đề xuất automation bổ sung
Agent sẽ phân tích dự án và đề xuất thêm (anh duyệt rồi mới tạo):
- CI/CD pipeline (khi sẵn sàng deploy)
- Dockerfile (khi cần container hóa)
- Keep-alive workflow (khi dùng Render/Railway free tier)
- Experiment tracking (khi là dự án ML)

### 💡 Lưu ý quan trọng
- Anh **KHÔNG cần** tạo thủ công bất kỳ file nào trong cấu trúc trên.
- Chỉ cần nói với Agent tech stack → Agent tự tạo toàn bộ → Anh review → Done.
- Cấu trúc này đã được quy định trong Global Rules (`~/.gemini/AGENTS.md` §7 + §8), nên Agent trên **mọi máy tính** của anh đều tuân thủ nhất quán.


