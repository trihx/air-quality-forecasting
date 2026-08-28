import re
from pathlib import Path
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MEMORY_DIR = PROJECT_ROOT / ".agents" / "memory"

MEMORY_HOT_FILE = MEMORY_DIR / "MEMORY_HOT.md"
LESSONS_FILE = MEMORY_DIR / "LESSONS_LEARNED.md"
DECISIONS_FILE = MEMORY_DIR / "DECISIONS_LOG.md"
NEXT_ACTIONS_FILE = MEMORY_DIR / "NEXT_ACTIONS.md"

def get_git_commits():
    """Lấy 5 commit gần nhất để phân tích."""
    try:
        result = subprocess.run(
            ["git", "log", "-n", "5", "--oneline"],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip().split("\n")
    except Exception:
        return []

def clean_and_format_file(file_path):
    """Đảm bảo file được lưu dưới dạng UTF-8 chuẩn và loại bỏ các ký tự lỗi."""
    if not file_path.exists():
        return ""
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        # Chuẩn hóa xuống dòng
        content = content.replace("\r\n", "\n")
        file_path.write_text(content, encoding="utf-8")
        return content
    except Exception as e:
        print(f"Lỗi khi chuẩn hóa {file_path.name}: {e}")
        return ""

from datetime import datetime

def update_memory():
    print("🔄 Đang chạy quy trình tự động tối ưu hóa bộ nhớ (L1 -> L3, L★)...")
    
    # 1. Chuẩn hóa mã hóa toàn bộ các file tài liệu
    clean_and_format_file(MEMORY_HOT_FILE)
    clean_and_format_file(LESSONS_FILE)
    clean_and_format_file(DECISIONS_FILE)
    clean_and_format_file(NEXT_ACTIONS_FILE)
    
    # 2. Đọc nội dung hiện tại
    hot_content = MEMORY_HOT_FILE.read_text(encoding="utf-8") if MEMORY_HOT_FILE.exists() else ""
    lessons_content = LESSONS_FILE.read_text(encoding="utf-8") if LESSONS_FILE.exists() else ""
    decisions_content = DECISIONS_FILE.read_text(encoding="utf-8") if DECISIONS_FILE.exists() else ""
    next_actions_content = NEXT_ACTIONS_FILE.read_text(encoding="utf-8") if NEXT_ACTIONS_FILE.exists() else ""
    
    # Phân tích số dòng để tối ưu
    hot_lines = [l for l in hot_content.split("\n") if l.strip()]
    lessons_lines = [l for l in lessons_content.split("\n") if l.strip()]
    next_lines = [l for l in next_actions_content.split("\n") if l.strip()]
    
    print(f"   * L1 (MEMORY_HOT.md): {len(hot_lines)} dòng (max: 80)")
    print(f"   * L2 (LESSONS_LEARNED.md): {len(lessons_lines)} dòng (max: 100)")
    print(f"   * L★ (NEXT_ACTIONS.md): {len(next_lines)} dòng (max: 30)")
    
    # 3. Tự động kiểm tra Git commit để cập nhật Lessons Learned
    commits = get_git_commits()
    new_lessons = []
    
    for commit in commits:
        match = re.match(r"^[a-f0-9]+\s+(fix|feat|refactor|deploy|config):\s*(.*)", commit, re.IGNORECASE)
        if match:
            cat = match.group(1).upper()
            desc = match.group(2)
            if desc not in lessons_content:
                today = datetime.today().strftime('%Y-%m-%d')
                new_lessons.append(f"| {today} | `{cat}` | {desc} | Git Commit |")
    
    if new_lessons:
        print(f"   * Phát hiện {len(new_lessons)} bài học mới từ Git commit.")
        if "Nguyên tắc" in lessons_content or "## 2. Quy tắc" in lessons_content:
            parts = re.split(r"(## 2\. Quy tắc.*)", lessons_content, maxsplit=1)
            table_part = parts[0]
            for lesson in new_lessons:
                table_part += lesson + "\n"
            updated_lessons = table_part + parts[1] if len(parts) > 1 else table_part
            LESSONS_FILE.write_text(updated_lessons, encoding="utf-8")
            print("   * Đã cập nhật LESSONS_LEARNED.md thành công.")

    # 4. Nén bộ nhớ nếu vượt quá giới hạn (Compression Protocol)
    raw_hot_lines = hot_content.split("\n")
    if len(raw_hot_lines) > 80:
        print("   * Cảnh báo: L1 (MEMORY_HOT.md) vượt quá 80 dòng. Tiến hành nén...")
        decisions_content += f"\n\n## [Archived Hot Memory] - Tối ưu hóa ngày {datetime.today().strftime('%Y-%m-%d')}\n"
        decisions_content += "\n".join(raw_hot_lines[20:])
        DECISIONS_FILE.write_text(decisions_content, encoding="utf-8")
        
        truncated_hot = "\n".join(raw_hot_lines[:20]) + "\n\n*(Ghi chú cũ hơn đã được chuyển lưu trữ sang DECISIONS_LOG.md)*"
        MEMORY_HOT_FILE.write_text(truncated_hot, encoding="utf-8")
        print("   * Đã nén MEMORY_HOT.md thành công.")
        
    print("✅ Hoàn thành tối ưu hóa bộ nhớ phân tầng!")

if __name__ == "__main__":
    update_memory()
