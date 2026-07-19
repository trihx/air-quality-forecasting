import re
from pathlib import Path
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"

MEMORY_HOT_FILE = DOCS_DIR / "MEMORY_HOT.md"
LESSONS_FILE = DOCS_DIR / "LESSONS_LEARNED.md"
DECISIONS_FILE = DOCS_DIR / "DECISIONS_LOG.md"

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

def update_memory():
    print("🔄 Đang chạy quy trình tự động tối ưu hóa bộ nhớ (L1 -> L3)...")
    
    # 1. Chuẩn hóa mã hóa toàn bộ các file tài liệu
    clean_and_format_file(MEMORY_HOT_FILE)
    clean_and_format_file(LESSONS_FILE)
    clean_and_format_file(DECISIONS_FILE)
    
    # 2. Đọc nội dung hiện tại
    hot_content = MEMORY_HOT_FILE.read_text(encoding="utf-8") if MEMORY_HOT_FILE.exists() else ""
    lessons_content = LESSONS_FILE.read_text(encoding="utf-8") if LESSONS_FILE.exists() else ""
    decisions_content = DECISIONS_FILE.read_text(encoding="utf-8") if DECISIONS_FILE.exists() else ""
    
    # Phân tích số dòng để tối ưu
    hot_lines = hot_content.split("\n")
    lessons_lines = lessons_content.split("\n")
    
    print(f"   * L1 (MEMORY_HOT.md): {len(hot_lines)} dòng")
    print(f"   * L2 (LESSONS_LEARNED.md): {len(lessons_lines)} dòng")
    
    # 3. Tự động kiểm tra Git commit để cập nhật Lessons Learned
    commits = get_git_commits()
    new_lessons = []
    
    for commit in commits:
        # Định dạng commit dạng: "hash fix/feat: nội dung" hoặc "hash [category] nội dung"
        match = re.match(r"^[a-f0-9]+\s+(fix|feat|refactor|deploy|config):\s*(.*)", commit, re.IGNORECASE)
        if match:
            cat = match.group(1).upper()
            desc = match.group(2)
            # Tránh lặp bài học đã có
            if desc not in lessons_content:
                from datetime import datetime
                today = datetime.today().strftime('%Y-%m-%d')
                new_lessons.append(f"| {today} | `{cat}` | {desc} | Git Commit |")
    
    if new_lessons:
        print(f"   * Phát hiện {len(new_lessons)} bài học mới từ Git commit.")
        # Thêm bài học mới vào bảng trong LESSONS_LEARNED.md
        if "Nguyên tắc" in lessons_content:
            parts = lessons_content.split("## 2. Quy tắc cốt lõi")
            table_part = parts[0]
            for lesson in new_lessons:
                table_part += lesson + "\n"
            updated_lessons = table_part + "\n## 2. Quy tắc cốt lõi" + parts[1]
            LESSONS_FILE.write_text(updated_lessons, encoding="utf-8")
            print("   * Đã cập nhật LESSONS_LEARNED.md thành công.")

    # 4. Nén bộ nhớ nếu vượt quá giới hạn (Compression Protocol)
    # Nếu MEMORY_HOT.md > 80 dòng, chuyển bớt phần thông tin cũ sang DECISIONS_LOG.md
    if len(hot_lines) > 80:
        print("   * Cảnh báo: L1 (MEMORY_HOT.md) vượt quá 80 dòng. Tiến hành nén...")
        # Giữ lại các mục chính, chuyển ghi chú phụ sang Decisions Log
        # Ở đây chỉ demo log đơn giản
        decisions_content += f"\n\n## [Archived Hot Memory] - Tối ưu hóa ngày {datetime.today().strftime('%Y-%m-%d')}\n"
        decisions_content += "\n".join(hot_lines[15:]) # Lưu trữ phần sau dòng 15
        DECISIONS_FILE.write_text(decisions_content, encoding="utf-8")
        
        # Cắt ngắn MEMORY_HOT.md
        truncated_hot = "\n".join(hot_lines[:15]) + "\n\n*(Một số ghi chú cũ đã được chuyển lưu trữ sang DECISIONS_LOG.md)*"
        MEMORY_HOT_FILE.write_text(truncated_hot, encoding="utf-8")
        print("   * Đã nén MEMORY_HOT.md thành công.")
        
    print("✅ Hoàn thành tối ưu hóa bộ nhớ!")

if __name__ == "__main__":
    update_memory()
