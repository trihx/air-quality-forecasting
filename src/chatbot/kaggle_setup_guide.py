"""Kaggle Ollama Setup Guide for PM2.5 AI Assistant.

Provides step-by-step instructions for setting up a free 32GB VRAM
LLM server on Kaggle with Ollama + Cloudflare Tunnel.
"""

import streamlit as st

# Kaggle notebook cell scripts - users copy-paste these into Kaggle
KAGGLE_CELL_1_SETUP = """# Cell 1: Cài đặt dependencies
!sudo apt-get install -y zstd
!curl -fsSL https://ollama.com/install.sh | sh
print("✅ Ollama đã cài đặt xong!")
"""

KAGGLE_CELL_2_START = """# Cell 2: Khởi động Ollama server
import subprocess, time
ollama_proc = subprocess.Popen(
    ["ollama", "serve"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)
time.sleep(5)
print("✅ Ollama server đang chạy!")
"""

KAGGLE_CELL_3_MODEL = """# Cell 3: Tải model (chọn 1 trong các lệnh dưới)
# Khuyến nghị cho demo (nhanh, nhẹ, Tiếng Việt tốt):
!ollama pull qwen3:4b

# Hoặc model mạnh hơn (cần thêm VRAM):
# !ollama pull qwen3:8b
# !ollama run hf.co/JonathanColetti/Qwen3.8-27B-Uncensored-GGUF:Q4_K_M
"""

KAGGLE_CELL_4_TUNNEL = """# Cell 4: Tạo Cloudflare Tunnel (public URL)
import subprocess, time, re

!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
!sudo dpkg -i cloudflared-linux-amd64.deb

tunnel_proc = subprocess.Popen(
    ["cloudflared", "tunnel", "--url", "http://127.0.0.1:11434",
     "--http-host-header", "localhost:11434"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

# Đọc URL tunnel từ output
time.sleep(10)
import select
while True:
    if select.select([tunnel_proc.stderr], [], [], 1)[0]:
        line = tunnel_proc.stderr.readline().decode()
        if "trycloudflare.com" in line:
            url = re.search(r"https://[\\w-]+\\.trycloudflare\\.com", line)
            if url:
                print(f"")
                print(f"🔗 TUNNEL URL: {url.group()}")
                print(f"")
                print(f"👉 Copy URL trên và paste vào Dashboard > Trợ Lý AI > Kaggle Ollama")
                break
    else:
        break

print("\\n✅ Server sẵn sàng! Giữ notebook này mở để duy trì kết nối.")
"""

KAGGLE_CELL_5_KEEPALIVE = """# Cell 5: Giữ session sống (chạy cuối cùng)
import time
print("⏳ Đang giữ session... (Ctrl+C để dừng)")
while True:
    time.sleep(60)
    print(".", end="", flush=True)
"""


def render_kaggle_setup_guide():
    """Render step-by-step Kaggle setup guide in an expander."""
    with st.expander("📋 Hướng dẫn setup Kaggle Ollama (5 bước)", expanded=False):
        st.markdown(
            """
        **Yêu cầu:** Tài khoản [Kaggle](https://kaggle.com) miễn phí

        **Cấu hình notebook:**
        - Accelerator: `GPU T4 x2` (32GB VRAM)
        - Internet: `ON` (bật kết nối internet)
        """
        )

        st.markdown("**Bước 1:** Cài đặt Ollama")
        st.code(KAGGLE_CELL_1_SETUP, language="python")

        st.markdown("**Bước 2:** Khởi động server")
        st.code(KAGGLE_CELL_2_START, language="python")

        st.markdown("**Bước 3:** Tải model AI")
        st.code(KAGGLE_CELL_3_MODEL, language="python")

        st.markdown("**Bước 4:** Tạo tunnel (lấy URL)")
        st.code(KAGGLE_CELL_4_TUNNEL, language="python")

        st.markdown("**Bước 5:** Giữ session sống")
        st.code(KAGGLE_CELL_5_KEEPALIVE, language="python")

        st.info("💡 **Lưu ý:** Kaggle cho phép 30h GPU/tuần miễn phí. URL tunnel sẽ thay đổi mỗi lần restart notebook.")
