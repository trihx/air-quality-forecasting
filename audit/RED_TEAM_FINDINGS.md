# 🔴 BÁO CÁO RÀ SOÁT LỖ HỔNG BẢO MẬT (RED TEAM AUDIT FINDINGS)
> **Dự án:** PM2.5 Multi-Resolution Forecasting Dashboard  
> **Tác tử thực thi:** Red Team (DeepSeek-R1 / Reasoning High)  
> **Thời gian:** 2026-10-03  
> **Tiêu chuẩn kiểm định:** OWASP Top 10, CWE / SANS Top 25, Nghị định 327-333/2026/NĐ-CP & Luật An ninh mạng Việt Nam.

---

## 🎯 1. Bảng Tổng Hợp Lỗ Hổng Phát Hiện (Vulnerability Summary)

| Mã Lỗ Hổng | Tên Lỗ Hổng / Vấn Đề An Toàn | Mức Độ | CWE ID | Vị Trí Phát Hiện | Tác Động Rủi Ro |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **VULN-01** | Lộ Tunnel URL Cloudflare dạng Plaintext trên Giao Diện | **HIGH** | CWE-312 / CWE-319 | `src/chatbot/chat_page.py` | Người lạ nhìn trộm màn hình (shoulder surfing) hoặc khi trình chiếu bảo vệ luận văn sẽ thấy toàn bộ URL Cloudflare, chiếm quyền GPU Kaggle. |
| **VULN-02** | Tiết Lộ Khóa PUK & Mã PIN Trên Màn Hình Tự Vệ Lockout | **CRITICAL** | CWE-200 / CWE-798 | `src/chatbot/pin_security.py` & `chat_page.py` | Màn hình hướng dẫn và placeholder hiển thị trực tiếp `MASTER-190034-UNLOCK` và `190034`, vô hiệu hóa 100% cơ chế khóa 5 lần thử. |
| **VULN-03** | Rò Rỉ Secret Unlock Token Trong URL Bar & Browser History | **MEDIUM** | CWE-598 / CWE-200 | `src/chatbot/pin_security.py` | Query parameter `?unlock_pin=190034` hoặc `?puk=...` tồn tại mãi mãi trên thanh URL, bị lưu vào lịch sử duyệt web và rò rỉ qua header `Referer`. |
| **VULN-04** | Regex Redaction Thiếu Sót (Lọt Groq API Key & Cloudflare URL) | **HIGH** | CWE-532 / CWE-209 | `src/chatbot/provider_config.py` | Hàm `sanitize_error_message` chỉ lọc `sk-` và `AIza`, bỏ sót hoàn toàn Groq Key (`gsk_...`), Cloudflare URLs và chuỗi kết nối Database trong logs/lỗi. |
| **VULN-05** | Endpoint Info-Cards Cho Phép Truy Cập & Ghi Đè Ciphertext Hệ Thống | **HIGH** | CWE-284 / CWE-862 | `src/api/routers/content.py` | Gọi `GET /content/info-cards?page=_system` hoặc `PUT /content/info-cards/_system_ai_credentials_enc` cho phép kẻ xấu đọc và phá hủy dữ liệu mã hóa. |
| **VULN-06** | Rò Rỉ Chuỗi Kết Nối / Topology Cơ Sở Dữ Liệu Qua `/health` | **MEDIUM** | CWE-209 | `src/api/main.py` | Khi DB lỗi, endpoint public `/health` in trực tiếp `f"error: {e}"` chứa chuỗi kết nối PostgreSQL (user, host, port). |
| **VULN-07** | Thiếu Hàng Rào Chống Trích Xuất Khóa (Jailbreak / Prompt Extraction) | **MEDIUM** | CWE-1427 | `src/chatbot/llm_client.py` & `guardrails.py` | Chưa có quy tắc Security Invariant trong System Prompt và Guardrails để chặn người dùng truy vấn hỏi xin API Key hoặc link Kaggle nội bộ. |

---

## 🔍 2. Chi Tiết Bằng Chứng Khai Thác (PoC & Evidence)

### 📌 VULN-01: Lộ Tunnel URL Cloudflare trong `st.text_input`
- **File:** `src/chatbot/chat_page.py` (Dòng 243 & 377).
- **Hiện trạng:**
  ```python
  # Dòng 243
  kaggle_url = st.text_input("Tunnel URL", key="input_kaggle_ollama_url", placeholder="https://xxx.trycloudflare.com", value=...)
  # Dòng 377
  k_url = st.text_input("Cloudflare Tunnel URL", key="main_kaggle_url", placeholder="https://xxx.trycloudflare.com", value=...)
  ```
- **Hành vi khai thác:** Trường nhập URL không có tham số `type="password"`. Khi anh Trí dán URL Cloudflare vào hoặc khi load lại từ bộ nhớ đã lưu, URL hiện nguyên văn `https://xxxx-xxxx-xxxx.trycloudflare.com`. Bất kỳ ai nhìn vào màn hình (hoặc khi chia sẻ Zoom/Google Meet) đều có thể copy URL này và gửi lệnh trái phép vào Ollama server đang chạy trên Kaggle GPU T4.

### 📌 VULN-02: Tiết Lộ Khóa PUK & Mã PIN Trên Màn Hình Tự Vệ Lockout
- **File:** `src/chatbot/pin_security.py` (Dòng 206 & 223), `src/chatbot/chat_page.py` (Dòng 697).
- **Hiện trạng:**
  ```python
  # pin_security.py:206
  placeholder="Nhập mã PUK (ví dụ: MASTER-190034-UNLOCK)..."
  # pin_security.py:223
  `- Bạn có thể mở khóa ngay lập tức bằng cách thêm tham số URL bí mật vào bookmark trình duyệt:
    ?unlock_pin=190034 hoặc ?puk=MASTER-190034-UNLOCK`
  # chat_page.py:697
  <span><strong>Xác thực PIN: Hợp lệ (190034)</strong> • Bảo vệ Brute-force & DoS: Đang hoạt động</span>
  ```
- **Hành vi khai thác:** Khi kẻ tấn công dò sai PIN 5 lần và bị màn hình đỏ khóa lại, giao diện lại in sẵn mã PUK mẫu `MASTER-190034-UNLOCK` ngay trong ô placeholder và hướng dẫn mở khóa bằng `?unlock_pin=190034`! Kẻ tấn công chỉ cần copy dán lại là mở khóa thành công trong 1 giây!

### 📌 VULN-03: Rò Rỉ Secret Token Trên Thanh URL
- **File:** `src/chatbot/pin_security.py` (Dòng 121-140).
- **Hiện trạng:** `check_and_apply_query_param_unlock` kiểm tra `query_params.get("unlock_pin")` và mở khóa phiên làm việc nhưng KHÔNG xóa query parameter khỏi thanh địa chỉ của trình duyệt.
- **Hành vi khai thác:** URL vẫn giữ nguyên `?unlock_pin=190034`. Khi người dùng click vào bất kỳ link ra bên ngoài (hoặc copy link gửi cho người khác), mã PIN bí mật bị rò rỉ qua HTTP header `Referer` và lịch sử trình duyệt.

### 📌 VULN-04: Bỏ Sót Lọc Thông Tin Nhạy Cảm Trong `sanitize_error_message`
- **File:** `src/chatbot/provider_config.py` (Dòng 348-370).
- **Hiện trạng:** Regex chỉ quét `sk-[A-Za-z0-9_\-]{8,}` và `AIza[A-Za-z0-9_\-]{10,}`.
- **Hành vi khai thác:**
  - Khóa Groq API có tiền tố `gsk_` $\to$ BỊ BỎ QUA, in nguyên văn trong error message!
  - Cloudflare Tunnel URL `https://*.trycloudflare.com` $\to$ BỊ BỎ QUA, in nguyên văn khi kết nối thất bại!
  - Connection string PostgreSQL `postgresql://user:pass@host/db` $\to$ BỊ BỎ QUA!

### 📌 VULN-05: Lỗ Hổng BOLA & Tampering Tại API Content Info-Cards
- **File:** `src/api/routers/content.py` (Dòng 71-120).
- **Hiện trạng:** Thẻ `_system_ai_credentials_enc` lưu thông tin nhạy cảm của hệ thống nhưng endpoint `GET /content/info-cards?page=_system` và `PUT /content/info-cards/{card_key}` không có cơ chế chặn truy cập thẻ `_system`.
- **Hành vi khai thác:** Kẻ tấn công có thể gửi HTTP `GET` để tải ciphertext về phân tích ngoại tuyến (offline attack) hoặc gửi HTTP `PUT` với chuỗi rác để phá hỏng toàn bộ cấu hình AI Assistant của hệ thống.

### 📌 VULN-06: Rò Rỉ Thông Tin Lỗi DB Qua Endpoint `/health`
- **File:** `src/api/main.py` (Dòng 145).
- **Hiện trạng:** `db_status = f"error: {e}"` trên endpoint public `/health`.
- **Hành vi khai thác:** Khi kết nối DB gặp sự cố, chuỗi exception của psycopg2 thường chứa tên user, database name, IP nội bộ của cloud server.

### 📌 VULN-07: Thiếu Hàng Rào Chặn Prompt Injection Trích Xuất Khóa
- **File:** `src/chatbot/llm_client.py` & `src/chatbot/guardrails.py`.
- **Hiện trạng:** Chưa có quy định cứng trong System Prompt và Guardrail về việc cấm tuyệt đối tiết lộ secret khi bị hỏi trực tiếp.

---

## 🛡️ 3. Yêu Cầu Chuyển Giao Cho Blue Team (Actionable Remediation Orders)
1. Che giấu toàn bộ Cloudflare Tunnel URL bằng `type="password"`.
2. Triệt tiêu hoàn toàn các chuỗi secret (`190034`, `MASTER-190034-UNLOCK`) khỏi giao diện, placeholder và text hướng dẫn.
3. Tự động dọn sạch (sanitize) tham số URL ngay sau khi mở khóa thành công.
4. Nâng cấp bộ lọc `sanitize_error_message`: hỗ trợ `gsk_`, `https://*.trycloudflare.com`, `sk-ant-`, PostgreSQL connection strings.
5. Khóa chặt các thẻ `_system` trong API FastAPI router.
6. Chuẩn hóa mã phản hồi lỗi DB trên endpoint `/health` thành chuỗi ẩn danh.
7. Bổ sung Security Invariant vào `SYSTEM_PROMPT` và kiểm tra Guardrail chống trích xuất secret.
