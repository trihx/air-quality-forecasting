"""
PIN Security, Denial-of-Service (DoS) Protection & Multi-Tier Unlock Mechanism.

Provides:
  - 6-digit PIN verification (Default: 190034) with timing-attack resistance (HMAC)
  - 5-attempt brute-force protection with client/session-scoped lockout (DoS-safe for public dashboards)
  - Progressive 15-minute cooldown timer with automatic expiration
  - Emergency Master Recovery PUK Key unlock
  - Secret URL Query Parameter bypass for dashboard owner (?unlock_pin=190034)
  - High-tier B2B/B2G Tech Dark Mode PIN Gate UI for Streamlit
"""

import hmac
import logging
import os
import time
from typing import Any

import streamlit as st

logger = logging.getLogger(__name__)

# ── Core Constants ──
DEFAULT_PIN: str = "190034"
DEFAULT_PUK: str = "MASTER-190034-UNLOCK"
MAX_FAILED_ATTEMPTS: int = 5
LOCKOUT_COOLDOWN_SECONDS: int = 900  # 15 minutes


def get_configured_pin() -> str:
    """Return the configured PIN from environment or default 190034."""
    return os.getenv("AI_ASSISTANT_PIN", DEFAULT_PIN).strip()


def get_configured_puk() -> str:
    """Return the master recovery PUK key from environment or default."""
    return os.getenv("AI_ASSISTANT_PUK", DEFAULT_PUK).strip()


def verify_pin(input_pin: str | None) -> bool:
    """Verify input PIN using constant-time comparison to prevent timing attacks."""
    if not input_pin:
        return False
    clean_input = str(input_pin).strip().encode("utf-8")
    expected = get_configured_pin().encode("utf-8")
    return hmac.compare_digest(clean_input, expected)


def verify_puk(input_puk: str | None) -> bool:
    """Verify emergency Master PUK key using constant-time comparison."""
    if not input_puk:
        return False
    clean_input = str(input_puk).strip().encode("utf-8")
    expected = get_configured_puk().encode("utf-8")
    return hmac.compare_digest(clean_input, expected)


def is_session_unlocked(session: Any) -> bool:
    """Check if the current session has successfully passed PIN authentication."""
    return bool(session.get("ai_pin_unlocked", False))


def get_lockout_info(session: Any) -> tuple[bool, int, int]:
    """Check if the session is currently locked out.

    Returns:
        (is_locked, remaining_seconds, failed_attempts)
    """
    failed_attempts = int(session.get("ai_pin_failed_attempts", 0))
    lockout_until = session.get("ai_pin_lockout_until")
    now = time.time()

    if lockout_until is not None:
        if now < lockout_until:
            remaining = int(lockout_until - now)
            return True, remaining, failed_attempts
        # Cooldown period has elapsed -> automatically reset lockout
        session["ai_pin_lockout_until"] = None
        session["ai_pin_failed_attempts"] = 0
        return False, 0, 0

    if failed_attempts >= MAX_FAILED_ATTEMPTS:
        session["ai_pin_lockout_until"] = now + LOCKOUT_COOLDOWN_SECONDS
        return True, LOCKOUT_COOLDOWN_SECONDS, failed_attempts

    return False, 0, failed_attempts


def record_failed_attempt(session: Any) -> tuple[int, bool]:
    """Record a failed PIN attempt.

    Returns:
        (new_attempt_count, is_now_locked)
    """
    attempts = int(session.get("ai_pin_failed_attempts", 0)) + 1
    session["ai_pin_failed_attempts"] = attempts
    if attempts >= MAX_FAILED_ATTEMPTS:
        session["ai_pin_lockout_until"] = time.time() + LOCKOUT_COOLDOWN_SECONDS
        return attempts, True
    return attempts, False


def unlock_session(session: Any) -> None:
    """Mark session as unlocked and reset all failed attempts and lockout timers."""
    session["ai_pin_unlocked"] = True
    session["ai_pin_failed_attempts"] = 0
    session["ai_pin_lockout_until"] = None


def lock_session(session: Any) -> None:
    """Explicitly lock the session (e.g. user clicks Lock button)."""
    session["ai_pin_unlocked"] = False


def reset_lockout(session: Any) -> None:
    """Reset lockout state without forcibly toggling unlocked flag."""
    session["ai_pin_failed_attempts"] = 0
    session["ai_pin_lockout_until"] = None


def check_and_apply_query_param_unlock(query_params: Any, session: Any) -> bool:
    """Check URL query parameters for valid unlock tokens (?unlock_pin=190034, ?puk=...).

    If matched, clears any lockout and unlocks the session immediately.
    """
    # Check PIN query parameters
    for key in ("unlock_pin", "pin", "unlock_ai", "ai_pin"):
        val = query_params.get(key)
        if val and verify_pin(str(val)):
            unlock_session(session)
            return True

    # Check PUK emergency query parameters
    for key in ("puk", "unlock_puk", "master_key", "master_puk"):
        val = query_params.get(key)
        if val and verify_puk(str(val)):
            unlock_session(session)
            return True

    return False


def render_pin_security_gate() -> None:
    """Render the PIN Security Gate UI in Streamlit with high-contrast B2B/B2G Tech theme.

    If locked out: displays a secure lockout screen with cooldown timer and emergency PUK recovery.
    If pending verification: displays the high-contrast B2B/B2G Tech PIN input card.
    """
    is_locked, remaining_seconds, failed_attempts = get_lockout_info(st.session_state)

    if is_locked:
        mins = remaining_seconds // 60
        secs = remaining_seconds % 60
        time_str = f"{mins:02d}:{secs:02d}"

        st.markdown(
            f"""
            <div style="
                background: #181111;
                border: 1px solid rgba(239, 68, 68, 0.5);
                border-left: 6px solid #EF4444;
                border-radius: 14px;
                padding: 2.2rem;
                margin: 1.5rem 0 2rem 0;
                text-align: center;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6), 0 0 20px rgba(239, 68, 68, 0.15);
            ">
                <div style="font-size: 3.5rem; margin-bottom: 0.5rem;">🛡️🔒</div>
                <h2 style="color: #EF4444; font-size: 1.85rem; margin-bottom: 0.6rem; font-weight: 800; letter-spacing: -0.01em;">
                    Chức Năng Trợ Lý AI Đang Tạm Khóa
                </h2>
                <p style="color: #F8FAFC; font-size: 1.05rem; max-width: 680px; margin: 0 auto 1.4rem auto; line-height: 1.65;">
                    Bạn đã nhập sai mã PIN <strong>5 lần liên tiếp</strong>. Hệ thống đã kích hoạt cơ chế tự vệ
                    chống tấn công dò mã (Brute-Force & DoS Protection) nhằm bảo vệ hạn mức API và tài nguyên GPU.
                </p>
                <div style="
                    display: inline-block;
                    background: #7F1D1D;
                    border: 1px solid #EF4444;
                    border-radius: 8px;
                    padding: 0.65rem 1.6rem;
                    color: #FFFFFF;
                    font-family: 'JetBrains Mono', monospace;
                    font-size: 1.25rem;
                    font-weight: 800;
                    margin-bottom: 0.5rem;
                    box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);
                ">
                    ⏳ Tự động mở lại sau: <span style="color: #FEF08A;">{time_str}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_left, col_right = st.columns([1, 1])
        with col_left, st.expander("🔑 Mở Khóa Cứu Hộ Bằng Mã PUK (Master Recovery Key)", expanded=True):
            st.caption(
                "Nếu bạn là quản trị viên hệ thống hoặc chủ nhân đề án và cần truy cập khẩn cấp, "
                "vui lòng nhập mã PUK bảo mật cấp cao:"
            )
            puk_input = st.text_input(
                "Mã PUK Cứu Hộ",
                type="password",
                key="puk_recovery_input_field",
                placeholder="Nhập mã PUK (ví dụ: MASTER-190034-UNLOCK)...",
            )
            if st.button(
                "🔓 Mở Khóa Khẩn Cấp Bằng PUK", key="btn_unlock_puk", type="primary", use_container_width=True
            ):
                if verify_puk(puk_input):
                    unlock_session(st.session_state)
                    st.toast("✅ Đã mở khóa khẩn cấp thành công bằng mã PUK cứu hộ!", icon="🔓")
                    st.rerun()
                else:
                    st.error("❌ Mã PUK không chính xác! Vui lòng kiểm tra lại biến môi trường hoặc tài liệu.")

        with col_right, st.expander("ℹ️ Hướng Dành Riêng Cho Quản Trị Viên (Anh Trí)", expanded=True):
            st.markdown(
                """
                    **Đặc quyền mở khóa cấp tốc:**
                    - Bạn có thể mở khóa ngay lập tức bằng cách thêm tham số URL bí mật vào bookmark trình duyệt:
                      `?unlock_pin=190034` hoặc `?puk=MASTER-190034-UNLOCK`
                    - Phiên làm việc trên các thiết bị khác hoặc của Hội đồng đánh giá hoàn toàn **không bị ảnh hưởng**
                      nhờ kiến trúc Client/Session-Scoped Lockout độc lập.
                    """
            )
            if st.button("🔄 Làm mới thời gian đếm ngược", key="btn_refresh_cooldown", use_container_width=True):
                st.rerun()
        return

    # ── Security PIN Input Card (High Contrast Dark Theme) ──
    attempts_left = MAX_FAILED_ATTEMPTS - failed_attempts

    st.markdown(
        """
        <div style="
            background: #0B1120;
            border: 1px solid rgba(0, 212, 170, 0.45);
            border-left: 6px solid #00D4AA;
            border-radius: 14px;
            padding: 1.8rem 2.2rem;
            margin: 1.2rem 0 1.8rem 0;
            box-shadow: 0 12px 35px rgba(0, 0, 0, 0.55), 0 0 25px rgba(0, 212, 170, 0.08);
        ">
            <div style="display: flex; align-items: center; gap: 1.2rem; margin-bottom: 0.8rem;">
                <div style="
                    font-size: 2.2rem;
                    background: rgba(0, 212, 170, 0.15);
                    border: 1px solid rgba(0, 212, 170, 0.5);
                    border-radius: 12px;
                    width: 58px;
                    height: 58px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: #00D4AA;
                ">🔒</div>
                <div>
                    <h2 style="color: #00D4AA; font-size: 1.65rem; margin: 0; font-weight: 800; letter-spacing: -0.01em;">
                        Bảo Mật Trợ Lý AI — Xác Thực Mã PIN
                    </h2>
                    <p style="color: #CBD5E1; font-size: 0.95rem; margin: 0.25rem 0 0 0; font-weight: 500;">
                        Cơ chế bảo vệ tài nguyên điện toán GPU &amp; hạn mức API Key mô hình ngôn ngữ lớn (LLM)
                    </p>
                </div>
            </div>
            <p style="color: #F8FAFC; font-size: 1.02rem; line-height: 1.7; margin-top: 1rem; margin-bottom: 0;">
                Chức năng Trợ Lý AI tích hợp tri thức toàn diện của Đề án ThS và kết nối trực tiếp với các dịch vụ suy luận
                (<strong style="color: #38BDF8;">Google Gemini</strong>, <strong style="color: #F97316;">Groq LPU</strong>, <strong style="color: #A855F7;">Kaggle Ollama 32GB VRAM</strong>, <strong style="color: #10B981;">OpenAI</strong>).
                Vui lòng nhập mã PIN bảo mật <span style="background: rgba(0, 212, 170, 0.2); color: #00D4AA; padding: 0.15rem 0.55rem; border-radius: 6px; font-weight: 700; font-family: monospace;">(6 số)</span> để tiếp tục.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_pin_form, col_pin_info = st.columns([1.2, 1])

    with col_pin_form:
        with st.form("form_pin_security_gate", clear_on_submit=False):
            pin_input = st.text_input(
                "Mã PIN Xác Thực (6 số):",
                type="password",
                max_chars=12,
                key="security_pin_input_field",
                placeholder="Nhập mã PIN 6 số...",
                help="Mã PIN mặc định của hệ thống được cấp cho đề án nghiên cứu.",
            )
            submit_pin = st.form_submit_button(
                "🔓 Xác Nhận & Mở Khóa Trợ Lý AI",
                type="primary",
                use_container_width=True,
            )

        if submit_pin:
            if verify_pin(pin_input):
                unlock_session(st.session_state)
                st.toast("✅ Xác thực mã PIN thành công! Chào mừng bạn đến với Trợ Lý AI.", icon="🔓")
                st.rerun()
            else:
                new_attempts, is_now_locked = record_failed_attempt(st.session_state)
                if is_now_locked:
                    st.error("❌ Bạn đã nhập sai mã PIN 5 lần! Hệ thống đã tạm khóa chức năng Trợ Lý AI.")
                else:
                    st.error(f"❌ Mã PIN không chính xác! Bạn còn lại {MAX_FAILED_ATTEMPTS - new_attempts} lần thử.")
                st.rerun()

        if failed_attempts > 0:
            st.warning(
                f"⚠️ Cảnh báo: Bạn đã nhập sai {failed_attempts}/5 lần. Còn lại {attempts_left} lần trước khi bị khóa!"
            )
        else:
            st.caption("🛡️ An toàn thông tin: Sau 5 lần nhập sai liên tiếp, hệ thống sẽ tạm khóa tự động 15 phút.")

    with col_pin_info:
        st.markdown(
            """
            <div style="
                background: #111827;
                border: 1px solid rgba(255, 255, 255, 0.15);
                border-left: 4px solid #38BDF8;
                border-radius: 12px;
                padding: 1.3rem 1.4rem;
                font-size: 0.92rem;
                color: #F1F5F9;
                line-height: 1.65;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
            ">
                <div style="display: flex; align-items: center; gap: 0.4rem; margin-bottom: 0.6rem;">
                    <span style="font-size: 1.1rem;">💡</span>
                    <strong style="color: #38BDF8; font-size: 1rem;">Lưu Ý Bảo Mật:</strong>
                </div>
                <ul style="margin: 0.2rem 0 0.2rem 1.2rem; padding: 0; color: #E2E8F0;">
                    <li style="margin-bottom: 0.4rem;">Mã PIN bảo mật giúp tránh hao hụt API quota do truy cập công khai ngoài ý muốn.</li>
                    <li style="margin-bottom: 0.4rem;">Phiên làm việc được ghi nhớ tự động trong suốt thời gian duyệt trang của bạn.</li>
                    <li>Sau khi hoàn thành tra cứu, bạn có thể chủ động bấm <strong style="color: #00D4AA;">🔒 Khóa lại</strong> bất cứ lúc nào.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
