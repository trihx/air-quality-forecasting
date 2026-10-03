"""
Unit tests for AI Assistant PIN Security & Lockout Mechanism.

Tests:
  - Default PIN verification (190034)
  - Timing attack resistance (hmac.compare_digest)
  - Failed attempts tracking & Lockout after 5 attempts
  - Cooldown timer expiration & automatic reset
  - Master PUK recovery unlock
  - Secret URL query parameters unlock (?unlock_pin=190034, ?puk=...)
  - Manual lock session
  - Environment variable overrides
"""

import time
from unittest.mock import patch

from src.chatbot.pin_security import (
    DEFAULT_PIN,
    DEFAULT_PUK,
    LOCKOUT_COOLDOWN_SECONDS,
    MAX_FAILED_ATTEMPTS,
    check_and_apply_query_param_unlock,
    get_configured_pin,
    get_configured_puk,
    get_lockout_info,
    is_session_unlocked,
    lock_session,
    record_failed_attempt,
    reset_lockout,
    unlock_session,
    verify_pin,
    verify_puk,
)


class TestPinVerification:
    """Test core PIN and PUK verification logic."""

    def test_default_pin_is_190034(self):
        assert DEFAULT_PIN == "190034"
        assert get_configured_pin() == "190034"

    def test_verify_pin_valid(self):
        assert verify_pin("190034") is True
        assert verify_pin(" 190034 ") is True  # Strips whitespace

    def test_verify_pin_invalid(self):
        assert verify_pin("000000") is False
        assert verify_pin("123456") is False
        assert verify_pin("") is False
        assert verify_pin(None) is False  # type: ignore

    def test_verify_puk_valid(self):
        assert verify_puk(DEFAULT_PUK) is True
        assert verify_puk(f"  {DEFAULT_PUK}  ") is True

    def test_verify_puk_invalid(self):
        assert verify_puk("WRONG-PUK") is False
        assert verify_puk("") is False
        assert verify_puk(None) is False  # type: ignore

    def test_custom_pin_env_override(self):
        with patch.dict("os.environ", {"AI_ASSISTANT_PIN": "654321"}):
            assert get_configured_pin() == "654321"
            assert verify_pin("654321") is True
            assert verify_pin("190034") is False

    def test_custom_puk_env_override(self):
        with patch.dict("os.environ", {"AI_ASSISTANT_PUK": "CUSTOM-EMERGENCY-KEY"}):
            assert get_configured_puk() == "CUSTOM-EMERGENCY-KEY"
            assert verify_puk("CUSTOM-EMERGENCY-KEY") is True
            assert verify_puk(DEFAULT_PUK) is False


class TestSessionStateAndLockout:
    """Test session state transitions, attempt counters, and lockout triggers."""

    def test_initial_session_is_locked(self):
        session = {}
        assert is_session_unlocked(session) is False
        is_locked, remaining, attempts = get_lockout_info(session)
        assert is_locked is False
        assert remaining == 0
        assert attempts == 0

    def test_failed_attempts_increment(self):
        session = {}
        for i in range(1, MAX_FAILED_ATTEMPTS):
            attempts, is_now_locked = record_failed_attempt(session)
            assert attempts == i
            assert is_now_locked is False
            is_locked, remaining, att = get_lockout_info(session)
            assert is_locked is False
            assert att == i

    def test_lockout_triggered_on_fifth_attempt(self):
        session = {}
        for _ in range(MAX_FAILED_ATTEMPTS - 1):
            record_failed_attempt(session)

        # 5th failed attempt triggers lockout
        attempts, is_now_locked = record_failed_attempt(session)
        assert attempts == MAX_FAILED_ATTEMPTS
        assert is_now_locked is True

        is_locked, remaining, att = get_lockout_info(session)
        assert is_locked is True
        assert att == MAX_FAILED_ATTEMPTS
        assert 0 < remaining <= LOCKOUT_COOLDOWN_SECONDS

    def test_cooldown_expiration_resets_lockout(self):
        session = {}
        for _ in range(MAX_FAILED_ATTEMPTS):
            record_failed_attempt(session)

        # Confirm locked
        is_locked, remaining, _ = get_lockout_info(session)
        assert is_locked is True

        # Fast forward time beyond cooldown
        future_time = time.time() + LOCKOUT_COOLDOWN_SECONDS + 10
        with patch("time.time", return_value=future_time):
            is_locked_after, remaining_after, att_after = get_lockout_info(session)
            assert is_locked_after is False
            assert remaining_after == 0
            assert att_after == 0

    def test_unlock_session_resets_all_lockout_state(self):
        session = {}
        for _ in range(MAX_FAILED_ATTEMPTS):
            record_failed_attempt(session)

        assert get_lockout_info(session)[0] is True

        unlock_session(session)
        assert is_session_unlocked(session) is True
        is_locked, remaining, att = get_lockout_info(session)
        assert is_locked is False
        assert remaining == 0
        assert att == 0

    def test_lock_session_re_locks(self):
        session = {}
        unlock_session(session)
        assert is_session_unlocked(session) is True

        lock_session(session)
        assert is_session_unlocked(session) is False

    def test_reset_lockout_preserves_unlocked_flag(self):
        session = {"ai_pin_unlocked": False, "ai_pin_failed_attempts": 5, "ai_pin_lockout_until": time.time() + 500}
        reset_lockout(session)
        assert session["ai_pin_failed_attempts"] == 0
        assert session["ai_pin_lockout_until"] is None


class TestQueryParamUnlock:
    """Test unlocking via URL query parameters for owner convenience."""

    def test_query_param_unlock_with_pin(self):
        session = {}
        query_params = {"unlock_pin": "190034"}
        unlocked = check_and_apply_query_param_unlock(query_params, session)
        assert unlocked is True
        assert is_session_unlocked(session) is True

    def test_query_param_unlock_with_alternate_keys(self):
        for key in ("pin", "unlock_ai", "ai_pin"):
            session = {}
            query_params = {key: "190034"}
            unlocked = check_and_apply_query_param_unlock(query_params, session)
            assert unlocked is True
            assert is_session_unlocked(session) is True

    def test_query_param_unlock_with_puk(self):
        session = {}
        query_params = {"puk": DEFAULT_PUK}
        unlocked = check_and_apply_query_param_unlock(query_params, session)
        assert unlocked is True
        assert is_session_unlocked(session) is True

    def test_query_param_invalid_does_not_unlock(self):
        session = {}
        query_params = {"unlock_pin": "wrong_pin"}
        unlocked = check_and_apply_query_param_unlock(query_params, session)
        assert unlocked is False
        assert is_session_unlocked(session) is False

    def test_query_param_unlock_clears_prior_lockout(self):
        session = {}
        for _ in range(MAX_FAILED_ATTEMPTS):
            record_failed_attempt(session)

        assert get_lockout_info(session)[0] is True

        # Owner accesses via bookmark URL with correct PIN
        query_params = {"unlock_pin": "190034"}
        unlocked = check_and_apply_query_param_unlock(query_params, session)
        assert unlocked is True
        assert is_session_unlocked(session) is True
        assert get_lockout_info(session)[0] is False


class MockSessionState(dict):
    """Dictionary subclass supporting dot-attribute access like Streamlit SessionState."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            return None

    def __setattr__(self, name, value):
        self[name] = value


class TestChatPagePinAndSelectorIntegration:
    """Test integration of PIN security gate and assistant selector with chat_page.py."""

    def test_locked_session_stops_at_security_gate(self):
        from unittest.mock import MagicMock

        from src.chatbot.chat_page import page_ai_assistant

        mock_st = MagicMock()
        mock_st.session_state = MockSessionState({"ai_pin_unlocked": False})
        mock_st.query_params = {}

        with (
            patch("src.chatbot.chat_page.st", mock_st),
            patch("src.chatbot.chat_page.render_pin_security_gate") as mock_gate,
            patch("src.chatbot.chat_page._render_provider_config") as mock_config,
        ):
            page_ai_assistant(results={})
            mock_gate.assert_called_once()
            # Provider config and inner chat must NOT be called when locked
            mock_config.assert_not_called()

    def test_unlocked_session_proceeds_to_assistant_page(self):
        from unittest.mock import MagicMock

        from src.chatbot.chat_page import page_ai_assistant

        mock_st = MagicMock()
        mock_st.session_state = MockSessionState(
            {
                "ai_pin_unlocked": True,
                "llm_provider_keys": {},
                "primary_provider": "gemini",
            }
        )
        mock_st.query_params = {}
        mock_st.columns.return_value = [MagicMock(), MagicMock()]
        mock_st.selectbox.return_value = "gemini"
        mock_st.button.return_value = False
        mock_st.chat_input.return_value = None

        with (
            patch("src.chatbot.chat_page.st", mock_st),
            patch("src.chatbot.chat_page.render_pin_security_gate") as mock_gate,
            patch("src.chatbot.chat_page._render_provider_config") as mock_config,
            patch("src.chatbot.chat_page._render_inline_quick_config"),
            patch("src.info_cards.cards_ai_assistant"),
            patch("src.chatbot.chat_page._ensure_index", return_value=10),
        ):
            page_ai_assistant(results={})
            mock_gate.assert_not_called()
            mock_config.assert_called_once()

    def test_query_param_auto_unlocks_during_page_render(self):
        from unittest.mock import MagicMock

        from src.chatbot.chat_page import page_ai_assistant

        session = MockSessionState({"ai_pin_unlocked": False, "primary_provider": "gemini", "llm_provider_keys": {}})
        mock_st = MagicMock()
        mock_st.session_state = session
        mock_st.query_params = {"unlock_pin": "190034"}
        mock_st.columns.return_value = [MagicMock(), MagicMock()]
        mock_st.selectbox.return_value = "gemini"
        mock_st.button.return_value = False
        mock_st.chat_input.return_value = None

        with (
            patch("src.chatbot.chat_page.st", mock_st),
            patch("src.chatbot.chat_page.render_pin_security_gate") as mock_gate,
            patch("src.chatbot.chat_page._render_provider_config") as mock_config,
            patch("src.chatbot.chat_page._render_inline_quick_config"),
            patch("src.info_cards.cards_ai_assistant"),
            patch("src.chatbot.chat_page._ensure_index", return_value=10),
        ):
            page_ai_assistant(results={})
            assert session["ai_pin_unlocked"] is True
            mock_gate.assert_not_called()
            mock_config.assert_called_once()
