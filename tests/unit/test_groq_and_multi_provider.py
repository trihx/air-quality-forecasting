"""
Unit tests for Groq and Multi-LLM Provider enhancements.

Verifies:
1. Groq default model is llama-3.1-8b-instant and GROQ_MODEL_RECOMMENDATIONS are defined.
2. validate_provider_connection with deep chat probe (probe success, 429 rate limit, 404 model not found, 400 bad request, 401 auth).
3. Payload sanitization: messages with empty or whitespace-only content are completely filtered out.
4. Groq max_tokens payload optimization: capped at 1024 to preserve TPM limit.
5. Error transparency in chat_stream: specific error messages are collected and displayed when providers fail.
6. Primary provider prioritization in detect_available_providers.
"""

from unittest.mock import MagicMock, patch

import pytest
from src.chatbot.llm_client import (
    _try_stream_provider,
    chat_stream,
)
from src.chatbot.provider_config import (
    GROQ_MODEL_RECOMMENDATIONS,
    PROVIDER_REGISTRY,
    LLMProvider,
    detect_available_providers,
    validate_provider_connection,
)


class TestGroqConfiguration:
    """Verify Groq default configuration and model recommendations."""

    def test_groq_default_model(self):
        """Groq default model must be openai/gpt-oss-20b."""
        groq_cfg = PROVIDER_REGISTRY.get("groq")
        assert groq_cfg is not None
        assert groq_cfg["default_model"] == "openai/gpt-oss-20b"

    def test_groq_model_recommendations_structure(self):
        """GROQ_MODEL_RECOMMENDATIONS must contain recommended models with proper metadata."""
        assert isinstance(GROQ_MODEL_RECOMMENDATIONS, list)
        assert len(GROQ_MODEL_RECOMMENDATIONS) >= 3

        names = [m["name"] for m in GROQ_MODEL_RECOMMENDATIONS]
        assert "openai/gpt-oss-20b" in names
        assert "openai/gpt-oss-120b" in names
        assert "qwen/qwen3.6-27b" in names

        # openai/gpt-oss-20b must be recommended
        rec = next(m for m in GROQ_MODEL_RECOMMENDATIONS if m["name"] == "openai/gpt-oss-20b")
        assert rec["recommended"] is True


class TestValidateProviderConnectionProbe:
    """Verify deep validation with chat probe."""

    @pytest.fixture
    def mock_provider(self):
        return LLMProvider(
            name="groq",
            display_name="Groq",
            base_url="https://api.groq.com/openai/v1",
            api_key="gsk-test-key",
            model="llama-3.1-8b-instant",
        )

    def test_probe_success(self, mock_provider):
        """When models.list and chat probe succeed, return success."""
        mock_client = MagicMock()
        model_item = MagicMock()
        model_item.id = "llama-3.1-8b-instant"
        mock_client.models.list.return_value.data = [model_item]

        mock_completion = MagicMock()
        mock_client.chat.completions.create.return_value = mock_completion

        with patch("openai.OpenAI", return_value=mock_client):
            ok, msg = validate_provider_connection(mock_provider)
            assert ok is True
            assert "Kết nối & kiểm tra Chat thành công" in msg
            assert "1 models" in msg

            # Verify chat probe was performed with lightweight settings
            mock_client.chat.completions.create.assert_called_once_with(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=2,
                timeout=5.0,
            )

    def test_probe_429_rate_limit(self, mock_provider):
        """When chat probe triggers HTTP 429 rate limit, return rate limit warning."""
        mock_client = MagicMock()
        mock_client.models.list.return_value.data = [MagicMock(id="llama-3.1-8b-instant")]
        mock_client.chat.completions.create.side_effect = Exception("429 Rate limit reached on TPM")

        with patch("openai.OpenAI", return_value=mock_client):
            ok, msg = validate_provider_connection(mock_provider)
            assert ok is False
            assert "Chạm giới hạn tốc độ" in msg or "Rate Limit" in msg
            assert "Groq" in msg

    def test_probe_404_model_not_found(self, mock_provider):
        """When model does not exist or does not support chat, return model error."""
        mock_client = MagicMock()
        mock_client.models.list.return_value.data = [MagicMock(id="non-existent-model")]
        mock_provider.model = "non-existent-model"
        mock_client.chat.completions.create.side_effect = Exception("404 Model non-existent-model not found")

        with patch("openai.OpenAI", return_value=mock_client):
            ok, msg = validate_provider_connection(mock_provider)
            assert ok is False
            assert "không tồn tại hoặc không hỗ trợ chat" in msg
            assert "non-existent-model" in msg

    def test_probe_400_bad_request(self, mock_provider):
        """When chat probe returns 400 Bad Request, return parameter error."""
        mock_client = MagicMock()
        mock_client.models.list.return_value.data = [MagicMock(id="llama-3.1-8b-instant")]
        mock_client.chat.completions.create.side_effect = Exception("400 Bad Request: invalid param")

        with patch("openai.OpenAI", return_value=mock_client):
            ok, msg = validate_provider_connection(mock_provider)
            assert ok is False
            assert "Lỗi tham số chat" in msg

    def test_probe_auto_switches_deprecated_model(self, mock_provider):
        """When user model is deprecated/missing from models.list, auto-switch to active chat model."""
        mock_client = MagicMock()
        # Server only has openai/gpt-oss-20b and audio model whisper
        mock_client.models.list.return_value.data = [
            MagicMock(id="whisper-large-v3"),
            MagicMock(id="openai/gpt-oss-20b"),
        ]
        # mock_provider still configured with deprecated model
        mock_provider.model = "llama-3.1-8b-instant"

        with patch("openai.OpenAI", return_value=mock_client):
            ok, msg = validate_provider_connection(mock_provider)
            assert ok is True
            assert "Tự động chuyển sang model hoạt động: 'openai/gpt-oss-20b'" in msg
            assert mock_provider.model == "openai/gpt-oss-20b"

            # Chat probe was executed on the auto-switched model
            mock_client.chat.completions.create.assert_called_once_with(
                model="openai/gpt-oss-20b",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=2,
                timeout=5.0,
            )


class TestPayloadSanitization:
    """Verify chat messages sanitization against empty and whitespace items."""

    def test_empty_and_whitespace_messages_filtered_out(self):
        """Messages with empty content or whitespace only must be purged before calling provider."""
        provider = LLMProvider(
            name="groq",
            display_name="Groq",
            base_url="https://api.groq.com/openai/v1",
            api_key="gsk-test",
            model="llama-3.1-8b-instant",
        )

        raw_messages = [
            {"role": "user", "content": "   "},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "  Dự án này làm gì?  "},
            {"role": "assistant", "content": None},
            {"role": "assistant", "content": "Dự án dự báo PM2.5"},
            {"role": "user", "content": "\n\t  \n"},
        ]

        with patch("src.chatbot.llm_client._try_stream_provider", return_value=iter(["OK"])) as mock_try:
            list(chat_stream(messages=raw_messages, providers=[provider]))

            mock_try.assert_called_once()
            passed_messages = mock_try.call_args[0][1]

            # Verify no empty or whitespace contents
            for m in passed_messages:
                assert m.get("content") is not None
                assert len(m["content"]) > 0
                assert m["content"].strip() == m["content"]

            # Contents check: system prompt + 2 valid messages
            roles_and_contents = [(m["role"], m["content"]) for m in passed_messages]
            assert ("user", "Dự án này làm gì?") in roles_and_contents
            assert ("assistant", "Dự án dự báo PM2.5") in roles_and_contents
            assert len([m for m in passed_messages if m["role"] == "user"]) == 1


class TestGroqPayloadOptimization:
    """Verify Groq max_tokens constraint."""

    def test_groq_caps_max_tokens_at_1024(self):
        """When provider is Groq, max_tokens must be capped at 1024 to preserve TPM."""
        groq_provider = LLMProvider(
            name="groq",
            display_name="Groq",
            base_url="https://api.groq.com/openai/v1",
            api_key="gsk-test",
            model="llama-3.1-8b-instant",
        )

        with patch("src.chatbot.llm_client._try_stream_provider", return_value=iter(["OK"])) as mock_try:
            list(chat_stream(messages=[{"role": "user", "content": "Hi"}], max_tokens=2048, providers=[groq_provider]))

            mock_try.assert_called_once()
            passed_max_tokens = mock_try.call_args[0][3]
            assert passed_max_tokens == 1024

    def test_non_groq_retains_full_max_tokens(self):
        """Non-Groq providers should retain the requested max_tokens."""
        gemini_provider = LLMProvider(
            name="gemini",
            display_name="Google Gemini",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key="aiza-test",
            model="gemini-2.0-flash",
        )

        with patch("src.chatbot.llm_client._try_stream_provider", return_value=iter(["OK"])) as mock_try:
            list(
                chat_stream(messages=[{"role": "user", "content": "Hi"}], max_tokens=2048, providers=[gemini_provider])
            )

            mock_try.assert_called_once()
            passed_max_tokens = mock_try.call_args[0][3]
            assert passed_max_tokens == 2048


class TestChatStreamErrorTransparency:
    """Verify detailed error messages when providers fail."""

    def test_all_providers_fail_shows_specific_reasons(self):
        """When all providers fail, chat_stream yields an informative breakdown of errors."""
        p_groq = LLMProvider(
            name="groq",
            display_name="Groq",
            base_url="https://api.groq.com/openai/v1",
            api_key="gsk-test",
            model="llama-3.1-8b-instant",
        )
        p_gemini = LLMProvider(
            name="gemini",
            display_name="Google Gemini",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key="aiza-test",
            model="gemini-2.0-flash",
        )

        def mock_try(provider, *args, **kwargs):
            if provider.name == "groq":
                provider._last_error = "Chạm giới hạn tốc độ (Rate Limit / TPM limit): 429 TPM limit"
            else:
                provider._last_error = "API key không hợp lệ hoặc hết hạn: 401 Unauthorized"
            return None

        with patch("src.chatbot.llm_client._try_stream_provider", side_effect=mock_try):
            chunks = list(chat_stream(messages=[{"role": "user", "content": "Hi"}], providers=[p_groq, p_gemini]))
            full_text = "".join(chunks)

            assert "Không thể kết nối với bất kỳ" in full_text
            assert "Groq" in full_text
            assert "Chạm giới hạn tốc độ (Rate Limit / TPM limit)" in full_text
            assert "Google Gemini" in full_text
            assert "API key không hợp lệ hoặc hết hạn" in full_text
            assert "Hướng dẫn khắc phục" in full_text

    def test_try_stream_provider_records_last_error_on_exception(self):
        """_try_stream_provider should set _last_error on the provider object on API errors."""
        provider = LLMProvider(
            name="groq",
            display_name="Groq",
            base_url="https://api.groq.com/openai/v1",
            api_key="gsk-test",
            model="openai/gpt-oss-20b",
        )

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("429 rate limit exceeded")

        with patch("src.chatbot.llm_client._build_client", return_value=mock_client):
            gen = _try_stream_provider(
                provider=provider,
                full_messages=[{"role": "user", "content": "Hi"}],
                temperature=0.3,
                max_tokens=100,
            )
            assert gen is None
            assert hasattr(provider, "_last_error")
            assert "Chạm giới hạn tốc độ" in provider._last_error

    def test_try_stream_provider_auto_recovers_deprecated_model(self):
        """When initial model fails with 404, query models.list and retry with active model."""
        provider = LLMProvider(
            name="groq",
            display_name="Groq",
            base_url="https://api.groq.com/openai/v1",
            api_key="gsk-test",
            model="deprecated-model-404",
        )

        mock_client = MagicMock()
        # First call fails with 404 model not found
        # Second call (retry) succeeds with active model stream
        mock_stream = [MagicMock()]
        mock_client.chat.completions.create.side_effect = [
            Exception("404 Model deprecated-model-404 not found"),
            mock_stream,
        ]
        mock_client.models.list.return_value.data = [
            MagicMock(id="whisper-large-v3"),
            MagicMock(id="openai/gpt-oss-20b"),
        ]

        with patch("src.chatbot.llm_client._build_client", return_value=mock_client):
            gen = _try_stream_provider(
                provider=provider,
                full_messages=[{"role": "user", "content": "Hi"}],
                temperature=0.3,
                max_tokens=100,
            )
            assert gen is not None
            assert provider.model == "openai/gpt-oss-20b"
            assert mock_client.chat.completions.create.call_count == 2


class TestPrimaryProviderPrioritization:
    """Verify primary provider prioritization in detect_available_providers."""

    def test_primary_provider_prioritized(self, monkeypatch):
        """Designated primary provider must be moved to index 0."""
        monkeypatch.setenv("RENDER", "true")
        session_keys = {
            "gemini": {"api_key": "k_gemini"},
            "groq": {"api_key": "k_groq"},
            "openai": {"api_key": "k_openai"},
        }

        # By default registry priorities: gemini (1) < openai (2) < groq (3)
        default_providers = detect_available_providers(session_keys)
        assert [p.name for p in default_providers] == ["gemini", "openai", "groq"]

        # When primary_provider is groq, groq must be first
        groq_first = detect_available_providers(session_keys, primary_provider="groq")
        assert groq_first[0].name == "groq"
        assert [p.name for p in groq_first] == ["groq", "gemini", "openai"]

        # When primary_provider is openai, openai must be first
        openai_first = detect_available_providers(session_keys, primary_provider="openai")
        assert openai_first[0].name == "openai"
        assert [p.name for p in openai_first] == ["openai", "gemini", "groq"]

        # When specified via session_keys["_primary"]
        session_keys_with_primary = dict(session_keys)
        session_keys_with_primary["_primary"] = "groq"
        detected = detect_available_providers(session_keys_with_primary)
        assert detected[0].name == "groq"
