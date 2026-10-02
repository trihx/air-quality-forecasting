"""Tests for Kaggle Ollama provider integration."""

from src.chatbot.provider_config import (
    KAGGLE_MODEL_RECOMMENDATIONS,
    PROVIDER_REGISTRY,
    detect_available_providers,
    get_provider_from_registry,
)


class TestKaggleProviderRegistry:
    """Tests for Kaggle Ollama provider in the registry."""

    def test_kaggle_provider_exists_in_registry(self):
        """Kaggle Ollama must be registered as a provider."""
        assert "kaggle_ollama" in PROVIDER_REGISTRY

    def test_kaggle_provider_has_correct_priority(self):
        """Kaggle Ollama should have highest priority (0)."""
        assert PROVIDER_REGISTRY["kaggle_ollama"]["priority"] == 0

    def test_kaggle_provider_display_name(self):
        """Display name should be descriptive."""
        assert "Kaggle" in PROVIDER_REGISTRY["kaggle_ollama"]["display_name"]
        assert "Ollama" in PROVIDER_REGISTRY["kaggle_ollama"]["display_name"]

    def test_kaggle_provider_default_model(self):
        """Default model should be set."""
        assert PROVIDER_REGISTRY["kaggle_ollama"]["default_model"] != ""

    def test_kaggle_provider_is_not_local(self):
        """Kaggle runs on cloud, not local."""
        assert PROVIDER_REGISTRY["kaggle_ollama"]["is_local"] is False

    def test_kaggle_provider_env_key(self):
        """Env key should be set for configuration."""
        assert PROVIDER_REGISTRY["kaggle_ollama"]["env_key"] == "KAGGLE_OLLAMA_URL"


class TestKaggleProviderDetection:
    """Tests for detecting Kaggle provider from session/env."""

    def test_kaggle_not_detected_without_url(self):
        """Kaggle should NOT be in providers when no URL configured."""
        providers = detect_available_providers({})
        kaggle = [p for p in providers if p.name == "kaggle_ollama"]
        assert len(kaggle) == 0

    def test_kaggle_detected_with_session_url(self):
        """Kaggle should be detected when tunnel URL is in session."""
        session_keys = {
            "kaggle_ollama": {
                "base_url": "https://test-abc.trycloudflare.com",
                "api_key": "ollama",
            }
        }
        providers = detect_available_providers(session_keys)
        kaggle = [p for p in providers if p.name == "kaggle_ollama"]
        assert len(kaggle) == 1
        assert kaggle[0].priority == 0

    def test_kaggle_has_highest_priority(self):
        """When configured, Kaggle should be first in provider list."""
        session_keys = {
            "kaggle_ollama": {
                "base_url": "https://test.trycloudflare.com",
                "api_key": "ollama",
            },
            "gemini": {"api_key": "test-key"},
        }
        providers = detect_available_providers(session_keys)
        assert providers[0].name == "kaggle_ollama"

    def test_kaggle_url_gets_v1_suffix(self):
        """Tunnel URL should get /v1 suffix for OpenAI compatibility."""
        session_keys = {
            "kaggle_ollama": {
                "base_url": "https://test.trycloudflare.com",
                "api_key": "ollama",
            }
        }
        providers = detect_available_providers(session_keys)
        kaggle = [p for p in providers if p.name == "kaggle_ollama"][0]
        assert kaggle.base_url.endswith("/v1")

    def test_kaggle_url_no_double_v1(self):
        """If URL already has /v1, don't add it again."""
        session_keys = {
            "kaggle_ollama": {
                "base_url": "https://test.trycloudflare.com/v1",
                "api_key": "ollama",
            }
        }
        providers = detect_available_providers(session_keys)
        kaggle = [p for p in providers if p.name == "kaggle_ollama"][0]
        assert not kaggle.base_url.endswith("/v1/v1")


class TestKaggleProviderFromRegistry:
    """Tests for creating Kaggle provider from registry."""

    def test_create_kaggle_provider(self):
        """Should create a valid provider."""
        provider = get_provider_from_registry(
            "kaggle_ollama",
            custom_base_url="https://test.trycloudflare.com",
        )
        assert provider is not None
        assert provider.name == "kaggle_ollama"
        assert provider.api_key == "ollama"

    def test_kaggle_provider_url_v1_suffix(self):
        """Provider URL should include /v1."""
        provider = get_provider_from_registry(
            "kaggle_ollama",
            custom_base_url="https://test.trycloudflare.com",
        )
        assert provider is not None
        assert provider.base_url.endswith("/v1")

    def test_kaggle_provider_custom_model(self):
        """Custom model should override default."""
        provider = get_provider_from_registry(
            "kaggle_ollama",
            custom_base_url="https://test.trycloudflare.com",
            custom_model="qwen3:8b",
        )
        assert provider is not None
        assert provider.model == "qwen3:8b"


class TestKaggleFallbackChain:
    """Tests for fallback behavior with Kaggle provider."""

    def test_fallback_without_kaggle(self):
        """Without Kaggle URL, fallback chain starts from next provider."""
        session_keys = {
            "gemini": {"api_key": "test-gemini-key"},
        }
        providers = detect_available_providers(session_keys)
        non_local = [p for p in providers if not p.is_local]
        assert non_local[0].name == "gemini"

    def test_full_fallback_chain_order(self):
        """Full chain: Kaggle → Gemini → Groq → LM Studio."""
        session_keys = {
            "kaggle_ollama": {
                "base_url": "https://test.trycloudflare.com",
                "api_key": "ollama",
            },
            "gemini": {"api_key": "test-gemini"},
            "groq": {"api_key": "test-groq"},
        }
        providers = detect_available_providers(session_keys)
        names = [p.name for p in providers]
        # Kaggle should be first (priority 0)
        assert names.index("kaggle_ollama") < names.index("gemini")
        assert names.index("gemini") < names.index("groq")
        assert names.index("groq") < names.index("lm_studio")


class TestKaggleModelRecommendations:
    """Tests for Kaggle model recommendations list."""

    def test_recommendations_not_empty(self):
        """Should have model recommendations."""
        assert len(KAGGLE_MODEL_RECOMMENDATIONS) > 0

    def test_recommendations_have_required_fields(self):
        """Each recommendation should have required fields."""
        for m in KAGGLE_MODEL_RECOMMENDATIONS:
            assert "name" in m
            assert "vram" in m
            assert "vietnamese" in m
            assert "best_for" in m
            assert "recommended" in m

    def test_at_least_one_recommended(self):
        """At least one model should be marked as recommended."""
        recommended = [m for m in KAGGLE_MODEL_RECOMMENDATIONS if m["recommended"]]
        assert len(recommended) >= 1


class TestKaggleSetupGuideCells:
    """Tests for Kaggle notebook setup cells (deadlock prevention & warm-up)."""

    def test_kaggle_cell_2_prevents_pipe_buffer_deadlock(self):
        """Cell 2 must write logs to file and avoid subprocess.PIPE to eliminate 64KB deadlock."""
        from src.chatbot.kaggle_setup_guide import KAGGLE_CELL_2_START

        assert 'open("ollama.log", "w")' in KAGGLE_CELL_2_START
        assert "stderr=subprocess.STDOUT" in KAGGLE_CELL_2_START
        assert "start_new_session=True" in KAGGLE_CELL_2_START
        assert "stdout=subprocess.PIPE" not in KAGGLE_CELL_2_START
        assert "stderr=subprocess.PIPE" not in KAGGLE_CELL_2_START

    def test_kaggle_cell_2_configures_env_and_cleans_old_processes(self):
        """Cell 2 must clean previous instance and configure keep-alive and host."""
        from src.chatbot.kaggle_setup_guide import KAGGLE_CELL_2_START

        assert 'subprocess.run(["pkill", "-f", "ollama serve"], check=False)' in KAGGLE_CELL_2_START
        assert "OLLAMA_HOST" in KAGGLE_CELL_2_START
        assert "0.0.0.0:11434" in KAGGLE_CELL_2_START
        assert "OLLAMA_ORIGINS" in KAGGLE_CELL_2_START
        assert "OLLAMA_KEEP_ALIVE" in KAGGLE_CELL_2_START
        assert "24h" in KAGGLE_CELL_2_START

    def test_kaggle_cell_2_includes_healthcheck_poll(self):
        """Cell 2 must include healthcheck poll with urllib.request for 127.0.0.1:11434."""
        from src.chatbot.kaggle_setup_guide import KAGGLE_CELL_2_START

        assert "urllib.request" in KAGGLE_CELL_2_START
        assert "http://127.0.0.1:11434" in KAGGLE_CELL_2_START
        assert "server_ready" in KAGGLE_CELL_2_START
        assert "Ollama server đã sẵn sàng" in KAGGLE_CELL_2_START

    def test_kaggle_cell_3_includes_model_warmup(self):
        """Cell 3 must warm up the model into GPU VRAM after pulling."""
        from src.chatbot.kaggle_setup_guide import KAGGLE_CELL_3_MODEL

        assert "!ollama pull qwen3:4b" in KAGGLE_CELL_3_MODEL
        assert '!ollama run qwen3:4b "Xin chào"' in KAGGLE_CELL_3_MODEL
        assert "warm-up" in KAGGLE_CELL_3_MODEL.lower()
        assert "VRAM" in KAGGLE_CELL_3_MODEL
