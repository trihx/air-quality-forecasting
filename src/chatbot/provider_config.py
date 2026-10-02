"""
Multi-LLM Provider Configuration for PM2.5 AI Assistant.

Supports OpenAI, Gemini, Groq, and LM Studio — all via OpenAI-compatible API.
Zero additional dependencies required (uses existing `openai` SDK).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class LLMProvider:
    """Configuration for a single LLM provider."""

    name: str
    base_url: str
    api_key: str = ""
    model: str = ""
    priority: int = 0  # Lower = higher priority
    is_local: bool = False
    display_name: str = ""
    _last_error: str = field(default="", repr=False)

    def __post_init__(self):
        if not self.display_name:
            self.display_name = self.name


def get_lm_studio_default_url() -> str:
    """Get default LM Studio URL dynamically from environment."""
    return os.getenv("LM_STUDIO_URL", "http://host.docker.internal:8888/v1")


# ── Provider Registry ──

PROVIDER_REGISTRY: dict[str, dict] = {
    "kaggle_ollama": {
        "display_name": "Kaggle Ollama (Free GPU)",
        "base_url": "",  # Dynamic — user pastes tunnel URL
        "default_model": "qwen3:4b",
        "env_key": "KAGGLE_OLLAMA_URL",
        "priority": 0,  # Highest priority when configured
        "is_local": False,
        "description": "Free 32GB VRAM qua Kaggle + Ollama + Cloudflare Tunnel",
    },
    "gemini": {
        "display_name": "Google Gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "default_model": "gemini-2.0-flash",
        "env_key": "GEMINI_API_KEY",
        "priority": 1,  # Highest (free, no CC required)
        "is_local": False,
        "description": "Miễn phí 15 RPM, không cần thẻ tín dụng",
    },
    "openai": {
        "display_name": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "env_key": "OPENAI_API_KEY",
        "priority": 2,
        "is_local": False,
        "description": "Mạnh mẽ, $5 credit khi đăng ký mới",
    },
    "groq": {
        "display_name": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.1-8b-instant",
        "env_key": "GROQ_API_KEY",
        "priority": 3,
        "is_local": False,
        "description": "Miễn phí 30 RPM, tốc độ cực nhanh",
    },
    "lm_studio": {
        "display_name": "LM Studio (Local)",
        "base_url": "http://host.docker.internal:8888/v1",
        "default_model": "",  # Auto-detect from server
        "env_key": "LM_STUDIO_API_KEY",
        "priority": 99,  # Fallback
        "is_local": True,
        "description": "Chạy local trên máy, không giới hạn, cần cài LM Studio",
    },
}

# Recommended local models for the committee
LOCAL_MODEL_RECOMMENDATIONS = [
    {
        "name": "Qwen3-4B (Q4_K_M)",
        "vram": "4GB",
        "vietnamese": "✅ Xuất sắc",
        "speed": "~30 tok/s",
        "best_for": "RAG Q&A, giải thích thesis",
        "recommended": True,
    },
    {
        "name": "Gemma 3 4B (Q4)",
        "vram": "4GB",
        "vietnamese": "✅ Tốt",
        "speed": "~35 tok/s",
        "best_for": "Tóm tắt, trích xuất",
        "recommended": False,
    },
    {
        "name": "Qwen3-8B (Q4_K_M)",
        "vram": "6GB",
        "vietnamese": "✅ Rất tốt",
        "speed": "~20 tok/s",
        "best_for": "Phân tích kỹ thuật chuyên sâu",
        "recommended": False,
    },
    {
        "name": "Llama 3.2 3B (Q4)",
        "vram": "3GB",
        "vietnamese": "⚠️ Khá",
        "speed": "~40 tok/s",
        "best_for": "Nhanh, tài nguyên thấp",
        "recommended": False,
    },
]

KAGGLE_MODEL_RECOMMENDATIONS = [
    {
        "name": "qwen3:4b",
        "vram": "~4GB",
        "vietnamese": "✅ Xuất sắc",
        "speed": "~30 tok/s trên T4",
        "best_for": "RAG Q&A, giải thích đề án, Tiếng Việt tốt",
        "recommended": True,
    },
    {
        "name": "gemma3:4b",
        "vram": "~4GB",
        "vietnamese": "✅ Tốt",
        "speed": "~35 tok/s trên T4",
        "best_for": "Tóm tắt, trích xuất nhanh",
        "recommended": False,
    },
    {
        "name": "qwen3:8b",
        "vram": "~6GB",
        "vietnamese": "✅ Rất tốt",
        "speed": "~20 tok/s trên T4",
        "best_for": "Phân tích kỹ thuật chuyên sâu",
        "recommended": False,
    },
    {
        "name": "hf.co/JonathanColetti/Qwen3.8-27B-Uncensored-GGUF:Q4_K_M",
        "vram": "~20GB",
        "vietnamese": "✅ Xuất sắc",
        "speed": "~8 tok/s trên 2×T4",
        "best_for": "Phân tích sâu, cần cả 2 GPU",
        "recommended": False,
    },
]

GROQ_MODEL_RECOMMENDATIONS = [
    {
        "name": "llama-3.1-8b-instant",
        "description": "Siêu tốc (>800 tok/s), hạn mức cao (30,000 TPM, 30 RPM), tối ưu hỏi đáp",
        "recommended": True,
    },
    {
        "name": "llama-3.3-70b-versatile",
        "description": "Mô hình 70B thông minh sâu sắc (hạn mức 6,000 TPM)",
        "recommended": False,
    },
    {
        "name": "gemma2-9b-it",
        "description": "Mô hình Google Gemma 2 tối ưu trên phần cứng LPU Groq",
        "recommended": False,
    },
]


def mask_api_key(key: str) -> str:
    """Mask API key for safe display in logs/UI.

    Examples:
        sk-abc123def456 → sk-abc...f456
        AIza1234567890 → AIza...7890
    """
    if not key or len(key) < 8:
        return "***"
    return f"{key[:4]}...{key[-4:]}"


def get_provider_from_registry(
    provider_name: str,
    api_key: str = "",
    custom_base_url: str = "",
    custom_model: str = "",
) -> LLMProvider | None:
    """Create LLMProvider from registry with optional overrides."""
    reg = PROVIDER_REGISTRY.get(provider_name)
    if not reg:
        logger.warning(f"Unknown provider: {provider_name}")
        return None

    # Resolve API key: explicit > session > env var > default
    resolved_key = api_key or os.getenv(reg["env_key"], "")
    if provider_name == "lm_studio" and not resolved_key:
        resolved_key = "lm-studio"  # LM Studio doesn't require real key
    elif provider_name == "kaggle_ollama" and not resolved_key:
        resolved_key = "ollama"  # Ollama doesn't require real key

    # Ensure Ollama endpoints have /v1 suffix
    final_base_url = custom_base_url or reg["base_url"]
    if provider_name == "lm_studio" and not custom_base_url:
        final_base_url = get_lm_studio_default_url()
    elif provider_name == "kaggle_ollama":
        final_base_url = custom_base_url or os.getenv(reg["env_key"], "") or reg["base_url"]
        if final_base_url:
            final_base_url = final_base_url.rstrip("/")
            if not final_base_url.endswith("/v1"):
                final_base_url += "/v1"

    return LLMProvider(
        name=provider_name,
        display_name=reg["display_name"],
        base_url=final_base_url,
        api_key=resolved_key,
        model=custom_model or reg["default_model"],
        priority=reg["priority"],
        is_local=reg["is_local"],
    )


def is_cloud_environment() -> bool:
    """Check if running in a cloud hosting environment (Render, Vercel, Fly, etc.).

    Detects common cloud platform environment variables or production flags.
    """
    cloud_env_keys = ("RENDER", "RENDER_SERVICE_ID", "VERCEL", "FLY_ALLOC_ID")
    if any(os.getenv(k) for k in cloud_env_keys):
        return True
    if os.getenv("ENVIRONMENT", "").lower() == "production":
        return True
    return os.getenv("IS_CLOUD", "").lower() in ("true", "1", "yes")


def detect_available_providers(
    session_keys: dict | None = None,
    primary_provider: str | None = None,
) -> list[LLMProvider]:
    """Detect all available providers from env vars and session state.

    Args:
        session_keys: Dict of {provider_name: {api_key, base_url, model}} from UI.
        primary_provider: Optional provider name to prioritize as #1.

    Returns:
        List of LLMProvider sorted by priority (lowest first = highest priority).
    """
    providers = []
    session_keys = session_keys or {}
    is_cloud = is_cloud_environment()

    for name, reg in PROVIDER_REGISTRY.items():
        # Check session state first, then env vars
        session_cfg = session_keys.get(name, {})
        api_key = session_cfg.get("api_key", "") or os.getenv(reg["env_key"], "")

        if name == "kaggle_ollama":
            # Kaggle Ollama uses tunnel URL instead of API key
            tunnel_url = session_cfg.get("base_url", "") or os.getenv(reg["env_key"], "")
            if tunnel_url:
                # Ensure URL ends with /v1 for OpenAI compatibility
                base = tunnel_url.rstrip("/")
                if not base.endswith("/v1"):
                    base += "/v1"
                provider = LLMProvider(
                    name=name,
                    display_name=reg["display_name"],
                    base_url=base,
                    api_key="ollama",  # Ollama doesn't need real API key
                    model=session_cfg.get("model", "") or reg["default_model"],
                    priority=reg["priority"],
                    is_local=False,
                )
                providers.append(provider)
        elif name == "lm_studio":
            default_docker_url = "http://host.docker.internal:8888/v1"
            session_url = session_cfg.get("base_url", "").strip()
            env_url = os.getenv("LM_STUDIO_URL", "").strip()

            if is_cloud:
                # On Cloud: only add LM Studio if explicitly configured with a non-default custom URL
                candidate_url = session_url or env_url
                if candidate_url and candidate_url.rstrip("/") != default_docker_url.rstrip("/"):
                    provider = LLMProvider(
                        name=name,
                        display_name=reg["display_name"],
                        base_url=candidate_url,
                        api_key=api_key or "lm-studio",
                        model=session_cfg.get("model", "") or reg["default_model"],
                        priority=reg["priority"],
                        is_local=True,
                    )
                    providers.append(provider)
            else:
                # Local environment: LM Studio is always available as fallback
                base_url = session_url or env_url or get_lm_studio_default_url()
                provider = LLMProvider(
                    name=name,
                    display_name=reg["display_name"],
                    base_url=base_url,
                    api_key=api_key or "lm-studio",
                    model=session_cfg.get("model", "") or reg["default_model"],
                    priority=reg["priority"],
                    is_local=True,
                )
                providers.append(provider)
        elif api_key:
            provider = LLMProvider(
                name=name,
                display_name=reg["display_name"],
                base_url=session_cfg.get("base_url", "") or reg["base_url"],
                api_key=api_key,
                model=session_cfg.get("model", "") or reg["default_model"],
                priority=reg["priority"],
                is_local=False,
            )
            providers.append(provider)

    # Sort by priority, placing primary_provider first if specified
    resolved_primary = (
        primary_provider
        or (session_keys.get("_primary") if session_keys else None)
        or (session_keys.get("primary_provider") if session_keys else None)
    )
    if resolved_primary:
        providers.sort(key=lambda p: (0 if p.name == resolved_primary else 1, p.priority))
    else:
        providers.sort(key=lambda p: p.priority)
    return providers


def sanitize_error_message(error_str: str) -> str:
    """Remove sensitive authorization tokens, keys, and credentials from error strings."""
    import re

    # Redact Bearer / Authorization tokens
    sanitized = re.sub(
        r"(Bearer\s+)[A-Za-z0-9_\-\.]+",
        r"\1[REDACTED]",
        error_str,
        flags=re.IGNORECASE,
    )
    # Redact api_key=..., key=..., token=...
    sanitized = re.sub(
        r"((?:api[_-]?key|token|auth(?:orization)?|secret)\s*[:=]\s*)[A-Za-z0-9_\-\.]+",
        r"\1[REDACTED]",
        sanitized,
        flags=re.IGNORECASE,
    )
    # Redact OpenAI sk-... or Google AIza... tokens
    sanitized = re.sub(r"\b(sk-[A-Za-z0-9_\-]{8,})\b", "[REDACTED_KEY]", sanitized)
    sanitized = re.sub(r"\b(AIza[A-Za-z0-9_\-]{10,})\b", "[REDACTED_KEY]", sanitized)
    return sanitized


def validate_provider_connection(provider: LLMProvider) -> tuple[bool, str]:
    """Test if a provider is reachable with a lightweight API call.

    Returns:
        (success, message) tuple.
    """
    from openai import OpenAI

    try:
        client = OpenAI(
            base_url=provider.base_url,
            api_key=provider.api_key,
            timeout=10.0,
        )
        models = client.models.list()
        model_count = len(models.data) if models.data else 0
    except Exception as e:
        error = sanitize_error_message(str(e))
        if "401" in error or "403" in error or "invalid" in error.lower():
            return False, "API key không hợp lệ"
        if "Connection" in error or "refused" in error or "timeout" in error.lower():
            return False, "Không thể kết nối server"
        if "429" in error:
            return False, "Rate limit — vui lòng thử lại sau"
        return False, f"Lỗi: {error[:100]}"

    # Chat probe: verify model can actually complete chat requests
    try:
        probe_model = provider.model or (models.data[0].id if models.data else "default")
        client.chat.completions.create(
            model=probe_model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=2,
            timeout=5.0,
        )
        return True, f"Kết nối & kiểm tra Chat thành công ({model_count} models)"
    except Exception as probe_err:
        err_str = sanitize_error_message(str(probe_err))
        if "429" in err_str or "rate limit" in err_str.lower():
            return False, f"Chạm giới hạn tốc độ (Rate Limit / TPM) trên {provider.display_name}: {err_str[:80]}"
        if "404" in err_str or "model" in err_str.lower():
            return False, f"Model '{probe_model}' không tồn tại hoặc không hỗ trợ chat"
        if "400" in err_str:
            return False, f"Lỗi tham số chat: {err_str[:80]}"
        return False, f"Lỗi chat probe: {err_str[:80]}"
