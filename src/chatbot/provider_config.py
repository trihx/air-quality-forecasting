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

    def __post_init__(self):
        if not self.display_name:
            self.display_name = self.name


# ── Provider Registry ──

PROVIDER_REGISTRY: dict[str, dict] = {
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
        "default_model": "llama-3.3-70b-versatile",
        "env_key": "GROQ_API_KEY",
        "priority": 3,
        "is_local": False,
        "description": "Miễn phí 30 RPM, tốc độ cực nhanh",
    },
    "lm_studio": {
        "display_name": "LM Studio (Local)",
        "base_url": os.getenv("LM_STUDIO_URL", "http://host.docker.internal:8888/v1"),
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
        "best_for": "Phân tích sâu, phản biện",
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

    return LLMProvider(
        name=provider_name,
        display_name=reg["display_name"],
        base_url=custom_base_url or reg["base_url"],
        api_key=resolved_key,
        model=custom_model or reg["default_model"],
        priority=reg["priority"],
        is_local=reg["is_local"],
    )


def detect_available_providers(
    session_keys: dict | None = None,
) -> list[LLMProvider]:
    """Detect all available providers from env vars and session state.

    Args:
        session_keys: Dict of {provider_name: {api_key, base_url, model}} from UI.

    Returns:
        List of LLMProvider sorted by priority (lowest first = highest priority).
    """
    providers = []
    session_keys = session_keys or {}

    for name, reg in PROVIDER_REGISTRY.items():
        # Check session state first, then env vars
        session_cfg = session_keys.get(name, {})
        api_key = session_cfg.get("api_key", "") or os.getenv(reg["env_key"], "")

        if name == "lm_studio":
            # LM Studio is always "available" as fallback — connection check at runtime
            base_url = session_cfg.get("base_url", "") or reg["base_url"]
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

    # Sort by priority
    providers.sort(key=lambda p: p.priority)
    return providers


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
        return True, f"Kết nối thành công ({model_count} models)"
    except Exception as e:
        error = str(e)
        if "401" in error or "403" in error or "invalid" in error.lower():
            return False, "API key không hợp lệ"
        if "Connection" in error or "refused" in error or "timeout" in error.lower():
            return False, "Không thể kết nối server"
        if "429" in error:
            return False, "Rate limit — vui lòng thử lại sau"
        return False, f"Lỗi: {error[:100]}"
