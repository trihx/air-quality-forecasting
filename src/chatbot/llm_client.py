"""
Multi-LLM Client for PM2.5 AI Assistant.

Supports tiered fallback across multiple providers:
  Cloud API (Gemini/OpenAI/Groq) → LM Studio (local)

All providers use OpenAI-compatible API — zero extra dependencies.
"""

import logging
from collections.abc import Generator

from openai import OpenAI

from src.chatbot.provider_config import (
    LLMProvider,
    detect_available_providers,
    mask_api_key,
)

logger = logging.getLogger(__name__)

# System prompt for project-aware assistant
SYSTEM_PROMPT = """\
Bạn là trợ lý AI chuyên gia về dự án nghiên cứu \
"Dự báo nồng độ PM2.5 sử dụng ML và DL". \
Dự án này là đề án Thạc sĩ tại Đại học Cần Thơ (CTU).

## Vai trò của bạn:
- Trả lời câu hỏi dựa HOÀN TOÀN trên context được cung cấp
- **ƯU TIÊN PHƯƠNG PHÁP LUẬN** (TẠI SAO, CÁCH thực hiện)
- Giải thích quy trình, quyết định thiết kế, bài học kinh nghiệm
- Hỗ trợ phản biện: "Tại sao?", "Làm sao?", "Cơ sở nào?"
- Trả lời tiếng Việt, thuật ngữ kỹ thuật tiếng Anh khi cần

## Trọng tâm (theo ưu tiên):
1. **Phương pháp luận**: Cơ sở khoa học? So sánh alternatives?
2. **Quy trình**: Pipeline end-to-end, đảm bảo tính hợp lệ
3. **Thiết kế**: Anti-leakage, imputation, feature engineering
4. **Đánh giá**: MASE, multi-horizon, shuffle test
5. **Bài học**: Lỗi đã gặp, kiến thức rút ra
6. **Kết quả**: Metrics cụ thể (chỉ khi có context)

## Quy tắc:
1. CHỈ dựa trên context. Không có → nói "không tìm thấy"
2. Trích dẫn nguồn file khi có thể
3. Số liệu chính xác, KHÔNG bịa số
4. Giải thích: (a) Vấn đề, (b) Giải pháp, (c) Tại sao, (d) Kết quả
5. MAE (µg/m³), MASE (vs Persistence), R² (phương sai)

## Thông tin cơ bản (v9 Multi-Resolution):
- Dữ liệu: 209K records, 3.1 năm, IoT (~2 phút/lần), Sa Đéc, Đồng Tháp
- Target: PM2.5 (µg/m³)
- Pipeline: Raw→Clean(S-ESD)→Resample(15m/30m/1h)→Impute(Spline+KNN)→Features(119)→Eval
- Models: Persistence, ARIMA, SARIMAX, LightGBM, ElasticNet, RF, LSTM, GRU, TFT, Ensemble
- Horizons: 1h, 6h, 24h | Resolutions: 15m, 30m, 1h | Tests: 188+ passed
- Best (1h): GRU_15m MASE=0.667 — phá vỡ autocorrelation trap ở 15m/30m
- Best (6h): Ensemble_Weighted_30m MASE=0.382 | Best (24h): Ensemble_Weighted_30m MASE=0.469
- Persistence mạnh ở 1h (res 1h), nhưng bị đánh bại ở 15m/30m nhờ multi-resolution
- Anti-leakage: shift(1) + Purging Gap | Fair DL > Expert DL Pipeline
- Ablation Study: Tabular features cho DL > raw data cho IoT time series
"""


def _build_client(provider: LLMProvider) -> OpenAI | None:
    """Create OpenAI-compatible client for any provider."""
    try:
        client = OpenAI(
            base_url=provider.base_url,
            api_key=provider.api_key,
            timeout=60.0,
        )
        return client
    except Exception as e:
        logger.error(f"Cannot create client for {provider.display_name}: {e}")
        return None


def check_connection(provider: LLMProvider | None = None) -> bool:
    """Check if a provider is reachable.

    If no provider given, checks LM Studio (legacy behavior).
    """
    if provider is None:
        # Legacy: check LM Studio
        from src.chatbot.provider_config import get_provider_from_registry

        provider = get_provider_from_registry("lm_studio")
        if not provider:
            return False

    client = _build_client(provider)
    if not client:
        return False
    try:
        client.models.list()
        return True
    except Exception:
        return False


def get_available_models(provider: LLMProvider | None = None) -> list[str]:
    """List models available on a provider."""
    if provider is None:
        from src.chatbot.provider_config import get_provider_from_registry

        provider = get_provider_from_registry("lm_studio")
        if not provider:
            return []

    client = _build_client(provider)
    if not client:
        return []
    try:
        models = client.models.list()
        return [m.id for m in models.data] if models.data else []
    except Exception:
        return []


def _try_stream_provider(
    provider: LLMProvider,
    full_messages: list[dict],
    temperature: float,
    max_tokens: int,
) -> Generator[str, None, None] | None:
    """Attempt to stream from a single provider. Returns None on failure."""
    client = _build_client(provider)
    if not client:
        return None

    model = provider.model
    if not model and provider.is_local:
        # Auto-detect model for LM Studio
        try:
            models_list = client.models.list()
            if models_list.data:
                model = models_list.data[0].id
            else:
                return None
        except Exception:
            model = "local-model"

    if not model:
        return None

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=full_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        def _gen():
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        return _gen()
    except Exception as e:
        error_msg = str(e)
        logger.warning(f"Provider {provider.display_name} failed: {error_msg[:100]}")
        return None


def chat_stream(
    messages: list[dict],
    context: str = "",
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 2048,
    providers: list[LLMProvider] | None = None,
    session_keys: dict | None = None,
) -> Generator[str, None, None]:
    """Stream chat completion with tiered fallback.

    Priority: Cloud API (by priority) → LM Studio (fallback)

    Args:
        messages: Chat history [{role, content}, ...]
        context: RAG context to inject
        model: Override model (used for explicit selection)
        temperature: Creativity (0.0-1.0)
        max_tokens: Max response length
        providers: Pre-sorted provider list (if None, auto-detect)
        session_keys: Session state keys for provider config

    Yields:
        Response text chunks for streaming
    """
    # Build system message with RAG context
    system_content = SYSTEM_PROMPT
    if context:
        system_content += f"\n\n## Context từ dữ liệu dự án:\n{context}"

    full_messages = [{"role": "system", "content": system_content}] + messages

    # Detect providers if not given
    if providers is None:
        providers = detect_available_providers(session_keys)

    if not providers:
        yield (
            "⚠️ Chưa cấu hình LLM provider nào.\n\n"
            "**Hướng dẫn:**\n"
            "1. Vào **⚙️ Cấu Hình AI** ở sidebar bên trái\n"
            "2. Nhập API key (khuyến nghị: **Gemini** — miễn phí)\n"
            "3. Hoặc cài **LM Studio** trên máy và load model\n"
        )
        return

    # Try each provider in priority order
    tried = []
    for provider in providers:
        # Override model if explicitly specified
        if model:
            provider.model = model

        logger.info(f"Trying provider: {provider.display_name} (key: {mask_api_key(provider.api_key)})")

        gen = _try_stream_provider(provider, full_messages, temperature, max_tokens)
        if gen is not None:
            # Yield provider info header
            yield f"*🤖 {provider.display_name}*\n\n"
            yield from gen
            return

        tried.append(provider.display_name)

    # All providers failed
    tried_str = ", ".join(tried)
    yield (
        f"❌ Không thể kết nối với bất kỳ LLM nào.\n\n"
        f"**Đã thử:** {tried_str}\n\n"
        "**Hướng dẫn khắc phục:**\n"
        "1. Kiểm tra kết nối internet (cho Cloud API)\n"
        "2. Kiểm tra API key còn hạn sử dụng\n"
        "3. Mở LM Studio → Load model → Bật Server port 8888\n"
        "4. Reload trang dashboard\n"
    )
