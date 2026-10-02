"""
Multi-LLM Client for PM2.5 AI Assistant.

Supports tiered fallback across multiple providers:
  Cloud API (Gemini/OpenAI/Groq) → LM Studio (local)

All providers use OpenAI-compatible API — zero extra dependencies.
"""

import contextlib
import hashlib
import logging
from collections import OrderedDict
from collections.abc import Generator

try:
    import httpx

    DEFAULT_TIMEOUT: float | httpx.Timeout = httpx.Timeout(
        connect=15.0,
        read=90.0,
        write=15.0,
        pool=15.0,
    )
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]
    DEFAULT_TIMEOUT = 60.0

from openai import OpenAI

from src.chatbot.provider_config import (
    LLMProvider,
    detect_available_providers,
    is_cloud_environment,
    mask_api_key,
    sanitize_error_message,
)

logger = logging.getLogger(__name__)

# System prompt for project-aware assistant
SYSTEM_PROMPT = """\
Bạn là trợ lý AI phân tích kỹ thuật chuyên sâu về hệ thống \
"Dự báo nồng độ PM2.5 đa độ phân giải bằng Học máy & Học sâu".

## Vai trò của bạn:
- Trả lời câu hỏi dựa HOÀN TOÀN trên context kỹ thuật được cung cấp
- **ƯU TIÊN BẢN CHẤT KỸ THUẬT & THỰC NGHIỆM** (Tại sao thiết kế như vậy, cơ chế hoạt động thực tế)
- Giải thích quy trình tiền xử lý, kiến trúc mô hình, kiểm định thực nghiệm và bài học vận hành
- Làm rõ các câu hỏi kỹ thuật chuyên sâu: "Tại sao?", "Cơ chế ra sao?", "Cơ sở thực nghiệm nào?"
- Trả lời bằng tiếng Việt tự nhiên, trực diện, gãy gọn; giữ nguyên thuật ngữ kỹ thuật chuyên ngành khi cần

## Trọng tâm (theo ưu tiên):
1. **Phương pháp luận & Thiết kế**: Cơ chế chống rò rỉ (shift-1), nội suy phân tầng, 119 đặc trưng
2. **Kiểm định thực nghiệm**: Điểm ngọt 30 phút, kiểm định Diebold-Mariano, bẫy tự tương quan
3. **Mô hình**: Ensemble Weighted (LightGBM + GRU), tham số, cấu hình huấn luyện
4. **Đánh giá & Độ bất định**: MASE, MAE, CQR + ACI khoảng tin cậy 90%
5. **Giải thích XAI**: Ngưỡng bùng phát SHAP (14–17 µg/m³ & >17 µg/m³)
6. **Bài học & Giới hạn**: Xử lý dữ liệu IoT thực tế, bias nguồn dữ liệu ngoài

## Quy tắc:
1. CHỈ dựa trên context kỹ thuật của hệ thống. Không có → nói rõ "không có trong tài liệu hệ thống"
2. Trích dẫn module hoặc tệp nguồn liên quan khi có thể
3. Số liệu chính xác tuyệt đối, KHÔNG ước đoán ngoài dữ liệu kiểm chứng
4. Cấu trúc giải thích: (a) Bản chất bài toán, (b) Giải pháp kỹ thuật, (c) Cơ chế hoạt động, (d) Kết quả đo lường
5. Đơn vị chuẩn: MAE (µg/m³), MASE (so với Persistence), R² (tỷ lệ phương sai giải thích)

## Thông tin hệ thống cốt lõi (v9 Multi-Resolution):
- Dữ liệu: 209K bản ghi, 3.1 năm, cảm biến IoT (~2 phút/lần), Sa Đéc, Đồng Tháp
- Target: PM2.5 (µg/m³)
- Pipeline 7 bước: Raw → Clean (S-ESD) → Resample (15m/30m/1h) → Impute (Spline+KNN) → Features (119) → Split → Models → Eval
- Models: Persistence, ARIMA, SARIMAX, LightGBM, ElasticNet, RF, LSTM, GRU, TFT, Ensemble
- Tầm dự báo: 1h, 6h, 24h | Độ phân giải: 15m, 30m, 1h | Kiểm thử tự động: 250 tests passed ✅
- Tối ưu 1h: GRU_15m MASE=0.667 — phá vỡ autocorrelation trap ở độ phân giải cao
- Tối ưu 6h: Ensemble_Weighted_30m MASE=0.382 | Tối ưu 24h: Ensemble_Weighted_30m MASE=0.469
- Điểm ngọt Pareto: 30 phút đạt hiệu năng số 1 trên >80% kịch bản đánh giá ở 6h và 24h
- Anti-leakage: shift(1) bắt buộc cho 100% biến trễ và rolling, triệt tiêu R² ảo
- XAI: Ngưỡng bùng phát ô nhiễm phi tuyến tại Sa Đéc: 14–17 µg/m³ và >17 µg/m³
"""


_MAX_CLIENT_CACHE_SIZE = 8
_TimeoutCacheKey = tuple[float | None, ...]
_ClientCacheKey = tuple[str, str, _TimeoutCacheKey]
_CLIENT_CACHE: OrderedDict[_ClientCacheKey, OpenAI] = OrderedDict()


def _normalize_timeout_key(
    timeout: float | httpx.Timeout | None,
) -> _TimeoutCacheKey:
    """Normalize timeout parameter into a hashable cache key."""
    if httpx is not None and isinstance(timeout, httpx.Timeout):
        return (timeout.connect, timeout.read, timeout.write, timeout.pool)
    if isinstance(timeout, (int, float)):
        return (float(timeout),)
    if timeout is None:
        return (None,)
    return (str(timeout),)  # type: ignore[return-value]


def clear_client_cache() -> None:
    """Clear cached OpenAI client instances and cleanly close connection pools."""
    for client in _CLIENT_CACHE.values():
        with contextlib.suppress(Exception):
            client.close()
    _CLIENT_CACHE.clear()


def _build_client(
    provider: LLMProvider,
    timeout: float | httpx.Timeout | None = None,
) -> OpenAI | None:
    """Create or retrieve cached OpenAI-compatible client for any provider.

    Reuses existing client instances to preserve HTTP connection pools and prevent
    socket descriptor exhaustion. Uses hashed key and bounds cache size to 8 entries.
    Defaults to granular timeout (connect=15s, read=90s, write=15s, pool=15s) to detect
    dead tunnels fast while allowing sufficient time for long inference generation.
    """
    effective_timeout = timeout if timeout is not None else DEFAULT_TIMEOUT
    key_hash = hashlib.sha256(provider.api_key.encode("utf-8")).hexdigest()[:16]
    timeout_key = _normalize_timeout_key(effective_timeout)
    cache_key = (provider.base_url, key_hash, timeout_key)

    if cache_key in _CLIENT_CACHE:
        _CLIENT_CACHE.move_to_end(cache_key)
        return _CLIENT_CACHE[cache_key]

    try:
        client = OpenAI(
            base_url=provider.base_url,
            api_key=provider.api_key,
            timeout=effective_timeout,
        )
        while len(_CLIENT_CACHE) >= _MAX_CLIENT_CACHE_SIZE:
            _, evicted_client = _CLIENT_CACHE.popitem(last=False)
            with contextlib.suppress(Exception):
                evicted_client.close()
        _CLIENT_CACHE[cache_key] = client
        return client
    except Exception as e:
        sanitized_err = sanitize_error_message(str(e))
        logger.error(f"Cannot create client for {provider.display_name}: {sanitized_err}")
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
        provider._last_error = "Không thể khởi tạo client (URL hoặc Key không hợp lệ)"
        return None

    model = provider.model
    if provider.name == "kaggle_ollama":
        # Auto-detect real model on Kaggle Ollama server if model not set or doesn't match
        try:
            models_list = client.models.list()
            available_ids = [m.id for m in models_list.data] if models_list.data else []
            if available_ids:
                if not model or model not in available_ids:
                    model = available_ids[0]
                    provider.model = model
            elif not model:
                model = "qwen3:4b"
        except Exception as e:
            sanitized_err = sanitize_error_message(str(e))
            logger.warning(f"Could not auto-detect Kaggle model: {sanitized_err}")
            if not model:
                model = "qwen3:4b"
    elif not model and provider.is_local:
        # Auto-detect model for LM Studio
        try:
            models_list = client.models.list()
            if models_list.data:
                model = models_list.data[0].id
                provider.model = model
            else:
                provider._last_error = "Không tìm thấy model nào đang được load trên LM Studio"
                return None
        except Exception as lm_err:
            sanitized_err = sanitize_error_message(str(lm_err))
            provider._last_error = f"Không thể lấy danh sách model từ LM Studio: {sanitized_err[:80]}"
            return None

    if not model:
        provider._last_error = "Chưa cấu hình model hoặc không phát hiện được model"
        return None

    def _create_guarded_generator(active_client, stream_obj, active_model):
        def _gen():
            total_yielded_chars = 0
            interrupted = False
            try:
                for chunk in stream_obj:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    content = getattr(delta, "content", None) or ""
                    # Check reasoning content (e.g. Qwen 3 thinking tokens)
                    reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "thinking", None) or ""
                    if reasoning:
                        logger.debug("Received reasoning token from %s", provider.display_name)
                    if content:
                        total_yielded_chars += len(content)
                        yield content
            except Exception as stream_err:
                interrupted = True
                sanitized_err = sanitize_error_message(str(stream_err))
                logger.warning(f"Stream interrupted on {provider.display_name}: {sanitized_err}")
                yield "\n\n⚠️ *[Kết nối bị gián đoạn giữa chừng]*"

            # Anti-Empty Guard: If stream finished normally without exception but produced 0 content chars
            # (e.g. model consumed tokens only in thinking or stream closed early),
            # execute a single non-stream call to retrieve the complete answer.
            if total_yielded_chars == 0 and not interrupted:
                logger.info(
                    f"Stream yielded 0 content chars on {provider.display_name}. "
                    "Triggering anti-empty non-streaming fallback."
                )
                try:
                    completion = active_client.chat.completions.create(
                        model=active_model,
                        messages=full_messages,  # type: ignore[arg-type]
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=False,
                    )
                    if completion.choices and completion.choices[0].message:
                        msg = completion.choices[0].message
                        raw_content = msg.content or ""
                        raw_reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "thinking", None) or ""
                        if raw_content and raw_content.strip():
                            yield raw_content
                        elif raw_reasoning and raw_reasoning.strip():
                            yield f"*(Trích xuất từ quá trình suy luận của mô hình)*\n\n{raw_reasoning.strip()}"
                except Exception as fallback_err:
                    sanitized_err = sanitize_error_message(str(fallback_err))
                    logger.warning(f"Non-stream fallback failed on {provider.display_name}: {sanitized_err}")

        return _gen()

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=full_messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        return _create_guarded_generator(client, stream, model)
    except Exception as e:
        error_msg = sanitize_error_message(str(e))
        logger.warning(f"Provider {provider.display_name} failed: {error_msg[:100]}")
        if "429" in error_msg or "rate limit" in error_msg.lower() or "tpm" in error_msg.lower():
            provider._last_error = f"Chạm giới hạn tốc độ (Rate Limit / TPM limit): {error_msg[:120]}"
        elif "401" in error_msg or "403" in error_msg or "invalid_api_key" in error_msg.lower():
            provider._last_error = f"API key không hợp lệ hoặc hết hạn: {error_msg[:120]}"
        elif "404" in error_msg or "model_not_found" in error_msg.lower() or "not found" in error_msg.lower():
            provider._last_error = f"Model '{model}' không tồn tại trên {provider.display_name}: {error_msg[:120]}"
        elif "400" in error_msg:
            provider._last_error = f"Lỗi tham số yêu cầu (400 Bad Request): {error_msg[:120]}"
        elif "connection" in error_msg.lower() or "timeout" in error_msg.lower():
            provider._last_error = f"Lỗi kết nối / timeout: {error_msg[:120]}"
        else:
            provider._last_error = f"Lỗi gọi API: {error_msg[:120]}"

        # If Kaggle Ollama failed due to model issue, retry with first available model if different
        if provider.name == "kaggle_ollama" and ("not found" in error_msg.lower() or "404" in error_msg):
            try:
                models_list = client.models.list()
                if models_list.data and models_list.data[0].id != model:
                    fallback_model = models_list.data[0].id
                    logger.info(f"Retrying Kaggle Ollama with detected model: {fallback_model}")
                    provider.model = fallback_model
                    retry_stream = client.chat.completions.create(
                        model=fallback_model,
                        messages=full_messages,  # type: ignore[arg-type]
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=True,
                    )

                    return _create_guarded_generator(client, retry_stream, fallback_model)
            except Exception as retry_err:
                sanitized_err = sanitize_error_message(str(retry_err))
                logger.warning(f"Retry Kaggle Ollama failed: {sanitized_err}")
                provider._last_error = f"Retry Kaggle thất bại: {sanitized_err[:120]}"
        return None


def chat_stream(
    messages: list[dict],
    context: str = "",
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 2048,
    providers: list[LLMProvider] | None = None,
    session_keys: dict | None = None,
    primary_provider: str | None = None,
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
        primary_provider: Optional provider name to prioritize as #1

    Yields:
        Response text chunks for streaming
    """
    # Build system message with RAG context
    system_content = SYSTEM_PROMPT
    if context and context.strip():
        system_content += f"\n\n## Context từ dữ liệu dự án:\n{context.strip()}"

    raw_messages = [{"role": "system", "content": system_content}] + messages

    # Sanitize messages: eliminate empty content and strip whitespace
    # Triệt tiêu 100% nguy cơ lỗi Groq 400 Bad Request
    full_messages = [
        {"role": m["role"], "content": str(m["content"]).strip()}
        for m in raw_messages
        if m.get("content") and str(m["content"]).strip()
    ]

    # Detect providers if not given
    if providers is None:
        providers = detect_available_providers(session_keys, primary_provider=primary_provider)

    if not providers:
        if is_cloud_environment():
            yield (
                "⚠️ Chưa kích hoạt AI Provider nào trên Cloud Server.\n\n"
                "**Hướng dẫn kích hoạt (Hoàn toàn Miễn phí):**\n"
                "1. **Google Gemini (Khuyên dùng):** Nhập API Key miễn phí (15 RPM) tại **⚙️ Cấu Hình AI Provider** ở sidebar\n"
                "2. **Kaggle Ollama (Free GPU 32GB):** Dán Cloudflare Tunnel URL từ Kaggle notebook vào sidebar\n"
                "3. **Groq Cloud:** Nhập Groq API Key siêu tốc (miễn phí 30 RPM)\n"
            )
        else:
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
    failed_reasons: dict[str, str] = {}
    for provider in providers:
        # Override model if explicitly specified
        if model:
            provider.model = model

        # Groq Payload Optimization: bound max_tokens to 1024 to preserve TPM limit
        effective_max_tokens = min(max_tokens, 1024) if provider.name == "groq" and max_tokens > 1024 else max_tokens

        logger.info(f"Trying provider: {provider.display_name} (key: {mask_api_key(provider.api_key)})")

        gen = _try_stream_provider(provider, full_messages, temperature, effective_max_tokens)
        if gen is not None:
            # Peek first chunk to verify provider actually produces content
            try:
                first_chunk = next(gen)
            except StopIteration:
                first_chunk = None
            except Exception as stream_err:
                first_chunk = None
                provider._last_error = sanitize_error_message(str(stream_err))

            if first_chunk is not None:
                # Yield provider info header
                yield f"*🤖 {provider.display_name}*\n\n"
                yield first_chunk
                yield from gen
                return
            else:
                if not getattr(provider, "_last_error", None):
                    provider._last_error = "Model không phản hồi dữ liệu (0 tokens)"

        err_reason = getattr(provider, "_last_error", "Không thể kết nối hoặc không nhận được phản hồi")
        failed_reasons[provider.display_name] = err_reason
        tried.append(provider.display_name)

    # All providers failed
    is_cloud = is_cloud_environment()
    if is_cloud:
        yield (f"❌ Không thể kết nối với bất kỳ AI Provider nào.\n\n**Đã thử ({len(tried)}):**\n")
        for p_name, p_err in failed_reasons.items():
            yield f"• **{p_name}**: {p_err}\n"
        yield (
            "\n**Hướng dẫn khắc phục trên Cloud Server:**\n"
            "1. **Groq**: Nếu chạm TPM limit (429), chuyển sang model `llama-3.1-8b-instant` (hạn mức 30,000 TPM) tại tab Groq hoặc sidebar\n"
            "2. **Google Gemini**: Kiểm tra lại API key hoặc quota tại Google AI Studio (miễn phí 15 RPM)\n"
            "3. **Kaggle Ollama**: Kiểm tra Cloudflare Tunnel URL còn online không (Kaggle session có bị timeout không)\n"
            "4. Cập nhật cấu hình tại **⚙️ Cấu Hình AI Provider** ở sidebar hoặc bảng trên trang\n"
            "5. Reload lại trang dashboard\n"
        )
    else:
        yield (f"❌ Không thể kết nối với bất kỳ LLM nào.\n\n**Đã thử ({len(tried)}):**\n")
        for p_name, p_err in failed_reasons.items():
            yield f"• **{p_name}**: {p_err}\n"
        yield (
            "\n**Hướng dẫn khắc phục:**\n"
            "1. **Groq / Cloud API**: Nếu chạm giới hạn TPM, đổi model sang `llama-3.1-8b-instant` hoặc kiểm tra kết nối internet / API key\n"
            "2. **LM Studio**: Mở LM Studio → Load model → Bật Server port 8888\n"
            "3. Kiểm tra API key còn hạn sử dụng\n"
            "4. Reload trang dashboard\n"
        )
