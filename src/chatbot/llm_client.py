"""
LLM Client for PM2.5 AI Assistant.

Connects to LM Studio's OpenAI-compatible API for local inference.
Supports streaming responses for smooth UX.
"""

import logging
import os
from collections.abc import Generator

from openai import OpenAI

logger = logging.getLogger(__name__)

# LM Studio config
LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_URL", "http://localhost:8888/v1")
LM_STUDIO_API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")

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

## Thông tin cơ bản:
- Dữ liệu: 209K records, 3.1 năm, IoT (~2 phút/lần)
- Target: PM2.5 (µg/m³)
- Pipeline: Raw→Clean(IQR 3.0)→Resample 1h→Impute→Features(95)→Eval
- Models: Persistence, ARIMA, SARIMAX, LightGBM, LSTM, GRU, TFT
- Horizons: 1h, 6h, 24h | Tests: 133/133 passed
- Best (24h): GRU MASE=0.727 (-27.3% vs Persistence)
- Persistence rất mạnh ở 1h (autocorrelation ≈ 0.97)
"""


def _build_client() -> OpenAI | None:
    """Create OpenAI client pointing to LM Studio."""
    try:
        client = OpenAI(
            base_url=LM_STUDIO_BASE_URL,
            api_key=LM_STUDIO_API_KEY,
            timeout=60.0,
        )
        return client
    except Exception as e:
        logger.error(f"Cannot create LM Studio client: {e}")
        return None


def check_connection() -> bool:
    """Check if LM Studio server is reachable."""
    client = _build_client()
    if not client:
        return False
    try:
        client.models.list()
        return True
    except Exception:
        return False


def get_available_models() -> list[str]:
    """List models loaded in LM Studio."""
    client = _build_client()
    if not client:
        return []
    try:
        models = client.models.list()
        return [m.id for m in models.data] if models.data else []
    except Exception:
        return []


def chat_stream(
    messages: list[dict],
    context: str = "",
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> Generator[str, None, None]:
    """
    Stream chat completion from LM Studio.

    Args:
        messages: Chat history [{role, content}, ...]
        context: RAG context to inject
        model: Model identifier (None = use default loaded model)
        temperature: Creativity (0.0-1.0, lower = more precise)
        max_tokens: Max response length

    Yields:
        Response text chunks for streaming
    """
    client = _build_client()
    if not client:
        yield "❌ Không thể kết nối LM Studio. Vui lòng kiểm tra LM Studio đang chạy."
        return

    # Build system message with RAG context
    system_content = SYSTEM_PROMPT
    if context:
        system_content += f"\n\n## Context từ dữ liệu dự án:\n{context}"

    full_messages = [{"role": "system", "content": system_content}] + messages

    try:
        # Auto-detect model if not specified
        if not model:
            try:
                models_list = client.models.list()
                if models_list.data:
                    model = models_list.data[0].id
                else:
                    yield "❌ Chưa load model trong LM Studio. Vui lòng load model trước."
                    return
            except Exception:
                model = "local-model"

        stream = client.chat.completions.create(
            model=model or "local-model",
            messages=full_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        error_msg = str(e)
        if "Connection" in error_msg or "refused" in error_msg:
            yield (
                "❌ Không thể kết nối đến LM Studio.\n\n"
                "**Hướng dẫn:**\n"
                "1. Mở LM Studio\n"
                "2. Load model (VD: Gemma 4 E4B)\n"
                "3. Bật Local Server tại port 8888\n"
                "4. Thử lại"
            )
        else:
            yield f"❌ Lỗi: {error_msg}"
