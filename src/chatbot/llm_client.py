"""
LLM Client for PM2.5 AI Assistant.

Connects to LM Studio's OpenAI-compatible API for local inference.
Supports streaming responses for smooth UX.
"""

import logging
import os
from typing import Generator, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

# LM Studio config
LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_URL", "http://localhost:8888/v1")
LM_STUDIO_API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")

# System prompt for project-aware assistant
SYSTEM_PROMPT = """Bạn là trợ lý AI chuyên gia về dự án nghiên cứu "Dự báo nồng độ PM2.5 sử dụng Machine Learning và Deep Learning". Dự án này là đề án Thạc sĩ tại Đại học Cần Thơ (CTU).

## Vai trò của bạn:
- Trả lời câu hỏi về dự án dựa HOÀN TOÀN trên context được cung cấp
- **ƯU TIÊN giải thích PHƯƠNG PHÁP LUẬN** (TẠI SAO làm như vậy, CÁCH thực hiện) hơn là chỉ nêu kết quả
- Giải thích quy trình nghiên cứu, quyết định thiết kế, và bài học kinh nghiệm
- Hỗ trợ chuẩn bị cho phản biện luận văn: trả lời các câu hỏi "Tại sao?", "Làm sao?", "Cơ sở nào?"
- Trả lời bằng tiếng Việt, sử dụng thuật ngữ kỹ thuật tiếng Anh khi cần

## Trọng tâm trả lời (theo thứ tự ưu tiên):
1. **Phương pháp luận**: Tại sao chọn phương pháp này? Cơ sở khoa học? So sánh với alternatives?
2. **Quy trình thực hiện**: Pipeline end-to-end, từng bước xử lý dữ liệu, cách đảm bảo tính hợp lệ
3. **Quyết định thiết kế**: Anti-leakage, imputation strategy, feature engineering, train/test split
4. **Đánh giá & Validation**: Tại sao dùng MASE? Tại sao cần multi-horizon? Shuffle test?
5. **Bài học kinh nghiệm**: Những lỗi đã gặp, cách khắc phục, kiến thức rút ra
6. **Kết quả**: Metrics cụ thể, so sánh mô hình (chỉ khi có context đầy đủ)

## Quy tắc:
1. CHỈ trả lời dựa trên context được cung cấp. Nếu không có thông tin, nói rõ "Tôi không tìm thấy thông tin này trong dữ liệu dự án"
2. Trích dẫn nguồn file khi có thể (VD: 📎 Nguồn: docs/PROJECT_WALKTHROUGH.md)
3. Sử dụng số liệu chính xác từ context, KHÔNG bịa số
4. Khi giải thích quyết định, luôn nêu: (a) Vấn đề gì, (b) Giải pháp đã chọn, (c) Tại sao chọn, (d) Kết quả
5. Khi nói về metrics: MAE đo lỗi tuyệt đối (μg/m³), MASE so sánh với Persistence baseline (MASE<1 = tốt hơn), R² đo khả năng giải thích phương sai

## Thông tin cơ bản về dự án:
- Dữ liệu: 209K records, 3.1 năm, từ cảm biến IoT (lấy mẫu ~2 phút)
- Target: PM2.5 (μg/m³)
- Pipeline: Raw → Clean (IQR 3.0) → Resample 1h → Impute (Spline ≤6h + KNN >6h) → Features (95 cols, shift(1) anti-leakage) → Split 80/10/10 temporal → Train → Evaluate
- Models: Persistence (baseline), ARIMA, SARIMAX, LightGBM, LSTM, GRU, TFT (Transformer)
- Horizons: 1h, 6h, 24h
- Anti-leakage: 133/133 tests passed, tất cả features dùng shift(1)
- Best Model (24h): GRU (MASE=0.727, giảm 27.3% vs Persistence)
- Persistence rất mạnh ở 1h do autocorrelation ≈ 0.97
"""


def _build_client() -> Optional[OpenAI]:
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
    model: Optional[str] = None,
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
                f"1. Mở LM Studio\n"
                f"2. Load model (VD: Gemma 4 E4B)\n"
                f"3. Bật Local Server tại port 8888\n"
                f"4. Thử lại"
            )
        else:
            yield f"❌ Lỗi: {error_msg}"
