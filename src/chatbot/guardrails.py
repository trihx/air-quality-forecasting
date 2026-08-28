"""
LLM Guardrails for PM2.5 AI Assistant.

Provides heuristic-based input validation to:
1. Prevent Prompt Injection and jailbreak attempts.
2. Ensure queries are relevant to the project domain.
"""

import re
import logging

logger = logging.getLogger(__name__)

class ChatGuardrails:
    # Danh sách các từ khóa thường dùng để Prompt Injection / Jailbreak
    PROMPT_INJECTION_PATTERNS = [
        r"ignore previous instructions",
        r"ignore all previous",
        r"bỏ qua (các|những)? chỉ dẫn",
        r"bỏ qua lệnh",
        r"quên (các|những)? (chỉ dẫn|lệnh)",
        r"you are now",
        r"bây giờ bạn là",
        r"system prompt",
        r"developer mode",
        r"dan (do anything now)",
        r"forget everything",
        r"new instructions:",
    ]
    
    # Danh sách các chủ đề liên quan (Positive List)
    RELEVANT_TOPICS = [
        r"pm2\.5", r"air quality", r"không khí", r"thời tiết", r"iot", r"sensor", r"cảm biến",
        r"dự báo", r"forecast", r"predict",
        r"machine learning", r"deep learning", r"ml", r"dl", r"ai", r"model", r"mô hình",
        r"lgbm", r"lightgbm", r"gru", r"lstm", r"tft", r"transformer", r"arima", r"sarimax",
        r"ensemble", r"baseline", r"persistence",
        r"metric", r"mase", r"mae", r"rmse", r"mape", r"r2", r"r bình",
        r"data", r"dữ liệu", r"missing", r"imputation", r"spline", r"knn",
        r"feature", r"đặc trưng", r"lag", r"rolling", r"temporal",
        r"leakage", r"rò rỉ", r"anti-leakage",
        r"luận văn", r"đồ án", r"nghiên cứu", r"thạc sĩ", r"ctu", r"đại học cần thơ",
        r"thesis", r"dashboard", r"ứng dụng", r"workflow", r"quy trình",
        r"shap", r"explainability", r"giải thích",
        r"chào", r"hello", r"hi", r"giúp", r"tên gì", r"ai (tạo|viết)",
    ]
    
    # Danh sách các chủ đề KHÔNG liên quan (Negative List) - Các lĩnh vực hoàn toàn không thuộc scope
    IRRELEVANT_TOPICS = [
        r"nấu ăn", r"nấu món", r"công thức nấu", r"món ăn",
        r"giải trí", r"bài hát", r"phim ảnh", r"ca nhạc", r"chơi game", r"code game",
        r"chính trị", r"tôn giáo", r"bóng đá", r"thể thao",
    ]

    @classmethod
    def _compile_regex(cls, patterns):
        return re.compile("|".join(patterns), re.IGNORECASE)

    _injection_regex = None
    _relevant_regex = None
    _irrelevant_regex = None

    @classmethod
    def get_injection_regex(cls):
        if cls._injection_regex is None:
            cls._injection_regex = cls._compile_regex(cls.PROMPT_INJECTION_PATTERNS)
        return cls._injection_regex

    @classmethod
    def get_relevant_regex(cls):
        if cls._relevant_regex is None:
            cls._relevant_regex = cls._compile_regex(cls.RELEVANT_TOPICS)
        return cls._relevant_regex
        
    @classmethod
    def get_irrelevant_regex(cls):
        if cls._irrelevant_regex is None:
            cls._irrelevant_regex = cls._compile_regex(cls.IRRELEVANT_TOPICS)
        return cls._irrelevant_regex

    @classmethod
    def validate_prompt(cls, prompt: str) -> tuple[bool, str | None]:
        """
        Validate the user prompt against security and relevance rules.
        
        Args:
            prompt: User input string
            
        Returns:
            (is_valid, error_message): (True, None) if valid, (False, "reason") if blocked.
        """
        if not prompt or not prompt.strip():
            return False, "Câu hỏi không hợp lệ."
            
        text_to_check = prompt.lower()
        
        # 1. Check for prompt injection
        if cls.get_injection_regex().search(text_to_check):
            logger.warning(f"Guardrail Blocked: Prompt Injection detected in: {prompt[:50]}...")
            return False, "⚠️ **Guardrail Alert:** Yêu cầu của bạn chứa câu lệnh can thiệp hệ thống (Prompt Injection) nên đã bị từ chối."
            
        # 2. Check for explicit irrelevant topics
        if cls.get_irrelevant_regex().search(text_to_check):
            logger.warning(f"Guardrail Blocked: Irrelevant explicit topic in: {prompt[:50]}...")
            return False, "⚠️ Trợ lý AI này chỉ chuyên hỗ trợ về **Luận văn Dự báo PM2.5 và Machine Learning**. Vui lòng không hỏi các chủ đề ngoài lề (giải trí, ẩm thực, v.v.)."
            
        # 3. Require at least one relevant keyword (Soft check)
        # Bỏ qua kiểm tra độ dài ngắn (chào hỏi)
        if len(text_to_check.split()) > 3:
            if not cls.get_relevant_regex().search(text_to_check):
                logger.info(f"Guardrail Blocked: No relevant keywords found in: {prompt[:50]}...")
                return False, "⚠️ Câu hỏi của bạn dường như không liên quan đến **Dự án Dự báo PM2.5** hoặc **Machine Learning**. Vui lòng đặt câu hỏi đúng chuyên môn dự án."
                
        return True, None
