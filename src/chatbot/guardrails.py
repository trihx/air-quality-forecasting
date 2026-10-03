"""
LLM Guardrails for PM2.5 AI Assistant.

Provides heuristic-based input validation to:
1. Prevent Prompt Injection and jailbreak attempts.
2. Ensure queries are relevant to the project domain.
"""

import logging
import re

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
        r"dan \(?do anything now\)?",
        r"\bdan mode\b",
        r"forget everything",
        r"new instructions:",
    ]

    # Danh sách các chủ đề liên quan (Positive List)
    RELEVANT_TOPICS = [
        # Domain & Vị trí địa lý
        r"pm2\.5",
        r"air quality",
        r"không khí",
        r"thời tiết",
        r"khí tượng",
        r"nhiệt độ",
        r"độ ẩm",
        r"gió",
        r"áp suất",
        r"iot",
        r"sensor",
        r"cảm biến",
        r"sa đéc",
        r"đồng tháp",
        # Dự báo & Machine Learning cơ bản
        r"dự báo",
        r"forecast",
        r"predict",
        r"machine learning",
        r"deep learning",
        r"ml",
        r"dl",
        r"ai",
        r"model",
        r"mô hình",
        # Mô hình cụ thể
        r"lgbm",
        r"lightgbm",
        r"gru",
        r"lstm",
        r"tft",
        r"transformer",
        r"arima",
        r"sarimax",
        r"ensemble",
        r"baseline",
        r"persistence",
        # Thống kê & Kiểm định chuỗi thời gian
        r"chuỗi thời gian",
        r"time series",
        r"iqr",
        r"outlier",
        r"ngoại lai",
        r"tính dừng",
        r"stationarity",
        r"adf",
        r"kpss",
        r"s-esd",
        r"shapiro",
        r"ljung-box",
        r"autocorrelation",
        r"tự tương quan",
        r"acf",
        r"pacf",
        r"diurnal",
        r"chu kỳ",
        r"resample",
        r"resampling",
        r"shuffle",
        # Metrics & Đánh giá
        r"đánh giá",
        r"horizon",
        r"horizons",
        r"metric",
        r"mase",
        r"mae",
        r"rmse",
        r"mape",
        r"r2",
        r"r bình",
        r"r2 âm",
        r"out-of-sample",
        r"ngoài mẫu",
        r"f1-score",
        r"f1",
        r"recall",
        r"độ nhạy",
        r"diebold-mariano",
        r"dm test",
        r"dấu âm",
        r"dấu dương",
        r"cqr",
        r"aci",
        r"conformal",
        r"mức danh định",
        r"prediction interval",
        r"khoảng tin cậy",
        r"khoảng dự báo",
        r"độ bao phủ",
        r"coverage",
        r"pareto",
        r"ablation",
        r"bóc tách",
        r"tipping point",
        r"điểm bùng phát",
        r"14–17",
        r"14-17",
        r"19\.810",
        r"19810",
        r"ngưỡng who",
        # Dữ liệu & Tiền xử lý
        r"data",
        r"dữ liệu",
        r"missing",
        r"imputation",
        r"nội suy",
        r"spline",
        r"knn",
        r"feature",
        r"đặc trưng",
        r"lag",
        r"rolling",
        r"temporal",
        r"split",
        r"train",
        r"validation",
        r"test",
        r"leakage",
        r"rò rỉ",
        r"anti-leakage",
        r"chu kỳ mùa",
        # Đề án & Nghiên cứu & Phản biện Hội đồng
        r"luận văn",
        r"đề án",
        r"đồ án",
        r"nghiên cứu",
        r"thạc sĩ",
        r"bảo vệ luận văn",
        r"grill-me",
        r"chất vấn",
        r"phản biện",
        r"hole",
        r"ctu",
        r"đại học cần thơ",
        r"thesis",
        r"dashboard",
        r"ứng dụng",
        r"workflow",
        r"quy trình",
        r"pipeline",
        r"shap",
        r"explainability",
        r"giải thích",
        r"bài học",
        r"kinh nghiệm",
        r"khắc phục",
        r"hạn chế",
        r"giải pháp",
        r"nhược điểm",
        r"kohler",
        r"neural ode",
        r"alarm fatigue",
        r"error floor",
        r"lỗi",
        # Hội thoại tự nhiên cơ bản
        r"chào",
        r"hello",
        r"hi",
        r"giúp",
        r"tên gì",
        r"ai (tạo|viết)",
    ]

    # Danh sách các chủ đề KHÔNG liên quan (Negative List) - Các lĩnh vực hoàn toàn không thuộc scope
    IRRELEVANT_TOPICS = [
        r"nấu ăn",
        r"nấu món",
        r"công thức nấu",
        r"món ăn",
        r"giải trí",
        r"bài hát",
        r"phim ảnh",
        r"ca nhạc",
        r"chơi game",
        r"code game",
        r"chính trị",
        r"tôn giáo",
        r"bóng đá",
        r"thể thao",
    ]

    @classmethod
    def _compile_regex(cls, patterns):
        return re.compile("|".join(patterns), re.IGNORECASE)

    CREDENTIAL_EXTRACTION_PATTERNS = [
        r"(?:cho tôi|xin|tiết lộ|show|give|tell|reveal|what is|lấy)\b.*?\b(?:api[\s_-]?key|mã pin|puk|password|mật khẩu|token bí mật|cloudflare tunnel|link cloudflare|tunnel url|credentials)",
        r"\b(?:api[\s_-]?key|mã pin|puk|password|mật khẩu|link cloudflare|tunnel url|credentials)\b.*?\b(?:của bạn|của hệ thống|ở đâu|là gì)\b",
        r"\b(?:what is the system pin|show me your api key|reveal credentials)\b",
    ]

    _injection_regex = None
    _extraction_regex = None
    _relevant_regex = None
    _irrelevant_regex = None

    @classmethod
    def reset_cache(cls):
        """Reset compiled regex caches."""
        cls._injection_regex = None
        cls._extraction_regex = None
        cls._relevant_regex = None
        cls._irrelevant_regex = None

    @classmethod
    def get_injection_regex(cls):
        if cls._injection_regex is None:
            cls._injection_regex = cls._compile_regex(cls.PROMPT_INJECTION_PATTERNS)
        return cls._injection_regex

    @classmethod
    def get_extraction_regex(cls):
        if cls._extraction_regex is None:
            cls._extraction_regex = cls._compile_regex(cls.CREDENTIAL_EXTRACTION_PATTERNS)
        return cls._extraction_regex

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

        # 0. Check for credential extraction attempts
        if cls.get_extraction_regex().search(text_to_check):
            logger.warning(f"Guardrail Blocked: Credential extraction detected in: {prompt[:50]}...")
            return (
                False,
                "⚠️ **Guardrail Security Alert:** Không thể cung cấp hoặc tiết lộ thông tin xác thực, API Key, mã PIN/PUK hoặc đường dẫn nội bộ của hệ thống.",
            )

        # 1. Check for prompt injection
        if cls.get_injection_regex().search(text_to_check):
            logger.warning(f"Guardrail Blocked: Prompt Injection detected in: {prompt[:50]}...")
            return (
                False,
                "⚠️ **Guardrail Alert:** Yêu cầu của bạn chứa câu lệnh can thiệp hệ thống (Prompt Injection) nên đã bị từ chối.",
            )

        # 2. Check for explicit irrelevant topics
        if cls.get_irrelevant_regex().search(text_to_check):
            logger.warning(f"Guardrail Blocked: Irrelevant explicit topic in: {prompt[:50]}...")
            return (
                False,
                "⚠️ Trợ lý AI này chỉ chuyên hỗ trợ về **Đề án Dự báo PM2.5 và Machine Learning**. Vui lòng không hỏi các chủ đề ngoài lề (giải trí, ẩm thực, v.v.).",
            )

        # 3. Require at least one relevant keyword (Soft check)
        # Bỏ qua kiểm tra độ dài ngắn (chào hỏi)
        if len(text_to_check.split()) > 3 and not cls.get_relevant_regex().search(text_to_check):
            logger.info(f"Guardrail Blocked: No relevant keywords found in: {prompt[:50]}...")
            return (
                False,
                "⚠️ Câu hỏi của bạn dường như không liên quan đến **Dự án Dự báo PM2.5** hoặc **Machine Learning**. Vui lòng đặt câu hỏi đúng chuyên môn dự án.",
            )

        return True, None
