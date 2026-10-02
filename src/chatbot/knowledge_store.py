"""
Zero-Dependency Curated Knowledge Store for PM2.5 Project AI Assistant.

Provides authoritative, pre-compiled domain knowledge extracted from the
Obsidian Second Brain vault (`knowledge_vault/`).
Operates with ZERO external dependencies (no PyTorch, no ChromaDB, no SentenceTransformers),
uses 0 MB RAM, and executes queries in <1ms.
Safe for Render 512MB RAM and serverless environments.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any


def _strip_accents(text: str) -> str:
    """Remove Vietnamese accents and lowercase for fuzzy zero-dep matching."""
    text = text.lower()
    text = text.replace("²", "2").replace("µ", "u")
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.replace("đ", "d").replace("Đ", "d")
    return text


def _tokenize(text: str) -> list[str]:
    """Tokenize text into alphanumeric words."""
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    return [w for w in cleaned.split() if len(w) >= 2]


@dataclass
class KnowledgeDocument:
    """Represents a curated knowledge document in the store."""

    doc_id: str
    title: str
    category: str
    keywords: list[str]
    content: str
    citations: list[str] = field(default_factory=list)
    source: str = ""
    _norm_title: str = field(init=False, default="")
    _norm_content: str = field(init=False, default="")
    _norm_keywords: list[str] = field(init=False, default_factory=list)
    _title_tokens: set[str] = field(init=False, default_factory=set)
    _content_tokens: list[str] = field(init=False, default_factory=list)
    _content_token_set: set[str] = field(init=False, default_factory=set)
    _kw_token_sets: list[set[str]] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self._norm_title = _strip_accents(self.title)
        self._norm_content = _strip_accents(self.content)
        self._norm_keywords = [_strip_accents(kw) for kw in self.keywords]
        self._title_tokens = set(_tokenize(self._norm_title))
        self._content_tokens = _tokenize(self._norm_content)
        self._content_token_set = set(self._content_tokens)
        self._kw_token_sets = [set(_tokenize(k)) for k in self._norm_keywords]


# ═══════════════════════════════════════════════════════════════════════════
# CURATED KNOWLEDGE DOCUMENTS (Second Brain Core)
# ═══════════════════════════════════════════════════════════════════════════

CURATED_DOCUMENTS: list[KnowledgeDocument] = [
    KnowledgeDocument(
        doc_id="pipeline_7_steps",
        title="Quy trình nghiên cứu & Kỹ nghệ dữ liệu 7 bước (End-to-End Pipeline)",
        category="data_engineering",
        keywords=[
            "quy trinh pipeline",
            "pipeline 7 buoc",
            "quy trình 7 bước",
            "end-to-end",
            "sankey",
            "209594",
            "38 thang",
            "hinh 1.1",
            "quy trinh",
            "cac buoc",
            "tong quan",
        ],
        source="knowledge_vault/01_Pipeline_and_Data_Engineering/01_Pipeline_7_Steps.md",
        citations=["Hình 1.1 Đề án", "Mục 3.1 & 3.2 Báo cáo ThS", "QĐ 1799/QĐ-ĐHCT"],
        content=(
            "### Quy trình nghiên cứu và kỹ nghệ dữ liệu 7 bước (Hình 1.1):\n\n"
            "1. **Thu thập dữ liệu IoT thô:** 209.594 bản ghi đo từ trạm quan trắc IoT ngoài trời tại TP. Sa Đéc, "
            "Đồng Tháp trong 38 tháng (16/03/2022 – 11/05/2025, 1.152 ngày), tần suất thô ~2 phút/lần.\n"
            "2. **Làm sạch & Kiểm định Domain Bounds:** Loại bỏ bản ghi lỗi; áp dụng Domain Bounds [0, 500] µg/m³ "
            "theo chuẩn WHO AQI; thuật toán S-ESD lọc lỗi phần cứng (flatline).\n"
            "3. **Tái lấy mẫu đa độ phân giải:** 15 phút (bắt xung phát thải), 30 phút (điểm ngọt Pareto), và 1 giờ "
            "(đối chuẩn trạm quốc gia) theo chuẩn WMO (≥75% mẫu hợp lệ).\n"
            "4. **Nội suy phục hồi phân tầng (Tiered Imputation):** Gap ≤6h dùng Akima/PCHIP Spline; Gap 6-24h dùng "
            "KNN (k=5, chỉ lấy donors quá khứ); Gap >24h dũng cảm loại bỏ (Drop 19.810 giờ khuyết dài) chống sinh ảo giác.\n"
            "5. **Kho 119 đặc trưng & Kỷ luật anti-leakage:** 40 lag, 36 rolling stats, 16 EWMA, 12 Fourier, "
            "11 calendar, 4 domain interactions. 100% biến trượt/sai phân bắt buộc áp dụng shift(1).\n"
            "6. **Phân tách mỏ neo 80:10:10:** Train 80% (03/2022-09/2024), Val 10% (10/2024-12/2024, Optuna TPE), "
            "Anchor Test 10% (01/2025-05/2025: 669h ở 1h, 863 mẫu ở 30m, 1.836 mẫu ở 15m). Test on Real Only (is_imputed==0).\n"
            "7. **Huấn luyện 11 mô hình & Đánh giá thống kê:** Tiêu chuẩn vàng MASE thay thế RMSE; kiểm định Diebold-Mariano "
            "với hiệu chỉnh HLN; định lượng độ bất định CQR + ACI 90.2%; giải thích mô hình bằng Tree SHAP & Permutation."
        ),
    ),
    KnowledgeDocument(
        doc_id="anti_leakage",
        title="Kỷ luật shift(1) chống rò rỉ dữ liệu & Vụ việc bóc tách R²=1.000 ảo",
        category="data_integrity",
        keywords=[
            "anti leakage",
            "anti-leakage",
            "shift 1",
            "shift(1)",
            "ro ri du lieu",
            "r2 1.0",
            "r2=1.000",
            "lookahead bias",
            "ro ri",
            "ky luat",
            "ao tuong",
        ],
        source="knowledge_vault/02_Data_Integrity_and_Anti_Leakage/01_Anti_Leakage_Shift1_Discipline.md",
        citations=["Mục 3.3 Báo cáo ThS", "51 Anti-Leakage Automated Tests", "Playbook"],
        content=(
            "### Kỷ luật shift(1) chống rò rỉ dữ liệu (Anti-Leakage Discipline):\n\n"
            "- **Vụ việc bóc tách R²=1.000 ảo:** Trong giai đoạn thử nghiệm ban đầu, mô hình hồi quy đạt R²=1.000 hoàn hảo. "
            "Truy vết phát hiện biến rolling mean được tính trực tiếp từ df['pm25'].rolling(6).mean(), vô tình đưa giá trị y_t "
            "vào đặc trưng đầu vào để dự báo chính nó (Lookahead Bias).\n"
            "- **Kỷ luật bất biến shift(1):** 100% các biến trễ (lag), biến cửa sổ trượt (rolling mean/std/min/max), EWMA và "
            "sai phân (diff/pct_change) BẮT BUỘC gọi .shift(1) trước khi tính toán: `df['pm25'].shift(1).rolling(w).mean()`.\n"
            "- **Kết quả thực tế:** R² ảo 1.000 sụp đổ về giá trị thực 0.267, buộc mô hình phải học quy luật vật lý thực chất.\n"
            "- **5 nguyên tắc chống rò rỉ:** (1) Shift(1) bắt buộc; (2) Scaler chỉ fit trên Train; (3) KNN donor chỉ lấy quá khứ t'<t; "
            "(4) Chia tập thời gian tuyến tính không xáo trộn; (5) Bộ 51 bài test tự động anti-leakage giám sát liên tục."
        ),
    ),
    KnowledgeDocument(
        doc_id="outlier_trap",
        title="Bẫy xóa ngoại lai IQR 3.0 & Bảo tồn đỉnh ô nhiễm bằng Domain Bounds",
        category="data_engineering",
        keywords=[
            "outlier",
            "bay ngoai lai",
            "iqr",
            "iqr 3.0",
            "bay iqr 3.0",
            "bay iqr",
            "xoa nham",
            "xoa nham 66 dinh",
            "66 dinh",
            "domain bounds",
            "domain bounds [0, 500]",
            "dinh o nhiem",
            "fat tailed",
            "fat-tailed",
            "skewness",
            "kurtosis",
            "who aqi",
            "500",
            "s-esd",
            "flatline",
            "f1-score",
            "f1 0.782",
            "ao tuong chinh xac",
        ],
        source="knowledge_vault/01_Pipeline_and_Data_Engineering/03_Outlier_Trap_and_Domain_Bounds.md",
        citations=["Mục 3.2 Báo cáo ThS", "Ablation Study v10 vs v9", "WHO AQI Guidelines"],
        content=(
            "### Bẫy xóa ngoại lai IQR 3.0 & Domain Bounds [0, 500] µg/m³:\n\n"
            "- **Bản chất bẫy IQR 3.0 cổ điển:** Phương pháp Tukey IQR truyền thống (Q3 + 3.0*IQR ~ 54 µg/m³) "
            "đã gọt nhầm 66 đỉnh ô nhiễm thực tế (55 – 120 µg/m³). Việc này tạo ra ảo tưởng chính xác giả tạo (False Sense of Accuracy): "
            "RMSE trên validation giảm đẹp giả tạo (2.12 µg/m³), nhưng khi ra môi trường thật mô hình hoàn toàn mù trước ô nhiễm nặng, "
            "khiến F1-score cảnh báo sớm sụt giảm nghiêm trọng xuống < 0.35.\n"
            "- **Bản chất phân phối Fat-Tailed của PM2.5:** Chuỗi PM2.5 Sa Đéc có Skewness = 2.0046 (lệch phải mạnh) và Kurtosis = 6.1458 (đuôi dày, đỉnh nhọn). "
            "Các đỉnh nồng độ cao là biến cố vi khí hậu có thật (nghịch nhiệt bề mặt, đốt rơm rạ vụ mùa), không phải nhiễu ngoại lai.\n"
            "- **Giải pháp thay thế Domain Bounds [0, 500] & S-ESD:**\n"
            "  Đề án loại bỏ IQR 3.0, chuyển sang Domain Bounds [0, 500] µg/m³ theo dải đo vật lý WHO AQI, kết hợp thuật toán S-ESD "
            "chỉ lọc các lỗi phần cứng cảm biến (như hiện tượng flatline kẹt giá trị liên tục). Nhờ bảo tồn 66 đỉnh thật, mô hình khôi phục "
            "F1-score cảnh báo sớm lên mức xuất sắc 0.782."
        ),
    ),
    KnowledgeDocument(
        doc_id="tiered_imputation",
        title="Chiến lược phục hồi dữ liệu phân tầng & Quyết định dũng cảm Drop 19.810 giờ",
        category="data_engineering",
        keywords=[
            "missing data",
            "khuyet thieu",
            "noi suy",
            "imputation",
            "phan tang",
            "tiered imputation",
            "spline",
            "knn",
            "drop 19810h",
            "drop 19.810",
            "drop 19.810 gio",
            "dung cam drop",
            "tai sao drop 19810",
            "gioi han 24h",
            "mnar",
            "data hallucination",
            "ve du lieu ao",
            "gan",
            "mice",
            "74%",
            "do thuc",
            "liem chinh nghien cuu",
        ],
        source="knowledge_vault/01_Pipeline_and_Data_Engineering/02_Tiered_Imputation_and_Data_Sparsity.md",
        citations=["Bảng 3.4 Báo cáo ThS", "Akima (1970)", "Mục 3.2 Luận văn"],
        content=(
            "### Chiến lược phục hồi dữ liệu phân tầng & Quyết định dũng cảm Drop 19.810 giờ:\n\n"
            "- **Thực trạng dữ liệu IoT Sa Đéc:** Tỷ lệ khuyết thiếu tích lũy lên tới 74% qua 38 tháng (209.594 bản ghi thô, "
            "do cúp điện, bảo trì trạm, nghẽn đường truyền 4G).\n"
            "- **Chiến lược 3 tầng xử lý & Giới hạn phục hồi 24h:**\n"
            "  + **Tầng 1 (Gap ≤ 6 giờ):** Dùng PCHIP/Akima Spline bảo toàn tính đơn điệu cục bộ, chống vọt lố (overshooting) qua các đỉnh dốc.\n"
            "  + **Tầng 2 (6h < Gap ≤ 24 giờ):** Dùng KNN Imputation (k=5), tuân thủ nghiêm ngặt donors chỉ lấy trong quá khứ (t' < t) chống rò rỉ.\n"
            "  + **Tầng 3 (Gap > 24 giờ): DŨNG CẢM LOẠI BỎ 19.810 giờ khuyết dài.** Thực nghiệm chứng minh: vượt quá 24h, sai số phục hồi bùng nổ "
            "(RMSE > 12 µg/m³, vượt quá 100% nồng độ trung bình nền), làm biến dạng hoàn toàn tương quan động học.\n"
            "- **Bản chất MNAR & Chống ảo giác dữ liệu (Anti-Data Hallucination):**\n"
            "  Dữ liệu khuyết dài ngày mang bản chất MNAR (Missing Not At Random — trạm mất điện kéo dài trong mùa bão hoặc hư hỏng linh kiện). "
            "Đề án dứt khoát KHÔNG dùng các mô hình sinh như GAN, VAE hay MICE để tự vẽ thêm dữ liệu giả mạo. Việc loại bỏ 19.810 giờ thể hiện "
            "kỷ luật liêm chính khoa học: thà cắt chuỗi thành các phân đoạn sạch có ý nghĩa vật lý (18.355 mẫu ở 15m, 8.625 mẫu ở 30m, 6.689 mẫu ở 1h) "
            "còn hơn huấn luyện mô hình trên dữ liệu ảo tạo."
        ),
    ),
    KnowledgeDocument(
        doc_id="multi_resolution",
        title="Khung đa độ phân giải (15m, 30m, 1h) & Điểm ngọt Pareto 30 phút",
        category="data_engineering",
        keywords=[
            "da do phan giai",
            "multi resolution",
            "multi-resolution",
            "15m",
            "30m",
            "1h",
            "diem ngot",
            "pareto",
            "sweet spot",
            "10/15",
            "can bang",
        ],
        source="knowledge_vault/01_Pipeline_and_Data_Engineering/04_Multi_Resolution_Framing.md",
        citations=["Mục 3.1 & 4.2 Báo cáo ThS", "Bảng 4.3 Đề án"],
        content=(
            "### Khung đa độ phân giải & Điểm ngọt Pareto 30 phút:\n\n"
            "- **Động lực:** Chuỗi 1h bị bẫy tự tương quan r=0.86; chuỗi 2-5 phút thô chứa nhiều nhiễu vi cơ học. "
            "Đề án xây dựng khung 3 độ phân giải đồng thời: 15m, 30m, 1h.\n"
            "- **Điểm ngọt Pareto 30 phút:** Chiếm 10/15 vị trí trong top-5 mô hình xuất sắc nhất toàn đề án. "
            "Đạt tỷ lệ tín hiệu trên nhiễu (SNR) tối ưu; Weighted Ensemble 30m đạt MASE=0.382 ở 6h và 0.469 ở 24h.\n"
            "- **Vai trò độ phân giải 15m:** Bắt trọn vi gia tốc biến thiên hạt bụi; mạng GRU 15m đạt MASE=0.667 tại tầm 1h "
            "(tương ứng 4 bước 15m), phá vỡ hoàn toàn bẫy tự tương quan của dữ liệu chuỗi giờ."
        ),
    ),
    KnowledgeDocument(
        doc_id="autocorrelation_trap",
        title="Bẫy tự tương quan (Autocorrelation Trap) & Đột phá GRU 15 phút",
        category="models",
        keywords=[
            "bay tu tuong quan",
            "autocorrelation trap",
            "persistence trap",
            "acf",
            "r=0.86",
            "r=0.97",
            "gru 15m",
            "mase=0.667",
            "pha bay",
            "tre pha",
            "phase lag",
            "r 0.86",
            "0.86",
            "buoc 1h",
            "pha bay tu tuong quan",
        ],
        source="knowledge_vault/04_Models_and_Architectures/02_Autocorrelation_Trap_and_GRU_15m.md",
        citations=["Mục 4.1 & 4.3 Báo cáo ThS", "Bảng PL.3.1"],
        content=(
            "### Bẫy tự tương quan 1h & Đột phá GRU 15m phá bẫy:\n\n"
            "- **Bản chất bẫy tự tương quan:** Tại chuỗi 1h, hệ số tự tương quan bậc 1 đạt r = 0.86 (chuỗi thô 15m đạt r = 0.97). "
            "Nồng độ bụi giờ hiện tại gần như bằng giờ trước, khiến baseline ngây thơ Persistence (y_{t+h} = y_t) cực kỳ khó đánh bại. "
            "Nhiều mô hình ML phức tạp khi chạy ở chuỗi giờ bị trễ pha (phase lag), đạt MASE > 1.0 (kém hơn đoán ngây thơ).\n"
            "- **Đột phá GRU 15 phút:** Tại độ phân giải 15m, tầm 1h tương ứng 4 bước (h=4). Mạng GRU với Update Gate và Reset Gate "
            "học được đạo hàm vi phân và gia tốc tích tụ hạt bụi trong 4 nhịp 15 phút, đạt MASE = 0.667 (đánh bại Persistence 33.3%)."
        ),
    ),
    KnowledgeDocument(
        doc_id="benchmark_models",
        title="Tổng quan 11 kiến trúc mô hình đối chuẩn & Cơ chế GRU vs LightGBM vs TFT",
        category="models",
        keywords=[
            "11 mo hinh",
            "benchmark models",
            "gru",
            "lightgbm",
            "tft",
            "temporal fusion transformer",
            "lstm",
            "arima",
            "sarimax",
            "so sanh mo hinh",
            "uu nhuoc diem",
            "co che",
            "gru 15m vs lightgbm vs tft",
            "l1 loss",
            "regression_l1",
        ],
        source="knowledge_vault/04_Models_and_Architectures/01_Benchmark_Models_Overview.md",
        citations=["Mục 4.1 & 4.2 Báo cáo ThS", "Bảng 4.2 Đề án"],
        content=(
            "### Tổng quan 11 kiến trúc mô hình đối chuẩn (Benchmark Architectures):\n\n"
            "- **4 trường phái đối chuẩn:**\n"
            "  1. *Cơ sở (Baselines):* Persistence Naive (rất mạnh ở 1h) và Moving Average.\n"
            "  2. *Thống kê cổ điển:* ARIMA (2,0,1) và SARIMAX (thêm khí tượng ngoại sinh).\n"
            "  3. *Học máy dạng bảng:* ElasticNet, Random Forest (200 cây), LightGBM (L1 loss regression_l1).\n"
            "  4. *Học sâu chuỗi thời gian:* LSTM (cổng nhớ tế bào), GRU (cổng tối giản, vi phân nhịp 15m), "
            "Temporal Fusion Transformer (TFT - Gated Residual Networks & Multi-Head Attention).\n"
            "  5. *Học kết hợp:* Weighted Optimization Ensemble (Mô hình quán quân toàn diện).\n"
            "- **So sánh cơ chế GRU 15m vs LightGBM vs TFT:**\n"
            "  + **GRU 15m:** Tinh gọn, học liên tục quán tính và gia tốc hạt bụi qua 4 bước 15m, phá vỡ bẫy tự tương quan ở 1h (MASE=0.667).\n"
            "  + **LightGBM:** Cực nhanh, chia nhánh cây phi tuyến theo các ngưỡng khí tượng (gió, độ ẩm), tối ưu L1 loss chống outlier.\n"
            "  + **TFT Transformer:** Cơ chế tự chú ý đa đầu nắm bắt phụ thuộc dài hạn và chọn lọc biến tự động, nhưng chi phí tính toán cao."
        ),
    ),
    KnowledgeDocument(
        doc_id="sweet_spot_ensemble",
        title="Sự thống trị của Ensemble Weighted tại điểm ngọt 30 phút ở tầm xa 6h và 24h",
        category="models",
        keywords=[
            "ensemble weighted",
            "mo hinh quan quan",
            "thong tri 6h 24h",
            "weighted ensemble",
            "tam xa 6h 24h",
            "error diversity",
            "bu tru sai so",
            "mase 0.382",
            "mase 0.469",
            "diem ngot 30 phut",
            "tai sao ensemble thong tri",
        ],
        source="knowledge_vault/04_Models_and_Architectures/03_Sweet_Spot_30m_Ensemble.md",
        citations=["Mục 4.3 Báo cáo ThS", "Bảng 4.3 Đề án"],
        content=(
            "### Sự thống trị của Mô hình Quán quân Weighted Optimization Ensemble tại 30m:\n\n"
            "- **Cơ chế tối ưu hóa trọng số lồi:** min_w sum |y_t - sum(w_i * y_hat_{i,t})| với sum(w_i)=1, w_i >= 0 "
            "trên tập Validation, phối hợp 3 trường phái: LightGBM (ngưỡng phi tuyến), GRU (quán tính chuỗi), Random Forest (làm mượt).\n"
            "- **Bảng thành tích vượt trội tại 30 phút:**\n"
            "  + Tầm 1h (h=1 30m): MASE = 0.712 (cải thiện +28.8% so với baseline).\n"
            "  + Tầm 6h (h=6 30m): MASE = 0.382 (cải thiện +25.5%, điểm tối ưu nhất toàn đề án).\n"
            "  + Tầm 24h (h=24 30m): MASE = 0.469 (cải thiện +15.0%).\n"
            "- **Tại sao Ensemble thống trị ở tầm xa 6h và 24h?**\n"
            "  1. *Bù trừ sai số đa dạng (Error Diversity):* Sai số phân tán của cây LightGBM và sai số làm mượt của GRU triệt tiêu lẫn nhau, "
            "hạ thấp đáng kể phương sai sai số tổng thể khi bước dự báo tăng xa.\n"
            "  2. *Điểm ngọt Pareto 30 phút:* Tỷ lệ tín hiệu trên nhiễu (SNR) tối ưu giúp bộ trọng số cân bằng hoàn hảo giữa thích ứng và ổn định."
        ),
    ),
    KnowledgeDocument(
        doc_id="optuna_tuning",
        title="Quy trình tối ưu siêu tham số Hyperparameter Tuning bằng Optuna TPE 50 trials",
        category="models",
        keywords=[
            "optuna",
            "tpe",
            "50 trials",
            "hyperparameter tuning",
            "toi uu sieu tham so",
            "validation set",
            "early stopping",
            "patience 10",
            "quy trinh toi uu",
            "loss l1",
            "mae",
        ],
        source="knowledge_vault/04_Models_and_Architectures/04_Hyperparameter_Tuning_Optuna.md",
        citations=["Mục 3.4 Báo cáo ThS", "Optuna Akiba (2019)"],
        content=(
            "### Quy trình tối ưu siêu tham số Hyperparameter Tuning bằng Optuna TPE 50 trials:\n\n"
            "- **Phương pháp luận:** Áp dụng thuật toán Tree-structured Parzen Estimator (TPE) 50 trials độc lập cho mỗi mô hình.\n"
            "- **Kỷ luật dữ liệu nghiêm ngặt:** Chỉ tối ưu trên Tập Xác Thực (Validation Set 10%, 10/2024 - 12/2024), "
            "tuyệt đối KHÔNG chạm vào tập Test. Hàm mục tiêu tối thiểu hóa MAE để đồng bộ với MASE.\n"
            "- **Cấu hình tối ưu tiêu biểu:**\n"
            "  + *LightGBM:* learning_rate=0.042, num_leaves=31, max_depth=6, feature_fraction=0.78, objective='regression_l1'.\n"
            "  + *GRU:* hidden_dim=64, num_layers=2, dropout=0.20, learning_rate=0.0012, batch_size=64 (AdamW + Cosine Annealing).\n"
            "  + *ElasticNet:* alpha=0.085, l1_ratio=0.45.\n"
            "- **Cơ chế Early Stopping:** patience=10 epochs cho GRU/LSTM và early_stopping_rounds=30 cho LightGBM, "
            "chống triệt để hiện tượng học vẹt (overfitting)."
        ),
    ),
    KnowledgeDocument(
        doc_id="mase_metric",
        title="Tiêu chuẩn vàng MASE thay thế RMSE & Mẫu số chuẩn hóa đồng nhất 1.821 µg/m³",
        category="evaluation",
        keywords=[
            "mase",
            "metric",
            "tieu chuan",
            "rmse",
            "mae",
            "hyndman",
            "hyndman 2006",
            "mau so",
            "1.821",
            "true skill",
            "chuan hoa",
        ],
        source="knowledge_vault/05_Evaluation_and_Uncertainty/01_Metrics_Standard_MASE_over_RMSE.md",
        citations=["Hyndman & Koehler (2006)", "Mục 3.5 Báo cáo ThS"],
        content=(
            "### Tiêu chuẩn vàng MASE (Mean Absolute Scaled Error) & Mẫu số chuẩn hóa:\n\n"
            "- **Tại sao không dùng RMSE/MAE/R²?** MAE/RMSE phụ thuộc thang đo (scale-dependent). Nồng độ Sa Đéc trung bình ~10 µg/m³, "
            "MAE=3.5 ở đây tương đương MAE=50 ở nơi ô nhiễm nặng, không thể so sánh chéo. R² dùng đường trung bình ngang làm baseline, "
            "phi thực tế trong chuỗi thời gian tự tương quan cao.\n"
            "- **Công thức MASE (Hyndman 2006):** MASE = MAE_model / MAE_{in-sample naive persistence}.\n"
            "- **Mẫu số chuẩn hóa đồng nhất đề án:** MAE_{Persistence_1h} = 1.821 µg/m³ được cố định cho tất cả 11 mô hình và mọi horizons.\n"
            "- **Ngưỡng phân định True Skill:** MASE < 1.0 chứng minh mô hình có năng lực dự báo thực chất; MASE ≥ 1.0 là vô giá trị "
            "(kém hơn sao chép giá trị gần nhất). Champion Ensemble của đề án đạt MASE từ 0.382 đến 0.712."
        ),
    ),
    KnowledgeDocument(
        doc_id="r2_out_of_sample_negative",
        title="Tại sao hệ số xác định R² ngoài mẫu (Out-of-Sample) lại nhận giá trị âm?",
        category="evaluation",
        keywords=[
            "r2 am",
            "r2 out of sample",
            "r2 ngoai mau",
            "negative r2",
            "ss_res",
            "ss_tot",
            "he so xac dinh",
            "ngoai mau am",
            "r2 am ngoai mau",
            "he so r2",
            "r bình âm",
            "r2 am la gi",
            "giai thich r2 am",
            "tai sao r2 am",
        ],
        source="knowledge_vault/05_Evaluation_and_Uncertainty/01_Metrics_Standard_MASE_over_RMSE.md",
        citations=["Hyndman & Koehler (2006)", "Armstrong (1985)", "Mục 3.5 Báo cáo ThS"],
        content=(
            "### Tại sao hệ số xác định R² ngoài mẫu (Out-of-Sample) lại nhận giá trị âm?\n\n"
            "- **Định nghĩa toán học của R²:**\n"
            "  $$R^2 = 1 - \\frac{SS_{res}}{SS_{tot}} = 1 - \\frac{\\sum_{t=1}^n (y_t - \\hat{y}_t)^2}{\\sum_{t=1}^n (y_t - \\bar{y}_{test})^2}$$\n\n"
            "- **Tại sao R² có thể âm trong chuỗi thời gian ngoài mẫu (Out-of-Sample)?**\n"
            "  1. **Trong OLS hồi quy cổ điển nội mẫu (In-sample):** Mô hình tuyến tính luôn có R² nằm trong [0, 1] do có chứa hệ số chặn "
            "(intercept) và SS_res <= SS_tot theo định lý phân rã phương sai.\n"
            "  2. **Trong dự báo chuỗi thời gian ngoài mẫu (Out-of-Sample):** Baseline ngầm định của R² là đường thẳng nằm ngang trung bình "
            "y_bar_test. Giả định này hoàn toàn phi thực tế vì trong thực tế ta không thể biết trước giá trị trung bình tương lai của tập test! "
            "Khi chuỗi thời gian có xu hướng (trend), chu kỳ mùa (seasonality), trôi dạt phân phối (concept drift), hoặc mô hình bị lệch pha "
            "(phase lag), tổng bình phương sai số của mô hình SS_res hoàn toàn có thể vượt qua tổng phương sai tự nhiên của chuỗi test SS_tot. "
            "Khi đó SS_res / SS_tot > 1 dẫn đến R² < 0.\n\n"
            "- **Khẳng định tính đúng đắn & Liêm chính khoa học:**\n"
            "  Giá trị R² < 0 KHÔNG PHẢI là lỗi lập trình hay mô hình bị sụp đổ, mà là minh chứng khoa học đanh thép chỉ ra rằng: "
            "Hệ số R² là một thước đo phi lý, nguy hiểm và gây hiểu lầm nghiêm trọng trong dự báo chuỗi thời gian (Hyndman & Koehler, 2006). "
            "Đó chính là lý do đề án thạc sĩ CTU dứt khoát loại bỏ R² và chọn MASE làm tiêu chuẩn vàng đồng nhất."
        ),
    ),
    KnowledgeDocument(
        doc_id="diebold_mariano",
        title="Kiểm định ý nghĩa thống kê Diebold-Mariano & Hiệu chỉnh Harvey-HLN",
        category="evaluation",
        keywords=[
            "diebold mariano",
            "diebold-mariano",
            "dm test",
            "kiem dinh",
            "y nghia thong ke",
            "harvey",
            "hln",
            "p-value",
            "dau am",
            "dau duong",
            "+13.729",
            "-8.452",
            "-5.891",
            "-4.120",
            "bang 4.7",
            "dong 1",
            "dong 1 bang diebold mariano",
            "tai sao dong 1 mang dau duong",
            "tai sao dm mang dau am",
            "quy uoc dau dm",
        ],
        source="knowledge_vault/05_Evaluation_and_Uncertainty/02_Diebold_Mariano_Hypothesis_Testing.md",
        citations=["Diebold & Mariano (1995)", "Harvey, Leybourne & Newbold (1997)", "Bảng 4.7 Đề án"],
        content=(
            "### Kiểm định ý nghĩa thống kê Diebold-Mariano (DM Test) & Hiệu chỉnh HLN (Bảng 4.7):\n\n"
            "- **Mục đích:** Khẳng định bằng toán học xác suất rằng sự vượt trội của mô hình không phải do ngẫu nhiên.\n"
            "- **Hiệu chỉnh Harvey-HLN (1997):** Áp dụng công thức hiệu chỉnh mẫu hữu hạn cho tầm dự báo đa bước h > 1: "
            "DM* = DM * sqrt([n + 1 - 2h + h(h-1)/n] / n).\n"
            "- **Quy ước vi phân tổn thất & Bản chất dấu của DM:**\n"
            "  Hàm sai số vi phân tổn thất: d_t = |e_{proposed, t}| - |e_{baseline, t}|.\n"
            "  Thống kê DM chuẩn tắc kiểm định giả thuyết H0: E(d_t) = 0.\n"
            "  + **Tại sao dòng 1 (1h) mang dấu dương (+13.729)?** Tại tầm cực ngắn 1h, chuỗi có tự tương quan rất cao (r = 0.86), "
            "baseline ngây thơ Persistence có sai số cực nhỏ (|e_base| < |e_proposed|). Do đó d_t > 0 => d_bar > 0 => DM* = +13.729 "
            "với p < 0.001. Con số dấu dương này phản ánh hoàn toàn trung thực với thực nghiệm: ở tầm 1h, Persistence đánh bại các mô hình học sâu đơn lẻ.\n"
            "  + **Tại sao dòng 2, 3, 4 (6h, 24h) mang dấu âm (-8.452, -5.891, -4.120)?** Ở tầm xa 6h và 24h, quán tính tắt dần, mô hình đề xuất vượt trội "
            "với sai số nhỏ hơn mô hình cơ sở (|e_proposed| < |e_baseline| => d_t < 0 => d_bar < 0 => DM* < 0). "
            "Giá trị thống kê DM* = -8.452 (Ensemble vs Persistence ở 6h), -5.891 (GRU 15m vs Persistence), -4.120 (LightGBM vs ARIMA) "
            "đều có p < 0.001, khẳng định mô hình đề xuất vượt trội có ý nghĩa thống kê 99.9%."
        ),
    ),
    KnowledgeDocument(
        doc_id="uncertainty_cqr_aci",
        title="Định lượng độ bất định Conformal Prediction (CQR) & Tương thích động ACI 90.2%",
        category="evaluation",
        keywords=[
            "conformal prediction",
            "cqr",
            "aci",
            "bat dinh",
            "khoang tin cay",
            "khoang du bao",
            "coverage",
            "do phu 90%",
            "concept drift",
            "bang 4.6",
        ],
        source="knowledge_vault/05_Evaluation_and_Uncertainty/03_Uncertainty_Quantification_CQR_ACI.md",
        citations=["Romano et al. (2019) CQR", "Gibbs & Candès (2021) ACI", "Bảng 4.6 Báo cáo ThS"],
        content=(
            "### Định lượng độ bất định Conformal Prediction (CQR) & ACI Thích Ứng:\n\n"
            "- **Giới hạn của khoảng tin cậy Gauss:** Phân phối PM2.5 lệch phải mạnh và có phần dư Ljung-Box p<0.05, "
            "khiến công thức chuẩn y ± 1.96*sigma thất bại hoàn toàn.\n"
            "- **Conformal Quantile Regression (CQR Tĩnh):** Tạo khoảng dự báo không phụ thuộc phân phối (Distribution-free) "
            "với mức danh định 90%. Tuy nhiên khi gặp trôi dạt phân phối (Concept Drift) giữa các mùa, độ phủ thực tế tụt về 76% - 80.5%.\n"
            "- **Adaptive Conformal Inference (ACI, gamma=0.01):** Cập nhật động ngưỡng hiệu chỉnh theo sai số bước trước: "
            "alpha_{t+1} = alpha_t + gamma*(alpha - err_t). Khôi phục độ phủ thực nghiệm trên test set đạt 90.2% "
            "(89.6% ở 1h, 89.4% ở 6h, 90.2% ở 24h), bảo vệ trọn vẹn cam kết danh định 90%."
        ),
    ),
    KnowledgeDocument(
        doc_id="shap_xai",
        title="Phân luồng giải thích XAI (Tree SHAP vs Permutation) & Điểm chuyển pha 14–17 µg/m³",
        category="explainability",
        keywords=[
            "shap",
            "tree shap",
            "xai",
            "giai thich",
            "permutation",
            "permutation importance",
            "tipping point",
            "chuyen pha",
            "14-17",
            "14 17",
            "who 15",
            "bang 4.5",
            "hinh 4.8",
        ],
        source="knowledge_vault/06_Explainability_and_Domain_Insights/01_Tree_SHAP_vs_Permutation_Importance.md",
        citations=["Lundberg et al. (2020) TreeExplainer", "Breiman (2001)", "Bảng 4.5 & Hình 4.8 Đề án"],
        content=(
            "### Phân luồng XAI & Điểm chuyển pha khí quyển 14–17 µg/m³:\n\n"
            "- **Chiến lược phân luồng:** Tree SHAP (TreeExplainer) cho LightGBM tính chính xác 100% Shapley values trong <100ms; "
            "Permutation Importance cho mạng sâu GRU/LSTM để tránh bẫy tốn 16 giờ của Kernel SHAP.\n"
            "- **Đồng thuận Top đặc trưng (Bảng 4.5):** Cả hai phương pháp độc lập đều xác nhận Top-1 là `pm25_lag_1h` "
            "và Top-2 là `pm25_roll_24h_mean` (nồng độ nền 24h).\n"
            "- **Phát hiện Điểm chuyển pha 14–17 µg/m³ (Hình 4.8):** SHAP Dependence Plot của biến nền 24h gãy khúc rõ rệt tại dải 14–17 µg/m³:\n"
            "  + Vùng 1 (<14 µg/m³): SHAP âm, khí quyển tự làm sạch tốt.\n"
            "  + Vùng 2 (14–17 µg/m³, trùng chuẩn WHO 15 µg/m³): Điểm chuyển pha nhạy cảm, khí quyển bão hòa khả năng tự làm sạch.\n"
            "  + Vùng 3 (>17 µg/m³): SHAP vọt dương dốc đứng, bùng phát ô nhiễm phi tuyến do bẫy nhiệt và lắng đọng bụi."
        ),
    ),
    KnowledgeDocument(
        doc_id="sa_dec_domain",
        title="Đặc thù vi khí hậu Sa Đéc (Đồng Tháp) & Chu kỳ ngày đêm, nguồn phát thải",
        category="domain_insights",
        keywords=[
            "sa dec",
            "sa đéc",
            "dong thap",
            "đồng tháp",
            "chu ky ngay dem",
            "diurnal",
            "nghich nhiet",
            "lang hoa",
            "rom ra",
            "dot dong",
            "giao thong thuy",
            "dinh sang 6h",
            "day trua 12h",
        ],
        source="knowledge_vault/06_Explainability_and_Domain_Insights/03_Diurnal_Cycle_and_Sa_Dec_Context.md",
        citations=["Mục 4.1.3 Báo cáo ThS", "Sở TN&MT Đồng Tháp"],
        content=(
            "### Đặc thù vi khí hậu và nguồn phát thải tại TP. Sa Đéc (Đồng Tháp):\n\n"
            "- **Chu kỳ ngày đêm (Diurnal Cycle):**\n"
            "  + Cực đại buổi sáng (06:00 – 08:00): Trùng giờ cao điểm giao thông chợ hoa/trường học kết hợp hiện tượng "
            "nghịch nhiệt bề mặt sáng sớm giữ khói bụi sát mặt đất.\n"
            "  + Cực tiểu buổi trưa (12:00 – 14:00): Bức xạ mặt trời mạnh làm bung lớp nghịch nhiệt, đối lưu nâng lớp biên lên 1.500m "
            "giúp bụi mịn khuếch tán nhanh.\n"
            "  + Đỉnh phụ buổi tối (18:00 – 20:00): Sinh hoạt và lưu lượng xe tải, ghe thuyền đường thủy ban đêm.\n"
            "- **Nguồn phát thải địa phương:** Giao thông thủy nội địa trên sông Tiền và kênh xáng Sa Đéc (động cơ diesel cũ xả muội than); "
            "làng hoa kiểng Sa Đéc (đốt trấu, chăm hoa vụ Tết); các đợt đốt rơm rạ mùa lúa (tháng 3-4 và tháng 8-9) gây ô nhiễm đột biến >60 µg/m³."
        ),
    ),
    KnowledgeDocument(
        doc_id="temporal_split",
        title="Phân tách thời gian 80:10:10 & Kỷ luật đánh giá trên dữ liệu thực (Test on Real Only)",
        category="data_integrity",
        keywords=[
            "temporal split",
            "chia tap",
            "train val test",
            "80:10:10",
            "80 10 10",
            "test on real only",
            "is_imputed==0",
            "is_imputed",
            "anchor test",
            "669h",
        ],
        source="knowledge_vault/02_Data_Integrity_and_Anti_Leakage/02_Temporal_Split_and_Test_on_Real_Only.md",
        citations=["Bảng 3.4 Báo cáo ThS", "Mục 3.3 Đề án"],
        content=(
            "### Phân tách thời gian 80:10:10 & Kỷ luật Test on Real Only:\n\n"
            "- **Cấm tuyệt đối K-Fold ngẫu nhiên:** Vi phạm tính nhân quả chuỗi thời gian do dùng tương lai dự đoán quá khứ.\n"
            "- **Anchor Temporal Split 80:10:10:**\n"
            "  + Train (80%): 16/03/2022 đến 30/09/2024 (~30 tháng, học quy luật dài hạn và chu kỳ mùa).\n"
            "  + Validation (10%): 01/10/2024 đến 31/12/2024 (~3 tháng, tinh chỉnh siêu tham số Optuna TPE 50 trials, early stopping).\n"
            "  + Anchor Test (10%): 01/01/2025 đến 11/05/2025 (~4.5 tháng; 669h ở 1h, 863 mẫu ở 30m, 1.836 mẫu ở 15m; kiểm định mù độc lập).\n"
            "- **Kỷ luật Test on Real Only:** Trên tập Test, toàn bộ metrics (MASE, RMSE, MAE) chỉ được tính trên những mẫu có số đo thật "
            "từ cảm biến (lọc `is_imputed == 0`). Không chấm điểm mô hình trên dữ liệu đã qua nội suy, bảo vệ 100% độ liêm chính học thuật."
        ),
    ),
    KnowledgeDocument(
        doc_id="feature_store_119",
        title="Kho đặc trưng 119 chiều (Feature Store) & Ma trận tương tác vi khí hậu",
        category="feature_engineering",
        keywords=[
            "feature store",
            "119 dac trung",
            "119 features",
            "lag features",
            "rolling stats",
            "ewma",
            "fourier",
            "calendar",
            "domain interaction",
            "ventilation index",
        ],
        source="knowledge_vault/03_Feature_Engineering/01_Feature_Store_119_Features.md",
        citations=["Mục 3.4 Báo cáo ThS", "marts_features.csv"],
        content=(
            "### Kho đặc trưng 119 chiều (Feature Store) gồm 6 nhóm:\n\n"
            "1. **Biến trễ (Lag — 40 đặc trưng):** Trễ từ t-1 đến t-168 (1 tuần) của PM2.5 và 4 biến khí tượng (Nhiệt độ, Độ ẩm, Gió, Áp suất).\n"
            "2. **Cửa sổ trượt (Rolling Stats — 36 đặc trưng):** Cửa sổ 3h, 6h, 12h, 24h, 48h, 72h, 168h tính mean, std, min, max, skew.\n"
            "3. **Làm mượt số mũ (EWMA — 16 đặc trưng):** Đa span thời gian nắm bắt quán tính phân rã ô nhiễm.\n"
            "4. **Điều hòa Fourier (Fourier Harmonics — 12 đặc trưng):** Cặp sin/cos chu kỳ 12h, 24h và 168h giúp mô hình học tuần hoàn trơn.\n"
            "5. **Lịch và Thời gian (Calendar — 11 đặc trưng):** Vòng tròn hour_sin/cos, dayofweek, month, cờ cuối tuần, cờ giờ cao điểm.\n"
            "6. **Tương tác chuyên miền (Domain Interactions — 4 đặc trưng):** Ventilation Index (Gió x Chiều cao lớp biên), "
            "tỷ số nhiệt/ẩm, tốc độ tích tụ ô nhiễm, tỷ lệ đóng góp ô nhiễm nền."
        ),
    ),
    KnowledgeDocument(
        doc_id="stationarity_adf_kpss",
        title="Kiểm định tính dừng (ADF & KPSS) & Bản chất phi chuẩn của chuỗi PM2.5",
        category="feature_engineering",
        keywords=[
            "adf",
            "kpss",
            "tinh dung",
            "stationarity",
            "shapiro wilk",
            "jarque bera",
            "phi chuan",
            "kinh te luong",
            "bang 3.5",
            "bang 4.1",
        ],
        source="knowledge_vault/03_Feature_Engineering/02_Stationarity_ADF_KPSS.md",
        citations=["Bảng 3.5 & 4.1 Báo cáo ThS", "Dickey-Fuller (1979)", "KPSS (1992)"],
        content=(
            "### Kiểm định tính dừng kết hợp ADF/KPSS & Tính phi chuẩn (Bảng 3.5 & 4.1):\n\n"
            "- **Augmented Dickey-Fuller (ADF):** Thống kê = -8.124, p < 0.001 -> Bác bỏ H0, chuỗi có tính dừng (Trend-stationary).\n"
            "- **KPSS Test:** Thống kê = 0.892, p = 0.010 < 0.05 -> Bác bỏ H0, chuỗi không dừng quanh mức trung bình tĩnh.\n"
            "- **Kết luận kinh tế lượng:** Chuỗi mang bản chất Dừng cục bộ & Biến thiên theo mùa (Locally Stationary with Seasonality). "
            "Không có nghiệm đơn vị nổ, nhưng kỳ vọng thay đổi tuần hoàn theo chu kỳ ngày đêm và mùa vụ.\n"
            "- **Tính phi chuẩn:** Shapiro-Wilk p < 0.001, Skewness = 2.0046 (lệch phải), Kurtosis = 6.1458 (đuôi dày). "
            "Bác bỏ giả định chuẩn Gauss, bắt buộc dùng Học máy phi tuyến và Conformal Prediction."
        ),
    ),
    KnowledgeDocument(
        doc_id="defense_faq",
        title="Bộ câu hỏi - đáp phản biện trọng điểm trước Hội đồng Đề án Thạc sĩ",
        category="defense",
        keywords=[
            "hoi dong",
            "phan bien",
            "defense",
            "faq",
            "cau hoi",
            "tra loi",
            "chat van",
            "chu tich",
            "tong ket",
            "luan an",
            "luan van",
            "de an",
            "grill-me",
            "grill me",
            "grill-me holes",
            "trong diem chat van",
            "bao ve de an",
            "cau hoi kho",
        ],
        source="knowledge_vault/07_Defense_Playbook_and_FAQ/01_Master_Defense_QnA_Hoi_Dong.md",
        citations=["Bộ 20 câu hỏi Hội đồng", "SLIDES_DEFENSE_PLAYBOOK.md"],
        content=(
            "### Bộ câu hỏi - đáp trọng điểm trước Hội đồng Đề án Thạc sĩ (Grill-Me Defense Holes):\n\n"
            "1. **Tại sao dùng MASE thay vì RMSE/MAE?** MASE scale-independent, mẫu số chuẩn hóa đồng nhất 1.821 µg/m³, "
            "phân định rõ True Skill (<1.0) và cho phép so sánh công bằng giữa các trạm quan trắc.\n"
            "2. **Tại sao dòng 1 bảng Diebold-Mariano mang dấu dương (+13.729) mà dòng 2, 3 mang dấu âm (-8.452)?**\n"
            "   Dấu của DM theo hàm vi phân tổn thất d_t = |e_proposed| - |e_base|. Ở tầm 1h, Persistence có sai số nhỏ hơn do "
            "tự tương quan cao r=0.86, dẫn đến d_t > 0 => DM* = +13.729. Ở tầm 6h và 24h, mô hình đề xuất vượt trội với sai số nhỏ hơn, "
            "d_t < 0 => DM* = -8.452 và -5.891 với p < 0.001, khẳng định ưu thế có ý nghĩa thống kê 99.9%.\n"
            "3. **Tại sao dũng cảm drop 19.810 giờ khuyết thiếu thay vì dùng GAN/Deep Learning?**\n"
            "   19.810h là dữ liệu khuyết dài >24h mang bản chất MNAR. Vượt quá 24h, RMSE tái tạo bùng nổ >12 µg/m³ (>100% nồng độ nền). "
            "Không dùng GAN/MICE để tự vẽ thêm dữ liệu ảo (chống Data Hallucination), bảo vệ tuyệt đối tính liêm chính nghiên cứu.\n"
            "4. **Tại sao hệ số xác định R² ngoài mẫu (Out-of-Sample) lại nhận giá trị âm?**\n"
            "   R² ngầm định đường trung bình tập test làm baseline (giả định biết trước tương lai). Khi chuỗi có xu hướng/mùa/trôi dạt "
            "hoặc mô hình bị lệch pha, sai số mô hình SS_res vượt qua phương sai tự nhiên SS_tot, dẫn đến R² < 0. Đây là lý do loại bỏ R².\n"
            "5. **Bẫy ngoại lai IQR 3.0 đã xóa nhầm dữ liệu ra sao?**\n"
            "   Tukey IQR 3.0 gọt nhầm 66 đỉnh ô nhiễm thực tế (55 - 120 µg/m³) do phân phối Fat-tailed (Skewness=2.00, Kurtosis=6.15), "
            "tạo ảo tưởng chính xác giả nhưng làm F1 cảnh báo tụt < 0.35. Đề án thay thế bằng Domain Bounds [0, 500] kết hợp S-ESD, "
            "khôi phục F1 lên 0.782.\n"
            "6. **Tại sao phân luồng XAI Tree SHAP vs Permutation?** Tree SHAP cho LightGBM chính xác 100% trong <100ms; "
            "Permutation cho GRU/LSTM tránh bẫy 16 giờ của Kernel SHAP."
        ),
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# FAST ZERO-DEPENDENCY SEARCH ENGINE (<1ms, 0 RAM)
# ═══════════════════════════════════════════════════════════════════════════


def search_curated_knowledge(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    """Search curated knowledge documents using zero-dependency keyword & semantic matching.

    Args:
        query: User prompt or question
        top_k: Number of top documents to return

    Returns:
        List of dicts: {"title": ..., "source": ..., "content": ..., "score": ...}
    """
    if not query or not query.strip():
        return []

    norm_query = _strip_accents(query)
    query_tokens = _tokenize(norm_query)
    raw_tokens = _tokenize(query)

    if not query_tokens:
        return []

    scored_results: list[tuple[float, KnowledgeDocument]] = []

    for doc in CURATED_DOCUMENTS:
        score = 0.0

        # 1. Exact phrase matching bonus
        if norm_query in doc._norm_title:
            score += 15.0
        if norm_query in doc._norm_content:
            score += 8.0

        # 2. Keywords matching (Highest signal)
        for i, norm_kw in enumerate(doc._norm_keywords):
            if norm_kw in norm_query or norm_query in norm_kw:
                score += 12.0
            matched_kw_tokens = doc._kw_token_sets[i].intersection(query_tokens)
            if matched_kw_tokens:
                score += len(matched_kw_tokens) * 3.5

        # 3. Title token matching
        matched_title = doc._title_tokens.intersection(query_tokens)
        score += len(matched_title) * 4.0

        # 4. Content token matching (with frequency dampening)
        matched_content = doc._content_token_set.intersection(query_tokens)
        for token in matched_content:
            tf = min(doc._content_tokens.count(token), 10)
            score += 0.8 * (1.0 + (tf**0.5))

        # 5. Raw token match bonus (accents preserved)
        raw_title = doc.title.lower()
        for rt in raw_tokens:
            if rt in raw_title:
                score += 2.0

        if score > 0.0:
            scored_results.append((score, doc))

    # Sort descending by score
    scored_results.sort(key=lambda x: x[0], reverse=True)

    results: list[dict[str, Any]] = []
    for score, doc in scored_results[:top_k]:
        results.append(
            {
                "title": doc.title,
                "source": doc.source,
                "content": doc.content,
                "score": round(score, 3),
                "category": doc.category,
                "citations": doc.citations,
            }
        )

    return results


def get_all_curated_topics() -> list[str]:
    """Return list of available curated topic titles."""
    return [doc.title for doc in CURATED_DOCUMENTS]


def get_curated_documents() -> list[KnowledgeDocument]:
    """Return all curated knowledge documents."""
    return list(CURATED_DOCUMENTS)
