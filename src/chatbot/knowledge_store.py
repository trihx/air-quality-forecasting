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
            "domain bounds",
            "dinh o nhiem",
            "fat tailed",
            "fat-tailed",
            "skewness",
            "kurtosis",
            "who aqi",
            "500",
        ],
        source="knowledge_vault/01_Pipeline_and_Data_Engineering/03_Outlier_Trap_and_Domain_Bounds.md",
        citations=["Mục 3.2 Báo cáo ThS", "Ablation Study v10 vs v9", "WHO AQI Guidelines"],
        content=(
            "### Bẫy xóa ngoại lai IQR 3.0 & Domain Bounds [0, 500] µg/m³:\n\n"
            "- **Bẫy IQR 3.0 cổ điển:** Phương pháp Tukey IQR (Q3 + 3.0*IQR ~ 54 µg/m³) đã gọt nhầm 66 đỉnh ô nhiễm thực tế "
            "(55 - 120 µg/m³). Việc này tạo ra ảo tưởng chính xác (False Sense of Accuracy): RMSE trên validation giảm đẹp giả tạo "
            "(2.12 µg/m³), nhưng khi ra môi trường thật mô hình hoàn toàn mù trước ô nhiễm nặng (F1 cảnh báo < 0.35).\n"
            "- **Bản chất phân phối Fat-Tailed của PM2.5:** Skewness = 2.0046 (lệch phải mạnh), Kurtosis = 6.1458 (đuôi dày, nhọn). "
            "Các đỉnh nồng độ cao là biến cố vi khí hậu có thật (nghịch nhiệt, đốt rơm rạ), không phải lỗi cảm biến.\n"
            "- **Giải pháp Domain Bounds [0, 500]:** Thiết lập ngưỡng vật lý [0, 500] µg/m³ theo chuẩn WHO AQI, kết hợp S-ESD "
            "chỉ loại bỏ lỗi phần cứng (treo cảm biến flatline). Khôi phục F1-score cảnh báo sớm lên 0.782."
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
            "74%",
            "do thuc",
        ],
        source="knowledge_vault/01_Pipeline_and_Data_Engineering/02_Tiered_Imputation_and_Data_Sparsity.md",
        citations=["Bảng 3.4 Báo cáo ThS", "Akima (1970)", "Mục 3.2 Luận văn"],
        content=(
            "### Chiến lược phục hồi dữ liệu phân tầng & Quyết định Drop 19.810 giờ:\n\n"
            "- **Thực trạng dữ liệu IoT Sa Đéc:** Tỷ lệ khuyết thiếu tích lũy lên tới 74% qua 38 tháng (do bảo trì, mất điện, mất sóng).\n"
            "- **Chiến lược 3 tầng xử lý:**\n"
            "  + **Tầng 1 (Gap ≤ 6 giờ):** Dùng PCHIP/Akima Spline bảo toàn tính đơn điệu cục bộ, chống vọt lố (overshooting).\n"
            "  + **Tầng 2 (6h < Gap ≤ 24 giờ):** Dùng KNN Imputation (k=5), tuân thủ nghiêm ngặt donors chỉ lấy trong quá khứ (t' < t).\n"
            "  + **Tầng 3 (Gap > 24 giờ):** DŨNG CẢM LOẠI BỎ 19.810 giờ khuyết dài. Không dùng MICE hay GAN để vẽ thêm dữ liệu giả mạo "
            "(chống Data Hallucination). Cắt chuỗi thành các phân đoạn liên tục sạch có ý nghĩa vật lý.\n"
            "- **Kích thước dữ liệu sạch sau xử lý:** 18.355 mẫu ở 15m, 8.625 mẫu ở 30m, 6.689 mẫu ở 1h."
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
            "bang 4.7",
        ],
        source="knowledge_vault/05_Evaluation_and_Uncertainty/02_Diebold_Mariano_Hypothesis_Testing.md",
        citations=["Diebold & Mariano (1995)", "Harvey, Leybourne & Newbold (1997)", "Bảng 4.7 Đề án"],
        content=(
            "### Kiểm định ý nghĩa thống kê Diebold-Mariano (DM Test) & Hiệu chỉnh HLN:\n\n"
            "- **Mục đích:** Khẳng định bằng toán học xác suất rằng sự vượt trội của mô hình không phải do ngẫu nhiên.\n"
            "- **Hiệu chỉnh Harvey-HLN (1997):** Áp dụng công thức hiệu chỉnh mẫu hữu hạn cho tầm dự báo đa bước h > 1.\n"
            "- **Giải trình quy ước dấu âm (Bảng 4.7):** Hàm vi phân tổn thất d_t = |e_{proposed}| - |e_{baseline}|. "
            "Khi mô hình đề xuất có sai số nhỏ hơn mô hình cơ sở, trung bình d_t < 0. Do đó, thống kê DM mang DẤU ÂM LỚN "
            "(ví dụ -8.452) và p < 0.001 chính là minh chứng mô hình đề xuất vượt trội có ý nghĩa thống kê.\n"
            "- **4 cặp đối kháng thực nghiệm:** Ensemble vs Persistence (DM*=-8.452, p<0.001); GRU 15m vs Persistence (DM*=-5.891, p<0.001); "
            "LightGBM vs ARIMA (DM*=-4.120, p=0.0002); Ensemble vs LightGBM (DM*=-2.134, p=0.033)."
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
        ],
        source="knowledge_vault/07_Defense_Playbook_and_FAQ/01_Master_Defense_QnA_Hoi_Dong.md",
        citations=["Bộ 20 câu hỏi Hội đồng", "SLIDES_DEFENSE_PLAYBOOK.md"],
        content=(
            "### Bộ câu hỏi - đáp trọng điểm trước Hội đồng Đề án Thạc sĩ:\n\n"
            "1. **Tại sao dùng MASE?** MASE scale-independent, mẫu số chuẩn hóa đồng nhất 1.821 µg/m³, phân định rõ True Skill (<1.0).\n"
            "2. **Tại sao DM mang dấu âm?** d_t = |e_proposed| - |e_baseline|. Sai số nhỏ hơn baseline tạo ra trung bình d_t < 0; "
            "thống kê DM < 0 và p < 0.001 khẳng định mô hình vượt trội có ý nghĩa thống kê.\n"
            "3. **Tại sao drop 19.810 giờ?** 19.810h là dữ liệu khuyết dài MNAR. Thà chịu mất mẫu còn hơn vẽ dữ liệu ảo (GAN/MICE) "
            "làm sai lệch quy luật tự nhiên.\n"
            "4. **Tại sao R² ngoài mẫu có thể âm?** R² so với đường trung bình ngang. Khi mô hình bị lệch pha ở dữ liệu ngoài mẫu, "
            "tổng bình phương sai số mô hình lớn hơn phương sai dữ liệu, dẫn đến R² < 0.\n"
            "5. **Tại sao phân luồng XAI?** Tree SHAP cho LightGBM chính xác 100% trong <100ms; Permutation cho GRU tránh bẫy 16h của Kernel SHAP."
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
