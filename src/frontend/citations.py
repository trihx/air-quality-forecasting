"""IEEE Citation Popup System + Pipeline Step Framework.

Provides two complementary utilities:
1. ``cite(ref_id)`` — Inline ``[number]`` tooltip for IEEE academic references.
2. ``step(n)`` — Circled number ``①②③`` for pipeline step indicators.

This separation ensures readers instantly distinguish *process steps* from
*literature citations* on the dashboard.

Usage in Streamlit pages::

    from src.frontend.citations import cite, step, render_references_section

    st.markdown(
        f"{step(3)} Impute {cite('troyanskaya2001')} → 7,742 rows",
        unsafe_allow_html=True,
    )
    render_references_section()  # at page bottom
"""

from __future__ import annotations

import urllib.parse
import streamlit as st

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# IEEE Reference Database
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

IEEE_REFS: dict[str, dict] = {
    "hyndman2006": {
        "id": 1,
        "authors": "R. J. Hyndman and A. B. Koehler",
        "title": "Another look at measures of forecast accuracy",
        "journal": "Int. J. Forecasting",
        "year": 2006,
        "vol": "22(4)",
        "pages": "679–688",
        "doi": "10.1016/j.ijforecast.2006.03.001",
        "used_in": "Evaluation — MASE metric definition",
        "context": "MASE (Mean Absolute Scaled Error) là thước đo scale-free chuẩn mực. Bằng cách chia sai số dự báo cho sai số của mô hình Naive, MASE khắc phục lỗi chia cho 0 của MAPE và ít bị nhiễu bởi outliers, cho phép so sánh chéo nhiều tập dữ liệu.",
    },
    "willmott2005": {
        "id": 2,
        "authors": "C. J. Willmott and K. Matsuura",
        "title": "Advantages of the mean absolute error (MAE) over the root mean square error (RMSE)",
        "journal": "Climate Research",
        "year": 2005,
        "vol": "30(1)",
        "pages": "79–82",
        "doi": "10.3354/cr030079",
        "used_in": "Evaluation — MAE as primary metric justification",
        "context": "Nghiên cứu chứng minh MAE đánh giá sai số trung bình tự nhiên và phản ánh thực tế tốt hơn RMSE. RMSE khuếch đại sai số theo hàm mũ, khiến các dự báo bị phạt quá nặng bởi các outliers cực đoan trong chuỗi thời gian.",
    },
    "gneiting2007": {
        "id": 3,
        "authors": "T. Gneiting and A. E. Raftery",
        "title": "Strictly proper scoring rules, prediction, and estimation",
        "journal": "J. Amer. Statistical Assoc.",
        "year": 2007,
        "vol": "102(477)",
        "pages": "359–378",
        "doi": "10.1198/016214506000001437",
        "used_in": "Evaluation — CRPS for probabilistic forecasts",
        "context": "CRPS (Continuous Ranked Probability Score) là thước đo độ chính xác cho dự báo phân phối xác suất. CRPS trừng phạt nghiêm ngặt cả việc mô hình thiếu tự tin (quá rộng) hoặc tự tin thái quá (quá hẹp), giúp đánh giá chất lượng của khoảng tin cậy (Prediction Intervals).",
    },
    "romano2019": {
        "id": 4,
        "authors": "Y. Romano, E. Patterson, and E. J. Candès",
        "title": "Conformalized quantile regression",
        "journal": "Advances in Neural Information Processing Systems",
        "year": 2019,
        "vol": "32",
        "pages": "",
        "doi": "",
        "used_in": "Prediction Intervals — CQR calibration",
        "context": "CQR (Conformalized Quantile Regression) kết hợp hồi quy phân vị (Quantile Regression) với Conformal Prediction. Kỹ thuật này giúp hiệu chỉnh (calibrate) các khoảng tin cậy để đảm bảo độ phủ biên (marginal coverage) luôn đạt mức kỳ vọng (vd: 90%) mà không cần giả định về phân phối của dữ liệu.",
    },
    "cho2014": {
        "id": 5,
        "authors": "K. Cho et al.",
        "title": "Learning phrase representations using RNN encoder-decoder for statistical machine translation",
        "journal": "arXiv:1406.1078",
        "year": 2014,
        "vol": "",
        "pages": "",
        "doi": "10.3115/v1/D14-1179",
        "used_in": "Models — GRU architecture",
        "context": "GRU (Gated Recurrent Unit) là một biến thể tối ưu của RNN, kết hợp Forget Gate và Input Gate thành một Update Gate duy nhất. GRU giảm đáng kể số lượng tham số so với LSTM, giúp huấn luyện nhanh hơn và chống overfit tốt hơn trên các bộ dữ liệu chuỗi thời gian có kích thước vừa và nhỏ.",
    },
    "ke2017": {
        "id": 6,
        "authors": "G. Ke et al.",
        "title": "LightGBM: A highly efficient gradient boosting decision tree",
        "journal": "Advances in Neural Information Processing Systems",
        "year": 2017,
        "vol": "30",
        "pages": "",
        "doi": "",
        "used_in": "Models — LightGBM tree-based baseline",
        "context": "LightGBM sử dụng chiến lược phát triển cây theo lá (Leaf-wise) kết hợp thuật toán GOSS (Gradient-based One-Side Sampling). Cơ chế này giúp mô hình đạt tốc độ huấn luyện vượt trội và độ chính xác cao đối với dữ liệu dạng bảng (tabular data), đặc biệt khi có các đặc trưng lag/rolling phức tạp.",
    },
    "hochreiter1997": {
        "id": 7,
        "authors": "S. Hochreiter and J. Schmidhuber",
        "title": "Long short-term memory",
        "journal": "Neural Computation",
        "year": 1997,
        "vol": "9(8)",
        "pages": "1735–1780",
        "doi": "10.1162/neco.1997.9.8.1735",
        "used_in": "Models — LSTM architecture",
        "context": "LSTM (Long Short-Term Memory) giải quyết triệt để bài toán suy giảm đạo hàm (Vanishing Gradient) trong chuỗi thời gian dài. Nhờ hệ thống Cell State và 3 cổng điều khiển (Input, Output, Forget), LSTM có khả năng 'ghi nhớ' các chu kỳ ô nhiễm PM2.5 kéo dài (ví dụ: chu kỳ mùa, ngày đêm).",
    },
    "lim2021": {
        "id": 8,
        "authors": "B. Lim et al.",
        "title": "Temporal Fusion Transformers for interpretable multi-horizon time series forecasting",
        "journal": "Int. J. Forecasting",
        "year": 2021,
        "vol": "37(4)",
        "pages": "1748–1764",
        "doi": "10.1016/j.ijforecast.2021.03.012",
        "used_in": "Models — TFT architecture",
        "context": "TFT (Temporal Fusion Transformer) kết hợp mạng RNN (để nắm bắt xu hướng cục bộ) và cơ chế Self-Attention (để học phụ thuộc xa). Khác với các mô hình Black-box, TFT cung cấp khả năng diễn giải mạnh mẽ thông qua Variable Selection Network (đánh giá tầm quan trọng của từng đặc trưng đầu vào).",
    },
    "peixeiro2022": {
        "id": 9,
        "authors": "M. Peixeiro",
        "title": "Time Series Forecasting in Python",
        "journal": "Manning Publications",
        "year": 2022,
        "vol": "",
        "pages": "",
        "doi": "",
        "used_in": "Pipeline — End-to-end forecasting methodology",
        "context": "Nghiên cứu áp dụng quy trình đánh giá chuẩn mực của Peixeiro: Xây dựng đường cơ sở tĩnh (Persistence) → Mô hình thống kê (ARIMA/SARIMA) → Machine Learning (LightGBM) → Deep Learning (GRU/LSTM/TFT) để chứng minh sự gia tăng hiệu suất thực sự của các mô hình phức tạp.",
    },
    "shumway2017": {
        "id": 10,
        "authors": "R. H. Shumway and D. S. Stoffer",
        "title": "Time Series Analysis and Its Applications: With R Examples",
        "journal": "Springer",
        "year": 2017,
        "vol": "4th ed.",
        "pages": "",
        "doi": "10.1007/978-3-319-52452-8",
        "used_in": "EDA — Statistical foundations (ADF, KPSS, ACF/PACF)",
        "context": "Nền tảng lý thuyết cho chuỗi thời gian: Sử dụng kiểm định ADF (Augmented Dickey-Fuller) và KPSS để xác minh tính dừng (stationarity). Phân tích ACF/PACF giúp xác định chính xác các bước trễ (lag) nội tại trong nồng độ PM2.5 do tính tự tương quan (autocorrelation).",
    },
    "diebold1995": {
        "id": 11,
        "authors": "F. X. Diebold and R. S. Mariano",
        "title": "Comparing predictive accuracy",
        "journal": "J. Bus. Econ. Stat.",
        "year": 1995,
        "vol": "13(3)",
        "pages": "253–263",
        "doi": "10.1080/07350015.1995.10524599",
        "used_in": "Evaluation — Diebold-Mariano test for model comparison",
        "context": "Kiểm định Diebold-Mariano (DM Test) được sử dụng để xác minh xem sự chênh lệch độ chính xác giữa 2 mô hình (vd: GRU so với LightGBM) có thực sự mang ý nghĩa thống kê (statistically significant) hay chỉ do sự tình cờ của nhiễu dữ liệu tập test.",
    },
    "akiba2019": {
        "id": 12,
        "authors": "T. Akiba et al.",
        "title": "Optuna: A next-generation hyperparameter optimization framework",
        "journal": "Proc. ACM SIGKDD",
        "year": 2019,
        "vol": "",
        "pages": "2623–2631",
        "doi": "10.1145/3292500.3330701",
        "used_in": "Training — Hyperparameter optimization with TPE sampler",
        "context": "Optuna là framework tối ưu hóa siêu tham số thế hệ mới. Áp dụng thuật toán lấy mẫu TPE (Tree-structured Parzen Estimator) dựa trên tối ưu hóa Bayes, Optuna có khả năng tự động khám phá không gian tìm kiếm và cắt tỉa (pruning) sớm các phép thử nghiệm kém hiệu quả, tiết kiệm đáng kể tài nguyên tính toán.",
    },
    # ── Thesis [14] Box-Cox ──
    "boxcox1964": {
        "id": 13,
        "authors": "G. E. P. Box and D. R. Cox",
        "title": "An Analysis of Transformations",
        "journal": "J. Royal Statistical Soc. B",
        "year": 1964,
        "vol": "26(2)",
        "pages": "211–252",
        "doi": "",
        "used_in": "Pipeline — Box-Cox transform for fat-tailed PM2.5",
        "context": "Phép biến đổi Box-Cox tìm giá trị λ tối ưu để ổn định phương sai của biến mục tiêu. Với PM2.5 (λ ≈ −0.147 ≈ 0), phép Log Transform giúp thu hẹp biên độ các đỉnh ô nhiễm cực đoan, cải thiện hiệu suất mô hình Học sâu.",
    },
    # ── Thesis [15] Rosner — S-ESD ──
    "rosner1983": {
        "id": 14,
        "authors": "B. Rosner",
        "title": "Percentage Points for a Generalized ESD Many-Outlier Procedure",
        "journal": "Technometrics",
        "year": 1983,
        "vol": "25(2)",
        "pages": "165–172",
        "doi": "10.1080/00401706.1983.10487848",
        "used_in": "Pipeline ② — Outlier detection (S-ESD / MAD)",
        "context": "Thủ tục ESD tổng quát (Generalized ESD) phát hiện đa ngoại lệ đồng thời. Luận văn áp dụng biến thể Seasonal-ESD kết hợp STL detrend + MAD để bảo toàn các đỉnh ô nhiễm thực sự của PM2.5, tránh cắt tín hiệu sinh thái.",
    },
    # ── Thesis [4] Tashman ──
    "tashman2000": {
        "id": 15,
        "authors": "L. J. Tashman",
        "title": "Out-of-sample tests of forecasting accuracy: an analysis and review",
        "journal": "Int. J. Forecasting",
        "year": 2000,
        "vol": "16(4)",
        "pages": "437–450",
        "doi": "",
        "used_in": "Pipeline ⑤ — Out-of-sample evaluation design",
        "context": "Bài tổng quan đặt nền tảng cho phương pháp đánh giá dự báo ngoài mẫu (out-of-sample). Luận văn áp dụng nguyên tắc: tập Test tách biệt hoàn toàn, chỉ dùng dữ liệu thật (real-only), không bao giờ dùng data đã impute để đánh giá.",
    },
    # ── Thesis [5] Hyndman FPP3 ──
    "hyndman2021": {
        "id": 16,
        "authors": "R. J. Hyndman and G. Athanasopoulos",
        "title": "Forecasting: Principles and Practice",
        "journal": "OTexts, 3rd ed.",
        "year": 2021,
        "vol": "",
        "pages": "",
        "doi": "",
        "used_in": "Pipeline ④⑤ — Walk-forward CV, anti-leakage design",
        "context": "Sách giáo khoa chuẩn mực về dự báo chuỗi thời gian. Luận văn áp dụng: (a) Walk-Forward Expanding Window thay K-Fold, (b) shift(1) anti-leakage cho diff/pct_change, (c) Purging Gap = max_lookback giữa Train và Test.",
    },
    # ── Thesis [6] M4 Competition ──
    "makridakis2020": {
        "id": 17,
        "authors": "S. Makridakis, E. Spiliotis, and V. Assimakopoulos",
        "title": "The M4 Competition: Results, findings, conclusion and way forward",
        "journal": "Int. J. Forecasting",
        "year": 2020,
        "vol": "36(1)",
        "pages": "54–74",
        "doi": "",
        "used_in": "Evaluation — M4 Competition benchmark methodology",
        "context": "Cuộc thi M4 (100.000 chuỗi thời gian) chứng minh: (a) mô hình đơn giản (ETS/ARIMA) vẫn cạnh tranh với ML/DL ở khung ngắn, (b) ensemble luôn thắng single model, (c) MASE là metric chuẩn để so sánh chéo. Luận văn áp dụng cả 3 nguyên tắc này.",
    },
    # ── Thesis [17] Gal — MC Dropout ──
    "gal2016": {
        "id": 18,
        "authors": "Y. Gal and Z. Ghahramani",
        "title": "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning",
        "journal": "Proc. ICML",
        "year": 2016,
        "vol": "48",
        "pages": "1050–1059",
        "doi": "",
        "used_in": "Prediction Intervals — MC Dropout (đã loại bỏ)",
        "context": "MC Dropout xấp xỉ Bayesian bằng cách bật Dropout lúc inference và chạy nhiều lần. Tuy nhiên, trên dataset PM2.5 nhỏ (7K mẫu), phương sai Dropout quá nhỏ → khoảng tin cậy cực hẹp (coverage 7,6% thay vì 90%). Luận văn đã thay thế bằng CQR.",
    },
    # ── Thesis [19] Deep Ensembles ──
    "lakshminarayanan2017": {
        "id": 19,
        "authors": "B. Lakshminarayanan, A. Pritzel, and C. Blundell",
        "title": "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles",
        "journal": "Advances in NeurIPS",
        "year": 2017,
        "vol": "30",
        "pages": "",
        "doi": "",
        "used_in": "Prediction Intervals — Deep Ensembles reference",
        "context": "Deep Ensembles ước lượng bất định bằng cách huấn luyện N mô hình độc lập rồi tổng hợp phân phối dự báo. Luận văn dùng GRU 5-seed ensemble (seeds: 42, 123, 456, 789, 2024) theo nguyên tắc này, đạt MASE = 0,72 tại h=24.",
    },
    # ── NEW: Lundberg SHAP ──
    "lundberg2017": {
        "id": 20,
        "authors": "S. M. Lundberg and S.-I. Lee",
        "title": "A Unified Approach to Interpreting Model Predictions",
        "journal": "Advances in NeurIPS",
        "year": 2017,
        "vol": "30",
        "pages": "4765–4774",
        "doi": "",
        "used_in": "Explainability — SHAP feature importance",
        "context": "SHAP (SHapley Additive exPlanations) dựa trên lý thuyết trò chơi Shapley để phân bổ công bằng đóng góp của từng đặc trưng vào dự báo. TreeExplainer cho LightGBM tính SHAP values chính xác trong O(TLD²), giúp giải thích tại sao pm25_lag_1h chi phối hoàn toàn ở h=1.",
    },
    # ── NEW: Cleveland STL ──
    "cleveland1990": {
        "id": 21,
        "authors": "R. B. Cleveland, W. S. Cleveland, J. E. McRae, and I. Terpenning",
        "title": "STL: A Seasonal-Trend Decomposition Procedure Based on Loess",
        "journal": "J. Official Statistics",
        "year": 1990,
        "vol": "6(1)",
        "pages": "3–73",
        "doi": "",
        "used_in": "EDA — Seasonal-Trend decomposition",
        "context": "STL phân tách chuỗi thời gian thành 3 thành phần: Trend, Seasonality, Remainder bằng thuật toán Loess (locally weighted regression). Luận văn dùng STL (period=24, robust=True) để xác nhận chu kỳ ngày/đêm của PM2.5 và phát hiện leakage khi fit STL trên toàn bộ data.",
    },
    # ── NEW: Troyanskaya KNN Imputation ──
    "troyanskaya2001": {
        "id": 22,
        "authors": "O. Troyanskaya et al.",
        "title": "Missing Value Estimation Methods for DNA Microarrays",
        "journal": "Bioinformatics",
        "year": 2001,
        "vol": "17(6)",
        "pages": "520–525",
        "doi": "10.1093/bioinformatics/17.6.520",
        "used_in": "Pipeline ③ — KNN Imputation (gap 6–24h)",
        "context": "KNN Imputation ước lượng giá trị thiếu bằng trung bình có trọng số (distance-weighted) của K hàng xóm gần nhất trong không gian đa biến. Luận văn áp dụng KNNImputer(n_neighbors=5) cho gap 6–24h, tận dụng tương quan chéo giữa PM2.5, nhiệt độ, độ ẩm, CO2.",
    },
    # ── NEW: Dickey-Fuller ADF ──
    "dickey1979": {
        "id": 23,
        "authors": "D. A. Dickey and W. A. Fuller",
        "title": "Distribution of the Estimators for Autoregressive Time Series with a Unit Root",
        "journal": "J. Amer. Statistical Assoc.",
        "year": 1979,
        "vol": "74(366)",
        "pages": "427–431",
        "doi": "10.1080/01621459.1979.10482531",
        "used_in": "EDA — Augmented Dickey-Fuller (ADF) stationarity test",
        "context": "Kiểm định ADF kiểm tra H₀: chuỗi có unit root (không dừng). Luận văn kết hợp ADF + KPSS (2 chiều giả thuyết) để xác nhận PM2.5 là trend-stationary (ADF reject H₀, KPSS reject H₀ stationarity) — cho thấy cần sai phân trước khi ARIMA.",
    },
    # ── NEW: KPSS ──
    "kwiatkowski1992": {
        "id": 24,
        "authors": "D. Kwiatkowski, P. C. B. Phillips, P. Schmidt, and Y. Shin",
        "title": "Testing the Null Hypothesis of Stationarity Against the Alternative of a Unit Root",
        "journal": "J. Econometrics",
        "year": 1992,
        "vol": "54(1–3)",
        "pages": "159–178",
        "doi": "10.1016/0304-4076(92)90104-Y",
        "used_in": "EDA — KPSS stationarity test (complementary to ADF)",
        "context": "KPSS kiểm tra H₀: chuỗi là dừng (ngược với ADF). Kết hợp ADF + KPSS giúp phân biệt 3 trường hợp: dừng, trend-stationary, hay có unit root — tránh kết luận sai khi chỉ dùng 1 test đơn lẻ.",
    },
    # ── NEW: Ljung-Box ──
    "ljung1978": {
        "id": 25,
        "authors": "G. M. Ljung and G. E. P. Box",
        "title": "On a Measure of Lack of Fit in Time Series Models",
        "journal": "Biometrika",
        "year": 1978,
        "vol": "65(2)",
        "pages": "297–303",
        "doi": "10.1093/biomet/65.2.297",
        "used_in": "Evaluation — Residual autocorrelation test",
        "context": "Kiểm định Ljung-Box kiểm tra H₀: phần dư (residuals) là nhiễu trắng (white noise). Nếu p < 0,05, phần dư còn cấu trúc tự tương quan → mô hình chưa khai thác hết thông tin. Trong luận văn, tất cả mô hình đều có Ljung-Box p ≈ 0 do PM2.5 có phân phối đuôi dài.",
    },
    # ── NEW: Wolpert Stacking ──
    "wolpert1992": {
        "id": 26,
        "authors": "D. H. Wolpert",
        "title": "Stacked Generalization",
        "journal": "Neural Networks",
        "year": 1992,
        "vol": "5(2)",
        "pages": "241–259",
        "doi": "10.1016/S0893-6080(05)80023-1",
        "used_in": "Models — Ensemble_Stack (meta-learner Ridge)",
        "context": "Stacked Generalization (Stacking) sử dụng mô hình cấp 2 (meta-learner) để học cách kết hợp tối ưu các dự báo từ nhiều mô hình cơ sở. Luận văn dùng Ridge regression làm meta-learner, đạt MASE = 0,70 tại h=24 — ổn định hơn từng mô hình đơn lẻ.",
    },
    # ── NEW: Breiman Random Forests ──
    "breiman2001": {
        "id": 27,
        "authors": "L. Breiman",
        "title": "Random Forests",
        "journal": "Machine Learning",
        "year": 2001,
        "vol": "45(1)",
        "pages": "5–32",
        "doi": "10.1023/A:1010933404324",
        "used_in": "Models — Random Forest baseline",
        "context": "Random Forests xây dựng nhiều cây quyết định độc lập trên các tập bootstrap (bagging) và chọn ngẫu nhiên đặc trưng tại mỗi nút chia. Kỹ thuật này giảm phương sai đáng kể so với cây đơn lẻ, là baseline ổn định cho dự báo dạng bảng (tabular data).",
    },
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Pipeline Step Framework — Circled Numbers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Unicode circled numbers: visually distinct from IEEE [number] citations.
_CIRCLED = {
    1: "①", 2: "②", 3: "③", 4: "④", 5: "⑤",
    6: "⑥", 7: "⑦", 8: "⑧", 9: "⑨", 10: "⑩",
    11: "⑪", 12: "⑫", 13: "⑬", 14: "⑭", 15: "⑮",
    16: "⑯", 17: "⑰", 18: "⑱", 19: "⑲", 20: "⑳",
}


def step(n: int) -> str:
    """Return an HTML-styled circled number for a pipeline step.

    This function is the **single source of truth** for rendering pipeline
    step indicators.  Changing the style here updates every step on every
    dashboard page automatically.

    Args:
        n: Step number (1–20).

    Returns:
        HTML ``<span>`` with the circled number and consistent styling.
    """
    icon = _CIRCLED.get(n, f"({n})")
    return (
        f'<span class="pipeline-step" '
        f'style="display:inline-block; '
        f'background:rgba(255,149,0,0.25); '
        f'color:#FF9500; '
        f'padding:0 0.35rem; '
        f'border-radius:4px; '
        f'font-size:0.85rem; '
        f'font-weight:700; '
        f'margin:0 2px; '
        f'font-family:\'Inter\',sans-serif;"'
        f'>{icon}</span>'
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CSS for tooltip (inject once per page)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


CITATION_CSS = """
<style>
.cite-tooltip {
    position: relative;
    display: inline-block;
    cursor: help;
    background: rgba(0,212,170,0.15);
    color: #00D4AA;
    padding: 0 0.35rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    vertical-align: super;
    line-height: 1;
    margin: 0 1px;
}
.cite-tooltip .cite-content {
    visibility: hidden;
    opacity: 0;
    position: absolute;
    bottom: 125%;
    left: -10px;
    background: #1A1D23;
    color: #E0E0E0;
    border: 1px solid rgba(0,212,170,0.3);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    font-size: 0.8rem;
    font-weight: 400;
    font-family: 'Inter', sans-serif;
    width: max-content;
    max-width: min(380px, 85vw);
    z-index: 99999;
    transition: opacity 0.2s ease;
    box-shadow: 0 8px 24px rgba(0,0,0,0.4);
    line-height: 1.5;
    text-align: left;
    pointer-events: auto; /* Allow clicking links inside tooltip */
}
.cite-tooltip:hover {
    z-index: 99999;
}
/* Fix Streamlit tab indicator bleed-through */
[data-baseweb="tab-list"] {
    z-index: 0 !important;
}
div[data-testid="stMarkdownContainer"] {
    z-index: 10;
}
.cite-tooltip:hover .cite-content {
    visibility: visible;
    opacity: 1;
}
.cite-content .cite-title {
    color: #00D4AA;
    font-weight: 600;
    margin-bottom: 0.25rem;
}
.cite-content .cite-meta {
    color: #8B95A5;
    font-size: 0.75rem;
}
.cite-content .cite-used {
    color: #FFE66D;
    font-size: 0.75rem;
    margin-top: 0.5rem;
    font-weight: 600;
}
.cite-content .cite-context {
    color: #E0E0E0;
    font-size: 0.75rem;
    margin-top: 0.3rem;
    line-height: 1.5;
    border-top: 1px dashed rgba(255,255,255,0.15);
    padding-top: 0.4rem;
}
.cite-content .cite-link {
    display: inline-block;
    margin-top: 0.5rem;
    font-size: 0.75rem;
    color: #4ECDC4;
    text-decoration: none;
    font-weight: 600;
}
.cite-content .cite-link:hover {
    text-decoration: underline;
}
</style>
"""


def _ensure_css():
    """Inject citation CSS. We inject it every time to ensure it persists across Streamlit reruns."""
    st.markdown(CITATION_CSS, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Public API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def cite(ref_id: str) -> str:
    """Return an inline HTML tooltip for a citation.

    Args:
        ref_id: Key in IEEE_REFS dictionary (e.g. "hyndman2006").

    Returns:
        HTML string like ``<span class="cite-tooltip">[1]<span...>...</span></span>``
    """
    _ensure_css()

    ref = IEEE_REFS.get(ref_id)
    if ref is None:
        return f'<span style="color:#FF6B6B">[?{ref_id}]</span>'

    # Build IEEE-style citation text
    vol_info = f", vol. {ref['vol']}" if ref["vol"] else ""
    page_info = f", pp. {ref['pages']}" if ref["pages"] else ""
    doi_info = f'<br>DOI: {ref["doi"]}' if ref["doi"] else ""
    used = ref.get("used_in", "")
    context = ref.get("context", "")
    
    context_html = f'<div class="cite-context">{context}</div>' if context else ""
    
    if ref["doi"]:
        link_html = f'<a href="https://doi.org/{ref["doi"]}" target="_blank" class="cite-link">📖 Đọc tài liệu (DOI) ↗</a>'
    else:
        # Fallback to Google Scholar search by title if no DOI exists
        search_query = urllib.parse.quote_plus(ref["title"])
        link_html = f'<a href="https://scholar.google.com/scholar?q={search_query}" target="_blank" class="cite-link" title="Tìm trên Google Scholar">🔍 Tìm sách/tài liệu: {ref["title"]} ↗</a>'

    tooltip = (
        f'<span class="cite-content">'
        f'<div class="cite-title">{ref["title"]}</div>'
        f'<div class="cite-meta">{ref["authors"]}, '
        f'<em>{ref["journal"]}</em>{vol_info}{page_info}, {ref["year"]}.'
        f'{doi_info}</div>'
        f'<div class="cite-used">📌 Áp dụng: {used}</div>'
        f'{context_html}'
        f'{link_html}'
        f'</span>'
    )

    return f'<span class="cite-tooltip">[{ref["id"]}]{tooltip}</span>'


def render_references_section():
    """Render a full IEEE-formatted references list at the bottom of a page."""
    _ensure_css()

    sorted_refs = sorted(IEEE_REFS.values(), key=lambda r: r["id"])

    rows = []
    for ref in sorted_refs:
        vol_info = f", vol. {ref['vol']}" if ref["vol"] else ""
        page_info = f", pp. {ref['pages']}" if ref["pages"] else ""
        if ref["doi"]:
            doi_link = f' <a href="https://doi.org/{ref["doi"]}" target="_blank" style="color:#60A5FA; text-decoration:none;">DOI↗</a>'
        else:
            search_query = urllib.parse.quote_plus(ref["title"])
            doi_link = f' <a href="https://scholar.google.com/scholar?q={search_query}" target="_blank" style="color:#60A5FA; text-decoration:none;">Scholar↗</a>'

        rows.append(
            f'<div style="margin-bottom:0.5rem; font-size:0.85rem; line-height:1.5;">'
            f'<span style="color:#00D4AA; font-weight:700;">[{ref["id"]}]</span> '
            f'{ref["authors"]}, '
            f'"<em>{ref["title"]}</em>," '
            f'<em>{ref["journal"]}</em>{vol_info}{page_info}, {ref["year"]}.'
            f'{doi_link}'
            f'</div>'
        )

    st.markdown(
        '<div style="margin-top:2rem; padding-top:1rem; '
        'border-top:2px solid rgba(0,212,170,0.2);">'
        '<h3 style="color:#00D4AA; font-size:1.1rem;">📚 Tài Liệu Tham Khảo (IEEE)</h3>'
        + "\n".join(rows)
        + "</div>",
        unsafe_allow_html=True,
    )
