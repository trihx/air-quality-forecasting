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
        "id": 20,
        "authors": "R. J. Hyndman and A. B. Koehler",
        "title": "Another look at measures of forecast accuracy",
        "journal": "Int. J. Forecasting",
        "year": 2006,
        "vol": "22(4)",
        "pages": "679–688",
        "doi": "10.1016/j.ijforecast.2006.03.001",
        "used_in": "Evaluation — MASE metric definition",
        "context": "MASE (Mean Absolute Scaled Error) là thước đo scale-free chuẩn mực. Bằng cách chia sai số dự báo cho sai số của mô hình Naive, MASE khắc phục lỗi chia cho 0 của MAPE và ít bị nhiễu bởi outliers, cho phép so sánh chéo nhiều tập dữ liệu.",
        "pdf_link": "https://robjhyndman.com/papers/mase.pdf",
    },
    "willmott2005": {
        "id": 21,
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
        "id": 22,
        "authors": "T. Gneiting and A. E. Raftery",
        "title": "Strictly proper scoring rules, prediction, and estimation",
        "journal": "J. Amer. Statistical Assoc.",
        "year": 2007,
        "vol": "102(477)",
        "pages": "359–378",
        "doi": "10.1198/016214506000001437",
        "used_in": "Evaluation — CRPS for probabilistic forecasts",
        "context": "CRPS (Continuous Ranked Probability Score) là thước đo độ chính xác cho dự báo phân phối xác suất. CRPS trừng phạt nghiêm ngặt cả việc mô hình thiếu tự tin (quá rộng) hoặc tự tin thái quá (quá hẹp), giúp đánh giá chất lượng của khoảng tin cậy (Prediction Intervals).",
        "quote": "Strictly proper scoring rules encourage the forecaster to make careful assessments and to be honest.",
        "location": "Abstract & Section 1",
        "pdf_link": "https://stat.uw.edu/sites/default/files/files/reports/2004/tr463.pdf",
    },
    "romano2019": {
        "id": 40,
        "authors": "Y. Romano, E. Patterson, and E. J. Candès",
        "title": "Conformalized quantile regression",
        "journal": "Advances in Neural Information Processing Systems",
        "year": 2019,
        "vol": "32",
        "pages": "",
        "doi": "",
        "used_in": "Prediction Intervals — CQR calibration",
        "context": "CQR (Conformalized Quantile Regression) kết hợp hồi quy phân vị (Quantile Regression) với Conformal Prediction. Kỹ thuật này giúp hiệu chỉnh (calibrate) các khoảng tin cậy để đảm bảo độ phủ biên (marginal coverage) luôn đạt mức kỳ vọng (vd: 90%) mà không cần giả định về phân phối của dữ liệu.",
        "quote": "Conformalized quantile regression is a generic method for constructing predictive intervals that attain valid marginal coverage without any assumptions on the distribution.",
        "location": "Abstract",
        "pdf_link": "https://arxiv.org/pdf/1905.03222.pdf",
    },
    "cho2014": {
        "id": 35,
        "authors": "K. Cho, B. van Merrienboer, C. Gulcehre ... Y. Bengio",
        "title": "Learning phrase representations using RNN encoder-decoder for statistical machine translation",
        "journal": "Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
        "year": 2014,
        "vol": "",
        "pages": "",
        "doi": "10.3115/v1/D14-1179",
        "used_in": "Models — GRU architecture",
        "context": "GRU (Gated Recurrent Unit) là một biến thể tối ưu của RNN, kết hợp Forget Gate và Input Gate thành một Update Gate duy nhất. GRU giảm đáng kể số lượng tham số so với LSTM, giúp huấn luyện nhanh hơn và chống overfit tốt hơn trên các bộ dữ liệu chuỗi thời gian có kích thước vừa và nhỏ.",
        "pdf_link": "https://arxiv.org/pdf/1406.1078.pdf",
    },
    "ke2017": {
        "id": 28,
        "authors": "G. Ke, Q. Meng, T. Finley ... T.Y. Liu",
        "title": "LightGBM: A highly efficient gradient boosting decision tree",
        "journal": "Advances in Neural Information Processing Systems",
        "year": 2017,
        "vol": "30",
        "pages": "",
        "doi": "",
        "used_in": "Models — LightGBM tree-based baseline",
        "context": "LightGBM sử dụng chiến lược phát triển cây theo lá (Leaf-wise) kết hợp thuật toán GOSS (Gradient-based One-Side Sampling). Cơ chế này giúp mô hình đạt tốc độ huấn luyện vượt trội và độ chính xác cao đối với dữ liệu dạng bảng (tabular data), đặc biệt khi có các đặc trưng lag/rolling phức tạp.",
        "pdf_link": "https://papers.nips.cc/paper_files/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf",
    },
    "hochreiter1997": {
        "id": 26,
        "authors": "S. Hochreiter and J. Schmidhuber",
        "title": "Long short-term memory",
        "journal": "Neural Computation",
        "year": 1997,
        "vol": "9(8)",
        "pages": "1735–1780",
        "doi": "10.1162/neco.1997.9.8.1735",
        "used_in": "Models — LSTM architecture",
        "context": "LSTM (Long Short-Term Memory) giải quyết triệt để bài toán suy giảm đạo hàm (Vanishing Gradient) trong chuỗi thời gian dài. Nhờ hệ thống Cell State và 3 cổng điều khiển (Input, Output, Forget), LSTM có khả năng 'ghi nhớ' các chu kỳ ô nhiễm PM2.5 kéo dài (ví dụ: chu kỳ mùa, ngày đêm).",
        "quote": "LSTM can learn to bridge time intervals in excess of 1000 steps even in case of noisy, incompressible input sequences, without loss of short time lag capabilities.",
        "location": "Abstract",
        "pdf_link": "http://www.bioinf.jku.at/publications/older/2604.pdf",
    },
    "lim2021": {
        "id": 27,
        "authors": "B. Lim, S.O. Arik, N. Loeff and T. Pfister",
        "title": "Temporal Fusion Transformers for interpretable multi-horizon time series forecasting",
        "journal": "Int. J. Forecasting",
        "year": 2021,
        "vol": "37(4)",
        "pages": "1748–1764",
        "doi": "10.1016/j.ijforecast.2021.03.012",
        "used_in": "Models — TFT architecture",
        "context": "TFT (Temporal Fusion Transformer) kết hợp mạng RNN (để nắm bắt xu hướng cục bộ) và cơ chế Self-Attention (để học phụ thuộc xa). Khác với các mô hình Black-box, TFT cung cấp khả năng diễn giải mạnh mẽ thông qua Variable Selection Network (đánh giá tầm quan trọng của từng đặc trưng đầu vào).",
        "pdf_link": "https://arxiv.org/pdf/1912.09363.pdf",
    },
    "peixeiro2022": {
        "id": 16,
        "authors": "M. Peixeiro",
        "title": "Time Series Forecasting in Python",
        "journal": "Manning Publications",
        "year": 2022,
        "vol": "",
        "pages": "",
        "doi": "",
        "used_in": "Pipeline — End-to-end forecasting methodology",
        "context": "Nghiên cứu áp dụng quy trình đánh giá chuẩn mực của Peixeiro: Xây dựng đường cơ sở tĩnh (Persistence) → Mô hình thống kê (ARIMA/SARIMA) → Machine Learning (LightGBM) → Deep Learning (GRU/LSTM/TFT) để chứng minh sự gia tăng hiệu suất thực sự của các mô hình phức tạp.",
        "quote": "Deep learning architectures have revolutionized time series forecasting by automatically learning temporal representations and handling complex non-linear relationships.",
        "location": "Chapter 12: Deep Learning for Time Series",
    },
    "shumway2017": {
        "id": 6,
        "authors": "R. H. Shumway and D. S. Stoffer",
        "title": "Time Series Analysis and Its Applications: With R Examples",
        "journal": "Springer",
        "year": 2017,
        "vol": "4th ed.",
        "pages": "",
        "doi": "10.1007/978-3-319-52452-8",
        "used_in": "EDA — Statistical foundations (ADF, KPSS, ACF/PACF)",
        "context": "Nền tảng lý thuyết cho chuỗi thời gian: Sử dụng kiểm định ADF (Augmented Dickey-Fuller) và KPSS để xác minh tính dừng (stationarity). Phân tích ACF/PACF giúp xác định chính xác các bước trễ (lag) nội tại trong nồng độ PM2.5 do tính tự tương quan (autocorrelation).",
        "quote": "Autocorrelation and cross-correlation functions provide a fundamental measure of the linear dependence between time series at different temporal lags.",
        "location": "Chapter 1: Characteristics of Time Series",
    },
    "diebold1995": {
        "id": 23,
        "authors": "F. X. Diebold and R. S. Mariano",
        "title": "Comparing predictive accuracy",
        "journal": "J. Bus. Econ. Stat.",
        "year": 1995,
        "vol": "13(3)",
        "pages": "253–263",
        "doi": "10.1080/07350015.1995.10524599",
        "used_in": "Evaluation — Diebold-Mariano test for model comparison",
        "context": "Kiểm định Diebold-Mariano (DM Test) được sử dụng để xác minh xem sự chênh lệch độ chính xác giữa 2 mô hình (vd: GRU so với LightGBM) có thực sự mang ý nghĩa thống kê (statistically significant) hay chỉ do sự tình cờ của nhiễu dữ liệu tập test.",
        "quote": "We propose tests of the null hypothesis of no difference in the accuracy of two competing forecasts, valid under a wide variety of assumptions.",
        "location": "Abstract",
    },
    "akiba2019": {
        "id": 33,
        "authors": "T. Akiba, S. Sano, T. Yanase, T. Ohta and M. Koyama",
        "title": "Optuna: A next-generation hyperparameter optimization framework",
        "journal": "Proc. ACM SIGKDD",
        "year": 2019,
        "vol": "",
        "pages": "2623–2631",
        "doi": "10.1145/3292500.3330701",
        "used_in": "Training — Hyperparameter optimization with TPE sampler",
        "context": "Optuna là framework tối ưu hóa siêu tham số thế hệ mới. Áp dụng thuật toán lấy mẫu TPE (Tree-structured Parzen Estimator) dựa trên tối ưu hóa Bayes, Optuna có khả năng tự động khám phá không gian tìm kiếm và cắt tỉa (pruning) sớm các phép thử nghiệm kém hiệu quả, tiết kiệm đáng kể tài nguyên tính toán.",
        "quote": "Optuna is a next-generation hyperparameter optimization software that defines the search space by the define-by-run API, allowing users to dynamically construct search spaces.",
        "location": "Abstract",
        "pdf_link": "https://arxiv.org/pdf/1907.10902.pdf",
    },
    # ── Thesis [13] Box-Cox ──
    "boxcox1964": {
        "id": 7,
        "authors": "G. E. P. Box and D. R. Cox",
        "title": "An Analysis of Transformations",
        "journal": "J. Royal Statistical Soc. B",
        "year": 1964,
        "vol": "26(2)",
        "pages": "211–252",
        "doi": "10.1111/j.2517-6161.1964.tb00553.x",
        "used_in": "Pipeline — Box-Cox transform for fat-tailed PM2.5",
        "context": "Phép biến đổi Box-Cox tìm giá trị λ tối ưu để ổn định phương sai của biến mục tiêu. Với PM2.5 (λ ≈ −0.147 ≈ 0), phép Log Transform giúp thu hẹp biên độ các đỉnh ô nhiễm cực đoan, cải thiện hiệu suất mô hình Học sâu.",
        "quote": "In this paper we make the less restrictive assumption that such a normal, homoscedastic, linear model is appropriate after some suitable transformation has been applied to the y's.",
        "location": "Abstract",
        "pdf_link": "docs/references/[13] Box & Cox 1964_Box-Cox Transform.pdf",
    },
    # ── Thesis [15] Rosner — S-ESD ──
    "rosner1983": {
        "id": 8,
        "authors": "B. Rosner",
        "title": "Percentage Points for a Generalized ESD Many-Outlier Procedure",
        "journal": "Technometrics",
        "year": 1983,
        "vol": "25(2)",
        "pages": "165–172",
        "doi": "10.1080/00401706.1983.10487848",
        "used_in": "Pipeline ② — Outlier detection (S-ESD / MAD)",
        "context": "Thủ tục ESD tổng quát (Generalized ESD) phát hiện đa ngoại lệ đồng thời. Luận văn áp dụng biến thể Seasonal-ESD kết hợp STL detrend + MAD để bảo toàn các đỉnh ô nhiễm thực sự của PM2.5, tránh cắt tín hiệu sinh thái.",
        "quote": "A generalized (extreme Studentized deviate) ESD many-outlier procedure is given for detecting from 1 to k outliers in a data set. This procedure controls the type I error both under the hypothesis of no outliers and under the alternative hypotheses of 1, 2, …. k-1 outliers.",
        "location": "Abstract",
        "pdf_link": "docs/references/[14] Rosner 1983_S-ESD Outliers.pdf",
    },
    # ── Thesis [4] Tashman ──
    "tashman2000": {
        "id": 18,
        "authors": "L. J. Tashman",
        "title": "Out-of-sample tests of forecasting accuracy: an analysis and review",
        "journal": "Int. J. Forecasting",
        "year": 2000,
        "vol": "16(4)",
        "pages": "437–450",
        "doi": "10.1016/S0169-2070(00)00065-0",
        "used_in": "Pipeline ⑤ — Out-of-sample evaluation design",
        "context": "Bài tổng quan đặt nền tảng cho phương pháp đánh giá dự báo ngoài mẫu (out-of-sample). Luận văn áp dụng nguyên tắc: tập Test tách biệt hoàn toàn, chỉ dùng dữ liệu thật (real-only), không bao giờ dùng data đã impute để đánh giá.",
        "quote": "The efficiency and reliability of out-of-sample tests for individual time series can be improved by employing rolling-origin evaluations, recalibrating coefficients, and using multiple test periods.",
        "location": "Abstract",
        "pdf_link": "docs/references/[15] Tashman 2000_Out-of-sample Testing.pdf",
    },
    # ── Thesis [5] Hyndman FPP3 ──
    "hyndman2021": {
        "id": 17,
        "authors": "R. J. Hyndman and G. Athanasopoulos",
        "title": "Forecasting: Principles and Practice",
        "journal": "OTexts, 3rd ed.",
        "year": 2021,
        "vol": "",
        "pages": "",
        "doi": "",
        "used_in": "Pipeline ④⑤ — Walk-forward CV, anti-leakage design",
        "context": "Sách giáo khoa chuẩn mực về dự báo chuỗi thời gian. Luận văn áp dụng: (a) Walk-Forward Expanding Window, (b) shift(1) anti-leakage, (c) Purging Gap, và (d) Sử dụng Naïve/Persistence method làm baseline vững chắc cho dữ liệu tự tương quan cao.",
        "quote": "When cross-validating time series models, the training set must only include observations that occurred prior to the test set to avoid future leakage.",
        "location": "Chapter 5: The forecaster's toolbox",
    },
    # ── Thesis [6] M4 Competition ──
    "makridakis2020": {
        "id": 19,
        "authors": "S. Makridakis, E. Spiliotis, and V. Assimakopoulos",
        "title": "The M4 Competition: Results, findings, conclusion and way forward",
        "journal": "Int. J. Forecasting",
        "year": 2020,
        "vol": "36(1)",
        "pages": "54–74",
        "doi": "",
        "used_in": "Evaluation — M4 Competition benchmark methodology",
        "context": "Cuộc thi M4 (100.000 chuỗi thời gian) chứng minh: (a) mô hình đơn giản (ETS/ARIMA) vẫn cạnh tranh với ML/DL ở khung ngắn, (b) ensemble luôn thắng single model, (c) MASE là metric chuẩn để so sánh chéo. Luận văn áp dụng cả 3 nguyên tắc này.",
        "quote": "The M4 Competition findings showed that machine learning models did not perform as well as statistical ones, while their combination provided the highest accuracy.",
        "location": "Abstract",
        "pdf_link": "https://forecasters.org/wp-content/uploads/Makridakis_The-M4-Competition.pdf",
    },
    # ── Thesis [17] Gal — MC Dropout ──
    "gal2016": {
        "id": 42,
        "authors": "Y. Gal and Z. Ghahramani",
        "title": "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning",
        "journal": "Proc. ICML",
        "year": 2016,
        "vol": "48",
        "pages": "1050–1059",
        "doi": "",
        "used_in": "Prediction Intervals — MC Dropout (đã loại bỏ)",
        "context": "MC Dropout xấp xỉ Bayesian bằng cách bật Dropout lúc inference và chạy nhiều lần. Tuy nhiên, trên dataset PM2.5 nhỏ (7K mẫu), phương sai Dropout quá nhỏ → khoảng tin cậy cực hẹp (coverage 7,6% thay vì 90%). Luận văn đã thay thế bằng CQR.",
        "quote": "We show that a neural network with arbitrary depth and non-linearities, with dropout applied before every weight layer, is mathematically equivalent to an approximation to the probabilistic deep Gaussian process.",
        "location": "Abstract",
        "pdf_link": "https://arxiv.org/pdf/1506.02142.pdf",
    },
    # ── Thesis [19] Deep Ensembles ──
    "lakshminarayanan2017": {
        "id": 43,
        "authors": "B. Lakshminarayanan, A. Pritzel, and C. Blundell",
        "title": "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles",
        "journal": "Advances in NeurIPS",
        "year": 2017,
        "vol": "30",
        "pages": "",
        "doi": "",
        "used_in": "Prediction Intervals — Deep Ensembles reference",
        "context": "Deep Ensembles ước lượng bất định bằng cách huấn luyện N mô hình độc lập rồi tổng hợp phân phối dự báo. Luận văn dùng GRU 5-seed ensemble (seeds: 42, 123, 456, 789, 2024) theo nguyên tắc này, đạt MASE = 0,72 tại h=24.",
        "quote": "Deep ensembles provide a simple, scalable method for estimating predictive uncertainty and are highly robust to dataset shift compared to Bayesian Neural Networks.",
        "location": "Abstract",
        "pdf_link": "https://arxiv.org/pdf/1612.01474.pdf",
    },
    "gibbs2021": {
        "id": 41,
        "authors": "I. Gibbs and E. J. Candès",
        "title": "Adaptive conformal inference under distribution shift",
        "journal": "NeurIPS",
        "year": 2021,
        "vol": "34",
        "pages": "1614–1626",
        "doi": "",
        "used_in": "Uncertainty Estimation — Adaptive prediction intervals",
        "context": "Adaptive Conformal Inference (ACI) điều chỉnh động độ rộng của khoảng dự báo dựa trên sai số của các dự báo gần nhất, giúp phương pháp Conformal Prediction chống lại distribution shift trong dữ liệu chuỗi thời gian.",
        "quote": "We present adaptive conformal inference (ACI), a method that achieves exact marginal coverage even when the data distribution shifts arbitrarily over time.",
        "location": "Abstract",
        "pdf_link": "https://arxiv.org/pdf/2102.10443.pdf",
    },
    # ── NEW: Lundberg SHAP ──
    "lundberg2017": {
        "id": 38,
        "authors": "S. M. Lundberg and S.-I. Lee",
        "title": "A Unified Approach to Interpreting Model Predictions",
        "journal": "Advances in NeurIPS",
        "year": 2017,
        "vol": "30",
        "pages": "4765–4774",
        "doi": "",
        "used_in": "Explainability — SHAP feature importance",
        "context": "SHAP (SHapley Additive exPlanations) dựa trên lý thuyết trò chơi Shapley để phân bổ công bằng đóng góp của từng đặc trưng vào dự báo. TreeExplainer cho LightGBM tính SHAP values chính xác trong O(TLD²), giúp giải thích tại sao pm25_lag_1h chi phối hoàn toàn ở h=1.",
        "quote": "SHAP values provide a unified measure of feature importance that guarantees both local accuracy and consistency across model interpretations.",
        "location": "Section 4: SHAP Values",
        "pdf_link": "https://papers.nips.cc/paper_files/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf",
    },
    # ── NEW: Cleveland STL ──
    "cleveland1990": {
        "id": 9,
        "authors": "R. B. Cleveland, W. S. Cleveland, J. E. McRae, and I. Terpenning",
        "title": "STL: A Seasonal-Trend Decomposition Procedure Based on Loess",
        "journal": "J. Official Statistics",
        "year": 1990,
        "vol": "6(1)",
        "pages": "3–73",
        "doi": "",
        "used_in": "EDA — Seasonal-Trend decomposition",
        "context": "STL phân tách chuỗi thời gian thành 3 thành phần: Trend, Seasonality, Remainder bằng thuật toán Loess (locally weighted regression). Luận văn dùng STL (period=24, robust=True) để xác nhận chu kỳ ngày/đêm của PM2.5 và phát hiện leakage khi fit STL trên toàn bộ data.",
        "quote": "STL is a filtering procedure for decomposing a seasonal time series into three components: trend, seasonal, and remainder. STL consists of a sequence of applications of the loess smoother.",
        "location": "Abstract",
        "pdf_link": "https://www.scb.se/contentassets/ca21efb41fee47d293bbee5bf7be7fb3/stl-a-seasonal-trend-decomposition-procedure-based-on-loess.pdf",
    },
    # ── NEW: Troyanskaya KNN Imputation ──
    "troyanskaya2001": {
        "id": 15,
        "authors": "O. Troyanskaya, M. Cantor, G. Sherlock ... R.B. Altman",
        "title": "Missing Value Estimation Methods for DNA Microarrays",
        "journal": "Bioinformatics",
        "year": 2001,
        "vol": "17(6)",
        "pages": "520–525",
        "doi": "10.1093/bioinformatics/17.6.520",
        "used_in": "Pipeline ③ — KNN Imputation (gap 6–24h)",
        "context": "KNN Imputation ước lượng giá trị thiếu bằng trung bình có trọng số (distance-weighted) của K hàng xóm gần nhất trong không gian đa biến. Luận văn áp dụng KNNImputer(n_neighbors=5) cho gap 6–24h, tận dụng tương quan chéo giữa PM2.5, nhiệt độ, độ ẩm, CO2.",
        "quote": "We present a comparative study of several methods for the estimation of missing values... We show that KNNimpute appears to provide a more robust and sensitive method for missing value estimation.",
        "location": "Abstract",
        "pdf_link": "docs/references/[22] Troyanskaya 2001_KNN Imputation.pdf",
    },
    # ── NEW: Dickey-Fuller ADF ──
    "dickey1979": {
        "id": 11,
        "authors": "D. A. Dickey and W. A. Fuller",
        "title": "Distribution of the Estimators for Autoregressive Time Series with a Unit Root",
        "journal": "J. Amer. Statistical Assoc.",
        "year": 1979,
        "vol": "74(366)",
        "pages": "427–431",
        "doi": "10.1080/01621459.1979.10482531",
        "used_in": "EDA — Augmented Dickey-Fuller (ADF) stationarity test",
        "context": "Kiểm định ADF kiểm tra H₀: chuỗi có unit root (không dừng). Luận văn kết hợp ADF + KPSS (2 chiều giả thuyết) để xác nhận PM2.5 là trend-stationary (ADF reject H₀, KPSS reject H₀ stationarity) — cho thấy cần sai phân trước khi ARIMA.",
        "quote": "The hypothesis that a time series is a random walk versus the alternative of stationarity can be tested using the Dickey-Fuller distribution.",
        "location": "Abstract",
    },
    # ── NEW: KPSS ──
    "kwiatkowski1992": {
        "id": 12,
        "authors": "D. Kwiatkowski, P. C. B. Phillips, P. Schmidt, and Y. Shin",
        "title": "Testing the Null Hypothesis of Stationarity Against the Alternative of a Unit Root",
        "journal": "J. Econometrics",
        "year": 1992,
        "vol": "54(1–3)",
        "pages": "159–178",
        "doi": "10.1016/0304-4076(92)90104-Y",
        "used_in": "EDA — KPSS stationarity test (complementary to ADF)",
        "context": "KPSS kiểm tra H₀: chuỗi là dừng (ngược với ADF). Kết hợp ADF + KPSS giúp phân biệt 3 trường hợp: dừng, trend-stationary, hay có unit root — tránh kết luận sai khi chỉ dùng 1 test đơn lẻ.",
        "quote": "We propose a test of the null hypothesis that an observable series is stationary around a deterministic trend against the alternative of a unit root.",
        "location": "Abstract",
    },
    # ── NEW: Ljung-Box ──
    "ljung1978": {
        "id": 13,
        "authors": "G. M. Ljung and G. E. P. Box",
        "title": "On a Measure of Lack of Fit in Time Series Models",
        "journal": "Biometrika",
        "year": 1978,
        "vol": "65(2)",
        "pages": "297–303",
        "doi": "10.1093/biomet/65.2.297",
        "used_in": "Evaluation — Residual autocorrelation test",
        "context": "Kiểm định Ljung-Box kiểm tra H₀: phần dư (residuals) là nhiễu trắng (white noise). Nếu p < 0,05, phần dư còn cấu trúc tự tương quan → mô hình chưa khai thác hết thông tin. Trong luận văn, tất cả mô hình đều có Ljung-Box p ≈ 0 do PM2.5 có phân phối đuôi dài.",
        "quote": "A modified portmanteau test is proposed that provides a more accurate approximation to the exact distribution for testing goodness of fit in time series models.",
        "location": "Abstract",
    },
    # ── NEW: Wolpert Stacking ──
    "wolpert1992": {
        "id": 31,
        "authors": "D. H. Wolpert",
        "title": "Stacked Generalization",
        "journal": "Neural Networks",
        "year": 1992,
        "vol": "5(2)",
        "pages": "241–259",
        "doi": "10.1016/S0893-6080(05)80023-1",
        "used_in": "Models — Ensemble_Stack (meta-learner Ridge)",
        "context": "Stacked Generalization (Stacking) sử dụng mô hình cấp 2 (meta-learner) để học cách kết hợp tối ưu các dự báo từ nhiều mô hình cơ sở. Luận văn dùng Ridge regression làm meta-learner, đạt MASE = 0,70 tại h=24 — ổn định hơn từng mô hình đơn lẻ.",
        "quote": "Stacked generalization works by deducing the biases of the generalizer(s) with respect to a provided learning set and using this to correct for those biases.",
        "location": "Abstract",
    },
    # ── NEW: Breiman Random Forests ──
    "breiman2001": {
        "id": 29,
        "authors": "L. Breiman",
        "title": "Random Forests",
        "journal": "Machine Learning",
        "year": 2001,
        "vol": "45(1)",
        "pages": "5–32",
        "doi": "10.1023/A:1010933404324",
        "used_in": "Models — Random Forest baseline",
        "context": "Random Forests xây dựng nhiều cây quyết định độc lập trên các tập bootstrap (bagging) và chọn ngẫu nhiên đặc trưng tại mỗi nút chia. Kỹ thuật này giảm phương sai đáng kể so với cây đơn lẻ, là baseline ổn định cho dự báo dạng bảng (tabular data).",
        "quote": "Random forests are a combination of tree predictors such that each tree depends on the values of a random vector sampled independently and with the same distribution for all trees.",
        "location": "Abstract",
    },
    "christ2018": {
        "id": 32,
        "authors": "M. Christ, N. Braun, J. Neuffer, and A. W. Kempa-Liehr",
        "title": "Time Series FeatuRe Extraction on basis of Scalable Hypothesis tests (tsfresh - A Python package)",
        "journal": "Neurocomputing",
        "year": 2018,
        "vol": "307",
        "pages": "72-77",
        "doi": "10.1016/j.neucom.2018.03.067",
        "used_in": "Pipeline - Feature Engineering",
        "context": "Mô tả sự cần thiết của việc trích xuất đồng thời hàng chục đến hàng trăm đặc trưng (như lag, rolling statistics, calendar, fourier) từ chuỗi thời gian nhằm nắm bắt toàn diện sự phụ thuộc tuyến tính lẫn phi tuyến tính trước khi đưa vào mô hình Machine Learning.",
        "quote": "The extraction of comprehensive features from time series, such as auto-correlation, rolling statistics, and spectral components, is crucial for improving the performance of machine learning algorithms in classification and regression tasks.",
        "location": "Abstract & Section 1",
        "pdf_link": "https://arxiv.org/pdf/1610.07717.pdf",
    },
    "who2021": {
        "id": 1,
        "authors": "World Health Organization",
        "title": "WHO global air quality guidelines: particulate matter (PM2.5 and PM10), ozone, nitrogen dioxide, sulfur dioxide and carbon monoxide",
        "journal": "WHO Guidelines",
        "year": 2021,
        "vol": "",
        "pages": "",
        "doi": "",
        "url": "https://www.who.int/publications/i/item/9789240034228",
        "used_in": "Data Cleaning - Domain clipping",
        "context": "WHO khuyến nghị giới hạn nồng độ bụi mịn PM2.5 ở mức rất thấp (15 µg/m3 cho trung bình 24h). Tuy nhiên, trên thang đo Air Quality Index (AQI) chuẩn, nồng độ PM2.5 thường được giới hạn tối đa ở mức 500 µg/m3 (mức Hazardous cực độ). Bất kỳ giá trị nào >500 đều được coi là lỗi cảm biến (anomalies) hoặc nằm ngoài domain do lượng vật lý của thiết bị IoT thông dụng.",
        "quote": "The guidelines offer global guidance on thresholds and limits for key air pollutants that pose health risks, providing a clear benchmark for evaluating air quality.",
        "location": "Executive Summary",
    },
    "barkjohn2021": {
        "id": 2,
        "authors": "K. K. Barkjohn, B. Gantt, and A. L. Clements",
        "title": "Development and application of a United States-wide correction for PM2.5 data collected with the PurpleAir sensor",
        "journal": "Atmospheric Measurement Techniques",
        "year": 2021,
        "vol": "14(6)",
        "pages": "4617-4637",
        "doi": "10.5194/amt-14-4617-2021",
        "url": "",
        "used_in": "Data Cleaning - Resampling strategy",
        "context": "Dữ liệu cảm biến bụi mịn IoT (ví dụ PurpleAir) ghi nhận dữ liệu ở tần suất rất cao (khoảng 2 phút/lần). Dữ liệu này thường chứa nhiễu (white noise) và sai số biến thiên liên tục. Nghiên cứu chuẩn hóa của US EPA (Cơ quan Bảo vệ Môi trường Mỹ) chỉ ra rằng việc tái lấy mẫu (resampling) trung bình theo 15m, 30m hoặc 1h giúp làm mịn (smooth out) các nhiễu loạn ngắn hạn, đồng thời đưa dữ liệu về cùng quy chuẩn để có thể so sánh và dự báo chính xác.",
        "quote": "The correction equation and proposed data-cleaning criteria significantly reduced the RMSE of the sensor data from 8 to 3 µg/m3.",
        "location": "Abstract, p. 4617",
        "pdf_link": "https://amt.copernicus.org/articles/14/4617/2021/amt-14-4617-2021.pdf",
    },
    "zhang2017": {
        "id": 3,
        "authors": "Z. Zhang",
        "title": "Multivariate Time Series Analysis in Climate and Environmental Research",
        "journal": "Springer",
        "year": 2017,
        "vol": "",
        "pages": "",
        "doi": "10.1007/978-3-319-67340-0",
        "used_in": "EDA - Mutual Information & Multivariate Analysis",
        "context": "Khác với Pearson chỉ bắt được tương quan tuyến tính, Mutual Information (MI) dựa trên lý thuyết thông tin giúp định lượng sự phụ thuộc phi tuyến. Luận văn áp dụng MI để chứng minh Nhiệt độ và Điểm sương dẫn dắt sự biến thiên của PM2.5, phù hợp với động lực học khí quyển.",
        "quote": "The complexity of climatic and environmental variability across all timescales requires the use of advanced methods to unravel primary dynamics from observations.",
        "location": "Book Overview",
    },
    "zannetti1990": {
        "id": 4,
        "authors": "P. Zannetti",
        "title": "Air Pollution Modeling",
        "journal": "Springer",
        "year": 1990,
        "vol": "",
        "pages": "",
        "doi": "10.1007/978-1-4757-4465-1",
        "used_in": "EDA - Temperature Inversion (Conditional Distribution)",
        "context": "Sách cơ sở về mô hình hóa ô nhiễm không khí. Luận văn áp dụng lý thuyết nghịch nhiệt (temperature inversion): khi nhiệt độ bề mặt thấp, khí quyển ổn định ngăn chặn đối lưu, giữ lại PM2.5 gần mặt đất. Điều này giải thích phân phối đuôi dài (long-tail) của ô nhiễm ở mức nhiệt <26°C.",
        "quote": "Inversions act as a lid on the lower atmosphere, trapping air pollutants such as particulate matter close to the ground and preventing their dispersion.",
        "location": "Chapter 3: Meteorological Dynamics",
    },
    "blanchard2003": {
        "id": 5,
        "authors": "C. L. Blanchard and S. Tanenbaum",
        "title": "Differences between weekday and weekend air pollutant levels in Southern California",
        "journal": "J. Air & Waste Manage. Assoc.",
        "year": 2003,
        "vol": "53(7)",
        "pages": "816-828",
        "doi": "10.1080/10473289.2003.10466222",
        "used_in": "EDA - Weekday vs Weekend Analysis",
        "context": "Hiệu ứng cuối tuần (Weekend Effect) được quan sát khi phát thải giảm nhưng mức độ ô nhiễm không giảm tương ứng. Luận văn sử dụng Boxplot để kiểm chứng: PM2.5 vào T7/CN duy trì ở mức cao tương tự ngày thường, cho thấy ô nhiễm không chỉ do giao thông công sở mà là sự cộng hưởng của nhiều nguồn.",
        "quote": "Weekday-weekend differences in ambient concentrations of primary pollutants provide a means for evaluating emission inventories.",
        "location": "Abstract, p. 816",
    },
    "joseph2022": {
        "id": 34,
        "authors": "M. Joseph",
        "title": "Modern Time Series Forecasting with Python",
        "journal": "Packt Publishing",
        "year": 2022,
        "vol": "",
        "pages": "",
        "doi": "",
        "used_in": "EDA - Forecastability and Complexity Assessment",
        "context": "Cuốn sách cung cấp framework để đánh giá tính khả thi dự báo (Forecastability) và độ phức tạp của chuỗi thời gian (Complexity Profile) trước khi chọn mô hình.",
        "quote": "Before jumping into modeling, it is essential to measure the forecastability and complexity of the time series.",
        "location": "Chapter 4",
    },
    "huang2022": {
        "id": 35,
        "authors": "C. Huang and A. Petukhina",
        "title": "Applied Time Series Analysis and Forecasting with Python",
        "journal": "Springer",
        "year": 2022,
        "vol": "",
        "pages": "",
        "doi": "10.1007/978-3-031-18084-3",
        "used_in": "EDA - Cross-Correlation and Spectral Analysis",
        "context": "Tài liệu tham khảo chuyên sâu về phân tích chuỗi thời gian áp dụng trong Python, sử dụng để xác nhận các frequencies bằng Periodogram và đo lường độ trễ (Lag) bằng Cross-Correlation.",
        "quote": "Spectral analysis provides a complementary view of time series data by identifying the dominant cyclical patterns.",
        "location": "Chapter 7",
    },
    "vishwas2020": {
        "id": 36,
        "authors": "B. V. Vishwas and A. Patel",
        "title": "Hands-on Time Series Analysis with Python",
        "journal": "Apress",
        "year": 2020,
        "vol": "",
        "pages": "",
        "doi": "10.1007/978-1-4842-5992-4",
        "used_in": "EDA - Seasonal Pattern Visualization",
        "context": "Hướng dẫn thực hành phân tích trực quan chuỗi thời gian, áp dụng trong việc phân tích phân phối PM2.5 theo từng giờ trong ngày qua Box Plot.",
        "quote": "Visualizing time series distributions across specific seasonal periods, such as hour of the day, reveals patterns crucial for feature engineering.",
        "location": "Chapter 4",
    },
    "kang2017": {
        "id": 37,
        "authors": "Y. Kang, R. J. Hyndman, and K. Smith-Miles",
        "title": "Visualising forecasting algorithm performance using time series instance spaces",
        "journal": "Int. J. Forecasting",
        "year": 2017,
        "vol": "33(2)",
        "pages": "345-358",
        "doi": "10.1016/j.ijforecast.2016.09.004",
        "used_in": "EDA - Time series complexity profiling",
        "context": "Nghiên cứu về không gian đặc trưng (instance space) của chuỗi thời gian, giúp trực quan hóa nhiều chiều độ phức tạp (Complexity Profile) như tính dừng, mùa vụ, nhiễu và độ dài dự báo.",
        "quote": "By projecting multiple time series features into a 2D instance space, we can better understand the diversity and complexity of the forecasting problem.",
        "location": "Abstract",
    },
    "cleveland1993": {
        "id": 10,
        "authors": "W. S. Cleveland",
        "title": "Visualizing Data",
        "journal": "Hobart Press",
        "year": 1993,
        "vol": "",
        "pages": "",
        "doi": "",
        "used_in": "EDA - Scatter Matrix",
        "context": "Cuốn sách kinh điển về trực quan hóa dữ liệu nhiều chiều. Luận văn áp dụng biểu đồ Scatter Matrix (Pairs Plot) để kiểm tra đồng thời mối quan hệ tuyến tính, phi tuyến tính và phân cụm giữa PM2.5 và tất cả các biến môi trường.",
        "quote": "A scatterplot matrix is a powerful tool for discovering the relationships between variables in multivariate data.",
        "location": "Chapter 3",
    },
    "shetty2024": {
        "id": 46,
        "authors": "S. Shetty, P.D. Hamer, K. Stebel ... P. Schneider",
        "title": "Daily high-resolution surface PM2.5 estimation over Europe by ML-based downscaling of the CAMS regional forecast",
        "journal": "Environmental Research",
        "year": 2024,
        "vol": "252",
        "pages": "120363",
        "doi": "10.1016/j.envres.2024.120363",
        "used_in": "Comparison - International",
        "context": "Mô hình dự báo PM2.5 cho toàn bộ Châu Âu sử dụng ML.",
        "quote": "",
        "location": "",
        "pdf_link": "https://sci-hub.se/10.1016/j.envres.2024.120363",
    },
    "tian2024": {
        "id": 47,
        "authors": "H. Tian, H. Kong and C. Wong",
        "title": "A Novel Stacking Ensemble Learning Approach for Predicting PM2.5 Levels in Dense Urban Environments Using Meteorological Variables: A Case Study in Macau",
        "journal": "Applied Sciences",
        "year": 2024,
        "vol": "14",
        "pages": "5062",
        "doi": "10.3390/app14125062",
        "used_in": "Comparison - International",
        "context": "Sử dụng Stacking Ensemble tại môi trường đô thị dày đặc Macau.",
        "quote": "",
        "location": "",
        "pdf_link": "https://www.mdpi.com/2076-3417/14/12/5062/pdf",
    },
    "inam2024": {
        "id": 48,
        "authors": "S.A. Inam, A.A. Khan, T. Mazhar ... H. Hamam",
        "title": "PR-FCNN: a data-driven hybrid approach for predicting PM2.5 concentration",
        "journal": "Earth Science Informatics",
        "year": 2024,
        "vol": "",
        "pages": "",
        "doi": "10.1007/s44163-024-00184-7",
        "used_in": "Comparison - International",
        "context": "Tiếp cận lai tạo (hybrid) dựa trên dữ liệu cho dự báo PM2.5.",
        "quote": "",
        "location": "",
        "pdf_link": "https://sci-hub.se/10.1007/s44163-024-00184-7",
    },
    "kim2023": {
        "id": 49,
        "authors": "B. Kim, E. Kim, S. Jung ... S. Kim",
        "title": "PM2.5 Concentration Forecasting Using Weighted Bi-LSTM and Random Forest Feature Importance-Based Feature Selection",
        "journal": "Atmosphere",
        "year": 2023,
        "vol": "14(6)",
        "pages": "968",
        "doi": "10.3390/atmos14060968",
        "used_in": "Comparison - International",
        "context": "Sử dụng Bi-LSTM và trích chọn đặc trưng bằng Random Forest.",
        "quote": "",
        "location": "",
        "pdf_link": "https://www.mdpi.com/2073-4433/14/6/968/pdf",
    },
    "patel2025": {
        "id": 50,
        "authors": "P. Patel, S. Patel, K. Shah ... S. Patel",
        "title": "A systematic study on PM2.5 and PM10 concentration prediction in air pollution using machine learning and deep learning model",
        "journal": "Environmental Challenges",
        "year": 2025,
        "vol": "",
        "pages": "",
        "doi": "10.1016/j.enceco.2025.07.001",
        "used_in": "Comparison - International",
        "context": "Nghiên cứu hệ thống các mô hình ML/DL cho dự báo PM2.5 và PM10.",
        "quote": "",
        "location": "",
        "pdf_link": "https://sci-hub.se/10.1016/j.enceco.2025.07.001",
    },
    "kaveh2025": {
        "id": 51,
        "authors": "M. Kaveh, M.S. Mesgari and M. Kaveh",
        "title": "A Novel Evolutionary Deep Learning Approach for PM2.5 Prediction Using Remote Sensing and Spatial-Temporal Data: A Case Study of Tehran",
        "journal": "ISPRS Int. J. Geo-Inf.",
        "year": 2025,
        "vol": "14(2)",
        "pages": "42",
        "doi": "10.3390/ijgi14020042",
        "used_in": "Comparison - International",
        "context": "Mô hình DL tiến hóa kết hợp Remote Sensing và Spatial-Temporal Data.",
        "quote": "",
        "location": "",
        "pdf_link": "https://www.mdpi.com/2220-9964/14/2/42/pdf",
    },
    "nguyen2024": {
        "id": 52,
        "authors": "N.T.N. Tuyết, T.T. Dũng, V.P.C.L. Thọ và P.T. Bảo",
        "title": "Statistical and machine learning approaches for estimating pollution of fine particulate matter (PM2.5) in Vietnam",
        "journal": "J. of Environ. Engineering & Landscape Management",
        "year": 2024,
        "vol": "32(4)",
        "pages": "292-304",
        "doi": "10.3846/jeelm.2024.22361",
        "used_in": "Comparison - Vietnam",
        "context": "Mô hình CNN+Bi-LSTM đánh giá độ bao phủ dữ liệu tại Việt Nam.",
        "quote": "",
        "location": "",
        "pdf_link": "https://doi.org/10.3846/jeelm.2024.22361",
    },
    "rakholia2022": {
        "id": 53,
        "authors": "R. Rakholia, Q. Lê, K. Vũ, B.Q. Hồ và R.S. Carbajo",
        "title": "AI-based air quality PM2.5 forecasting models for developing countries: A case study of Ho Chi Minh City, Vietnam",
        "journal": "Urban Climate",
        "year": 2022,
        "vol": "44",
        "pages": "101315",
        "doi": "10.1016/j.uclim.2022.101315",
        "used_in": "Comparison - Vietnam",
        "context": "Nghiên cứu dự báo PM2.5 ứng dụng AI tại Thành phố Hồ Chí Minh.",
        "quote": "",
        "location": "",
        "pdf_link": "https://sci-hub.se/10.1016/j.uclim.2022.101315",
    },
    # ── NEW: Moritz Time Series Imputation ──
    "moritz2015": {
        "id": 14,
        "authors": "S. Moritz, A. Sardá, T. Bartz-Beielstein, M. Zaefferer and J. Stork",
        "title": "Comparison of different Methods for Univariate Time Series Imputation in R",
        "journal": "arXiv preprint",
        "year": 2015,
        "vol": "arXiv:1510.03924",
        "pages": "",
        "doi": "10.48550/arXiv.1510.03924",
        "used_in": "Pipeline ③ — Tiered Imputation Strategy (Drop long gaps)",
        "context": "Cơ sở lý thuyết cho chiến lược 'Tiered Imputation': Các khoảng trống dữ liệu ngắn có thể nội suy an toàn, nhưng các khoảng trống dài (long gaps) làm mất đi động lực học cục bộ (local dynamics), việc cố gắng khôi phục chúng bằng nội suy đơn giản sẽ dẫn đến sai lệch (bias) lớn. Do đó, luận văn chọn phương án loại bỏ (drop) các gap > 24h.",
        "quote": "The performance of imputation algorithms is highly dependent on the length of the missing data gaps... In case of long gaps, most imputation algorithms fail to provide reasonable estimates.",
        "location": "Abstract & Conclusion",
        "pdf_link": "https://arxiv.org/pdf/1510.03924.pdf",
    },
    "box2015": {
        "id": 24,
        "authors": "G. E. P. Box, G. M. Jenkins, G. C. Reinsel, and G. M. Ljung",
        "title": "Time Series Analysis: Forecasting and Control",
        "journal": "John Wiley & Sons",
        "year": 2015,
        "vol": "5th Edition",
        "pages": "",
        "doi": "10.1002/9781118619193",
        "used_in": "Pipeline ⑥ — Modeling",
        "context": "Mô hình thống kê cơ sở: SARIMAX (Seasonal Auto-Regressive Integrated Moving Average with eXogenous factors).",
        "quote": "The Box-Jenkins methodology provides a systematic procedure for the identification, estimation, and diagnostic checking of ARIMA and SARIMA models for time series forecasting.",
        "location": "Chapters 3 & 9",
        "pdf_link": "docs/references/[48] Time Series Analysis_ Forecasting and Control.pdf",
    },
    "dietterich2000": {
        "id": 30,
        "authors": "T. G. Dietterich",
        "title": "Ensemble Methods in Machine Learning",
        "journal": "Multiple Classifier Systems",
        "year": 2000,
        "vol": "LNCS 1857",
        "pages": "1-15",
        "doi": "10.1007/3-540-45014-9_1",
        "used_in": "Pipeline ⑥ — Modeling",
        "context": "Cơ sở lý thuyết cho Voting Ensemble, chứng minh việc kết hợp nhiều mô hình (ensemble) giúp giảm variance và tăng độ chính xác tổng thể so với một mô hình đơn lẻ.",
        "quote": "Ensemble methods are learning algorithms that construct a set of classifiers and then classify new data points by taking a (weighted) vote of their predictions... ensembles are often much more accurate than the individual classifiers that make them up.",
        "location": "Abstract & Introduction",
        "pdf_link": "https://doi.org/10.1007/3-540-45014-9_1",
    },
    # ── NEW: Explainable AI (XAI) References ──
    "fisher2019": {
        "id": 39,
        "authors": "A. Fisher, C. Rudin, and F. Dominici",
        "title": "All Models are Wrong, but Many are Useful: Learning a Variable's Importance by Studying an Entire Class of Prediction Models Simultaneously",
        "journal": "Journal of Machine Learning Research",
        "year": 2019,
        "vol": "20(177)",
        "pages": "1-81",
        "doi": "",
        "used_in": "Explainability — Permutation Importance (GRU/LSTM)",
        "context": "Cơ sở lý thuyết cho Permutation Importance, một phương pháp model-agnostic hiệu quả để đánh giá độ quan trọng của đặc trưng đối với các mô hình Neural Networks phức tạp mà không cần truy cập vào cấu trúc nội bộ.",
        "quote": "We propose reliance as a measure of variable importance. Reliance is the change in the model's error when the variable is scrambled.",
        "location": "Abstract",
        "pdf_link": "https://jmlr.org/papers/volume20/18-760/18-760.pdf",
    },
    "gu2021": {
        "id": 44,
        "authors": "Y. Gu, B. Li, and Q. Meng",
        "title": "Hybrid interpretable predictive machine learning model for air pollution prediction",
        "journal": "Neurocomputing",
        "year": 2021,
        "vol": "466",
        "pages": "341-355",
        "doi": "10.1016/j.neucom.2021.09.051",
        "used_in": "Explainability — Applied SOTA Literature",
        "context": "Nghiên cứu ứng dụng thực tiễn chứng minh hiệu quả của việc kết hợp mô hình học máy (đặc biệt là các mô hình LightGBM/XGBoost) với công cụ SHAP để phân tích các yếu tố tác động đến dự báo ô nhiễm không khí.",
        "quote": "The integration of predictive machine learning with interpretability methods like SHAP facilitates understanding the contribution of different variables in air pollution forecasting.",
        "location": "Abstract",
        "pdf_link": "https://doi.org/10.1016/j.neucom.2021.09.051",
    },
    "houdou2024": {
        "id": 45,
        "authors": "A. Houdou, I. El Badisy, K. Khomsi ... M. Khalis",
        "title": "Interpretable Machine Learning Approaches for Forecasting and Predicting Air Pollution: A Systematic Review",
        "journal": "Aerosol and Air Quality Research",
        "year": 2024,
        "vol": "24",
        "pages": "230151",
        "doi": "10.4209/aaqr.230151",
        "used_in": "Explainability — Systematic Literature Review",
        "context": "Nghiên cứu tổng quan xác nhận SHAP đang là phương pháp được sử dụng nhiều nhất (chiếm 46.4%) trong lĩnh vực giải thích các mô hình dự báo ô nhiễm không khí.",
        "quote": "SHAP has emerged as the most widely used interpretation technique, effectively bridging the gap between high-performance modeling and policy-oriented transparency.",
        "location": "Abstract",
        "pdf_link": "https://aaqr.org/articles/aaqr-23-06-oa-0151.pdf",
    },
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Pipeline Step Framework — Circled Numbers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Unicode circled numbers: visually distinct from IEEE [number] citations.
_CIRCLED = {
    1: "①",
    2: "②",
    3: "③",
    4: "④",
    5: "⑤",
    6: "⑥",
    7: "⑦",
    8: "⑧",
    9: "⑨",
    10: "⑩",
    11: "⑪",
    12: "⑫",
    13: "⑬",
    14: "⑭",
    15: "⑮",
    16: "⑯",
    17: "⑰",
    18: "⑱",
    19: "⑲",
    20: "⑳",
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
        f"background:rgba(255,149,0,0.25); "
        f"color:#FF9500; "
        f"padding:0 0.35rem; "
        f"border-radius:4px; "
        f"font-size:0.85rem; "
        f"font-weight:700; "
        f"margin:0 2px; "
        f"font-family:'Inter',sans-serif;\""
        f">{icon}</span>"
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

.cite-content .cite-title,
.cite-content .cite-meta,
.cite-content .cite-used,
.cite-content .cite-context,
.cite-content .cite-quote,
.cite-content .cite-location {
    display: block;
}

.cite-content .cite-quote {
    color: #F87171;
    font-size: 0.75rem;
    font-style: italic;
    margin-top: 0.4rem;
    border-left: 2px solid #F87171;
    padding-left: 0.4rem;
}

.cite-content .cite-location {
    color: #A78BFA;
    font-size: 0.7rem;
    margin-top: 0.2rem;
    font-weight: 600;
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
    doi_info = f"<br>DOI: {ref['doi']}" if ref["doi"] else ""
    used = ref.get("used_in", "")
    context = ref.get("context", "")
    quote = ref.get("quote", "")
    location = ref.get("location", "")

    # Use span instead of div to avoid breaking Markdown parser when nested in **...**
    context_html = f'<span class="cite-context">{context}</span>' if context else ""
    quote_html = f'<span class="cite-quote">"{quote}"</span>' if quote else ""
    location_html = f'<span class="cite-location">📍 {location}</span>' if location else ""

    if ref["doi"]:
        link_html = (
            f'<a href="https://doi.org/{ref["doi"]}" target="_blank" class="cite-link">🔗 Đọc tài liệu (DOI) ↗</a>'
        )
    else:
        # Fallback to Google Scholar search by title if no DOI exists
        search_query = urllib.parse.quote_plus(ref["title"])
        link_html = f'<a href="https://scholar.google.com/scholar?q={search_query}" target="_blank" class="cite-link" title="Tìm trên Google Scholar">🔍 Tìm sách/tài liệu: {ref["title"]} ↗</a>'

    pdf_link_html = (
        f'<a href="{ref["pdf_link"]}" target="_blank" class="cite-link pdf-link" style="color: #FF6B6B; margin-left: 10px;">📄 Tải PDF ↗</a>'
        if ref.get("pdf_link")
        else ""
    )

    tooltip = (
        f'<span class="cite-content">'
        f'<span class="cite-title">{ref["title"]}</span>'
        f'<span class="cite-meta">{ref["authors"]}, '
        f"<em>{ref['journal']}</em>{vol_info}{page_info}, {ref['year']}."
        f"{doi_info}</span>"
        f'<span class="cite-used">📌 Áp dụng: {used}</span>'
        f"{context_html}"
        f"{quote_html}"
        f"{location_html}"
        f'<span style="display: block; margin-top: 0.5rem;">{link_html}{pdf_link_html}</span>'
        f"</span>"
    )

    return f'<span class="cite-tooltip">[{ref["id"]}]{tooltip}</span>'


def render_references_section(title: str = "📚 Tài Liệu Tham Khảo (IEEE)", filter_ids: list = None):
    """Render a full IEEE-formatted references list at the bottom of a page."""
    _ensure_css()

    refs = IEEE_REFS.values()
    if filter_ids is not None:
        refs = [r for r in refs if r["id"] in filter_ids]

    sorted_refs = sorted(refs, key=lambda r: r["id"])

    rows = []
    for ref in sorted_refs:
        vol_info = f", vol. {ref['vol']}" if ref["vol"] else ""
        page_info = f", pp. {ref['pages']}" if ref["pages"] else ""

        if ref["doi"]:
            doi_link = f' DOI: <a href="https://doi.org/{ref["doi"]}" target="_blank" style="color:#60A5FA; text-decoration:none;">{ref["doi"]}</a>.'
        else:
            search_query = urllib.parse.quote_plus(ref["title"])
            doi_link = f' [<a href="https://scholar.google.com/scholar?q={search_query}" target="_blank" style="color:#60A5FA; text-decoration:none;">Scholar</a>]'

        # Format detection: if it doesn't have vol and pages, treat as a book
        is_book = not ref["vol"] and not ref["pages"]

        if is_book:
            # IEEE Book format: Author, Title (italic), Edition. Place: Publisher, Year.
            # We map "journal" to publisher here
            title_str = f"<em>{ref['title']}</em>."
            journal_str = f"{ref['journal']}"
        else:
            # IEEE Article format: Author, "Title," Journal (italic), vol., pp., Year.
            title_str = f'"{ref["title"]},"'
            journal_str = f"<em>{ref['journal']}</em>"

        rows.append(
            f'<div style="margin-bottom:0.75rem; font-size:0.9rem; line-height:1.5; padding-left: 2.2rem; text-indent: -2.2rem;">'
            f'<span style="color:#00D4AA; font-weight:700; display:inline-block; width: 2rem;">[{ref["id"]}]</span>'
            f"{ref['authors']}, "
            f"{title_str} "
            f"{journal_str}{vol_info}{page_info}, {ref['year']}."
            f"{doi_link}"
            f"</div>"
        )

    if title == "VERIFIED_CARD_INTL":
        st.markdown(
            '<div style="background: rgba(0,212,170,0.03); border-radius: 12px; padding: 1.5rem; margin-top: 1rem; border: 1px solid rgba(0,212,170,0.15); box-shadow: inset 0 0 20px rgba(0,0,0,0.2);">'
            '<div style="display: flex; align-items: center; margin-bottom: 1.2rem;">'
            '<span style="background: #00D4AA; color: #0E1117; padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 0.8rem; letter-spacing: 1px; margin-right: 10px;">VERIFIED</span>'
            '<b style="color: #00D4AA; font-size: 1.1rem; letter-spacing: 0.5px;">📎 Danh Mục Nguồn Tham Khảo & DOI (Quốc tế)</b>'
            "</div>"
            '<div style="display: flex; flex-direction: column; gap: 0.8rem;">' + "\n".join(rows) + "</div></div>",
            unsafe_allow_html=True,
        )
    elif title == "VERIFIED_CARD_VN":
        st.markdown(
            '<div style="background: rgba(0,212,170,0.03); border-radius: 12px; padding: 1.5rem; margin-top: 1rem; border: 1px solid rgba(0,212,170,0.15); box-shadow: inset 0 0 20px rgba(0,0,0,0.2);">'
            '<div style="display: flex; align-items: center; margin-bottom: 1.2rem;">'
            '<span style="background: #00D4AA; color: #0E1117; padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 0.8rem; letter-spacing: 1px; margin-right: 10px;">VERIFIED</span>'
            '<b style="color: #00D4AA; font-size: 1.1rem; letter-spacing: 0.5px;">📎 Danh Mục Nguồn Tham Khảo & DOI (Việt Nam)</b>'
            "</div>"
            '<div style="display: flex; flex-direction: column; gap: 0.8rem;">' + "\n".join(rows) + "</div></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="margin-top:2rem; padding-top:1rem; '
            'border-top:2px solid rgba(0,212,170,0.2);">'
            f'<h3 style="color:#00D4AA; font-size:1.1rem; margin-bottom: 1rem;">{title}</h3>'
            + "\n".join(rows)
            + "</div>",
            unsafe_allow_html=True,
        )
