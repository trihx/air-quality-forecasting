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
    # ── [1] WHO (2021) ──
    "who2021": {
        "id": 1,
        "authors": "World Health Organization",
        "title": "WHO Global Air Quality Guidelines: Particulate Matter (PM2.5 and PM10), Ozone, Nitrogen Dioxide, Sulfur Dioxide and Carbon Monoxide",
        "journal": "Geneva, Switzerland: World Health Organization",
        "year": 2021,
        "vol": "",
        "pages": "",
        "doi": "",
        "used_in": "Domain Bounds & §1.2 — Ngưỡng an toàn nồng độ PM2.5 WHO (15 µg/m³ 24h, 5 µg/m³ năm)",
        "context": "Khung hướng dẫn chất lượng không khí toàn cầu của WHO thiết lập ngưỡng khuyến nghị 24 giờ cho PM2.5 là 15 µg/m³ và trung bình năm là 5 µg/m³. Đề án sử dụng ngưỡng này làm mốc mỏ neo để phân tích rủi ro sức khỏe cộng đồng và đánh giá năng lực cảnh báo sớm của mô hình tại Sa Đéc.",
        "quote": "Air pollution is one of the greatest environmental risks to health. By reducing air pollution levels, countries can reduce the burden of disease.",
        "location": "Executive Summary & Recommendations",
        "pdf_link": "docs/references/[28]WHO 2021_AQI Guidelines.pdf",
    },
    # ── [2] Zannetti (1990) ──
    "zannetti1990": {
        "id": 2,
        "authors": "P. Zannetti",
        "title": "Air Pollution Modeling: Theories, Computational Methods and Available Software",
        "journal": "Southampton, UK: Computational Mechanics Publications",
        "year": 1990,
        "vol": "",
        "pages": "",
        "doi": "",
        "used_in": "EDA & §2.1 — Cơ sở động học khí quyển và cơ chế khuếch tán ô nhiễm",
        "context": "Tác phẩm kinh điển về mô hình hóa ô nhiễm không khí, cung cấp nền tảng lý thuyết về quá trình phát thải, lan truyền, lắng đọng và tương tác giữa các chất ô nhiễm với biến số khí tượng (nhiệt độ, độ ẩm, áp suất).",
        "quote": "Air pollution modeling is the mathematical simulation of how air pollutants disperse and react in the atmosphere.",
        "location": "Chapter 1: Atmospheric Dynamics",
        "pdf_link": "docs/references/[31] P. Zannetti_Air Pollution Modeling.pdf",
    },
    # ── [3] Barkjohn et al. (2021) ──
    "barkjohn2021": {
        "id": 3,
        "authors": "R. L. Barkjohn, B. Gantt, and A. L. Clements",
        "title": "Development and application of a national correction equation for PurpleAir PM2.5 sensors",
        "journal": "Atmospheric Measurement Techniques",
        "year": 2021,
        "vol": "14(6)",
        "pages": "4617–4637",
        "doi": "10.5194/amt-14-4617-2021",
        "used_in": "Pipeline ② — Hiệu chỉnh cảm biến chi phí thấp & Resampling đa phân giải",
        "context": "Nghiên cứu của Cơ quan Bảo vệ Môi trường Hoa Kỳ (US EPA) chuẩn hóa phương trình hiệu chỉnh cho cảm biến quang học chi phí thấp (như PurpleAir/Plantower). Nghiên cứu chứng minh độ ẩm không khí cao (RH) gây hiệu ứng hút ẩm trương nở hạt bụi, dẫn đến đo nồng độ ảo, định hình cơ chế tiền xử lý và kỹ nghệ đặc trưng tương tác trong đề án.",
        "quote": "The nationwide correction reduced the root mean square error (RMSE) of PM2.5 measurements from low-cost sensors by over 30% across diverse geographical regions.",
        "location": "Section 3.2: Correction Equation",
        "pdf_link": "docs/references/[29] Barkjohn 2021_PurpleAir correction.pdf",
    },
    # ── [4] Moritz et al. (2015) ──
    "moritz2015": {
        "id": 4,
        "authors": "S. Moritz, A. Sardá, N. Bartz-Beielstein, M. Zaefferer, and J. Stork",
        "title": "Comparison of different methods for univariate time series imputation in R",
        "journal": "arXiv preprint arXiv:1510.03924",
        "year": 2015,
        "vol": "",
        "pages": "",
        "doi": "10.48550/arXiv.1510.03924",
        "used_in": "Pipeline ③ — Chiến lược điền khuyết phân tầng (Gap ngắn vs Gap dài)",
        "context": "Khảo sát và so sánh thực nghiệm toàn diện các thuật toán điền khuyết (imputation) trên chuỗi thời gian. Đề tài kế thừa nguyên tắc: sử dụng nội suy Spline/Linear cho các khoảng gián đoạn ngắn (≤ 6 bước) và duy trì trạng thái quá khứ cho khoảng gián đoạn dài để tránh tạo dữ liệu giả lập méo mó.",
        "quote": "Interpolation methods, specifically linear and spline interpolation, perform consistently well for short gaps with high temporal autocorrelation.",
        "location": "Section 4: Empirical Results",
        "pdf_link": "docs/references/[47] Moritz, S., et al. (2015 Comparison of different Methods for Univariate Time Series Imputation in R.pdf",
    },
    # ── [5] Tashman (2000) ──
    "tashman2000": {
        "id": 5,
        "authors": "L. J. Tashman",
        "title": "Out-of-sample tests of forecasting accuracy: An analysis and review",
        "journal": "International Journal of Forecasting",
        "year": 2000,
        "vol": "16(4)",
        "pages": "437–450",
        "doi": "10.1016/S0169-2070(00)00065-0",
        "used_in": "Pipeline ⑤ & §3.4 — Phân chia dữ liệu theo dòng thời gian (Temporal Split) & Chính sách Test-on-Real-Only",
        "context": "Nền tảng phương pháp luận cho kiểm định dự báo ngoài mẫu. Đề tài tuân thủ nghiêm ngặt nguyên tắc của Tashman: Phân tách tập dữ liệu Train/Test theo thời gian tuyệt đối không shuffle; tập Test chỉ đánh giá trên dữ liệu thực đo (is_imputed = 0) để đảm bảo tính khách quan tuyệt đối.",
        "quote": "Out-of-sample tests are essential because in-sample fit provides an overoptimistic assessment of forecast accuracy.",
        "location": "Section 2: Principles of Out-of-sample Testing",
        "pdf_link": "docs/references/[15] Tashman 2000_Out-of-sample Testing.pdf",
    },
    # ── [6] Hyndman & Koehler (2006) ──
    "hyndman2006": {
        "id": 6,
        "authors": "R. J. Hyndman and A. B. Koehler",
        "title": "Another look at measures of forecast accuracy",
        "journal": "International Journal of Forecasting",
        "year": 2006,
        "vol": "22(4)",
        "pages": "679–688",
        "doi": "10.1016/j.ijforecast.2006.03.001",
        "used_in": "Evaluation & §3.5 — Định nghĩa thước đo chuẩn hóa MASE & phá vỡ bẫy quán tính",
        "context": "Công trình kinh điển đề xuất chỉ số MASE (Mean Absolute Scaled Error). Bằng cách chuẩn hóa sai số MAE của mô hình so với sai số của mô hình cơ sở Persistence trên tập huấn luyện, MASE khắc phục triệt để lỗi chia cho 0 của MAPE và tính nhạy cảm thái quá của R², cho phép so sánh công bằng giữa các mô hình và cấp phân giải.",
        "quote": "MASE is recommended as the default measure of forecast accuracy because it is scale-free, less sensitive to outliers, and never gives infinite or undefined values.",
        "location": "Section 3.2: Scaled Errors",
        "pdf_link": "docs/references/[01]_Hyndman 2006.pdf",
    },
    # ── [7] Zhang (2011) ──
    "zhang2011": {
        "id": 7,
        "authors": "Z. Zhang",
        "title": "Multivariate Time Series Analysis in Climate and Environmental Research",
        "journal": "Dordrecht, Netherlands: Springer",
        "year": 2011,
        "vol": "",
        "pages": "",
        "doi": "10.1007/978-3-642-18231-0",
        "used_in": "EDA & §4.1 — Phân tích tương quan đa biến (Pearson, Spearman, Mutual Information)",
        "context": "Cung cấp nền tảng toán học cho phân tích chuỗi thời gian đa biến trong khoa học khí hậu và môi trường. Đề án áp dụng để phân tích quan hệ phi tuyến và độ trễ pha giữa nồng độ PM2.5 với các yếu tố vi khí hậu (nhiệt độ, độ ẩm, điểm sương, CO2).",
        "quote": "Environmental time series exhibit complex multi-scale interactions where non-linear dependence structures must be assessed alongside linear correlations.",
        "location": "Chapter 3: Correlation and Dependence Analysis",
        "pdf_link": "docs/references/[30] Z. Zhang_ Multivariate Time Series Analysis in Climate and Environmental Research (Zhihua Zhang (auth.)).pdf",
    },
    # ── [8] Blanchard & Tanenbaum (2003) ──
    "blanchard2003": {
        "id": 8,
        "authors": "C. L. Blanchard and S. J. Tanenbaum",
        "title": "Differences between weekday and weekend air pollutant levels in Southern California",
        "journal": "Journal of the Air & Waste Management Association",
        "year": 2003,
        "vol": "53(7)",
        "pages": "816–828",
        "doi": "10.1080/10473289.2003.10466222",
        "used_in": "Feature Engineering & §4.2 — Kỹ nghệ đặc trưng lịch (Weekend Effect & Rush Hour)",
        "context": "Chứng minh hiệu ứng chênh lệch mức độ ô nhiễm giữa ngày làm việc trong tuần và ngày cuối tuần (Weekend Effect) do hoạt động giao thông và sản xuất công nghiệp thay đổi theo chu kỳ xã hội, là cơ sở khoa học để thiết kế nhóm đặc trưng Calendar Features (is_weekend, is_rush_hour, sin/cos cyclical).",
        "quote": "The weekend effect provides a natural experiment to study the atmospheric response to substantial reductions in precursor emissions.",
        "location": "Abstract & Discussion",
        "pdf_link": "docs/references/[32] C. L. Blanchard and S. Tanenbaum (2003)_Differences between Weekday and Weekend Air Pollutant Levels in Southern California.pdf",
    },
    # ── [9] Shumway & Stoffer (2017) ──
    "shumway2017": {
        "id": 9,
        "authors": "R. H. Shumway and D. S. Stoffer",
        "title": "Time Series Analysis and Its Applications: With R Examples",
        "journal": "4th ed. Cham, Switzerland: Springer",
        "year": 2017,
        "vol": "",
        "pages": "",
        "doi": "10.1007/978-3-319-52452-8",
        "used_in": "EDA & Modeling §3.2 — Nền tảng tự tương quan ACF/PACF & Mô hình ARIMA",
        "context": "Giáo trình chuẩn mực về lý thuyết chuỗi thời gian. Đề án sử dụng để phân tích hàm tự tương quan mẫu ACF/PACF nhằm xác định các chu kỳ trễ quan trọng (1h, 24h, 168h) và thiết lập mô hình thống kê chuẩn tắc ARIMA(p,d,q).",
        "quote": "Autocorrelation functions provide a fundamental measure of the temporal dependence between observations at different time horizons.",
        "location": "Chapter 1: Characteristics of Time Series & Chapter 3: ARIMA Models",
        "pdf_link": "docs/references/[10] Time Series Analysis and Its Applications With R Examples (Robert H. Shumway, David S. Stoffer (auth.)) (Z-Library).pdf",
    },
    # ── [10] Dickey & Fuller (1979) ──
    "dickey1979": {
        "id": 10,
        "authors": "D. A. Dickey and W. A. Fuller",
        "title": "Distribution of the estimators for autoregressive time series with a unit root",
        "journal": "Journal of the American Statistical Association",
        "year": 1979,
        "vol": "74(366a)",
        "pages": "427–431",
        "doi": "10.1080/01621459.1979.10482531",
        "used_in": "EDA & §4.1.1 — Kiểm định nghiệm đơn vị Augmented Dickey-Fuller (ADF Test)",
        "context": "Công trình nền tảng của kiểm định nghiệm đơn vị ADF nhằm kiểm tra tính dừng (stationarity) của chuỗi thời gian. Kết quả ADF trên chuỗi PM2.5 tại Sa Đéc bác bỏ giả thuyết H0 (p < 0.001), khẳng định chuỗi dừng ở mức ý nghĩa 1%.",
        "quote": "The distribution of the least-squares estimator of the autoregressive parameter for a unit root process is tabulated and applied to test for stationarity.",
        "location": "Section 1: Introduction and Theory",
        "pdf_link": "docs/references/[23] Dickey 1979_ADF Test.pdf",
    },
    # ── [11] Kwiatkowski et al. (1992) ──
    "kwiatkowski1992": {
        "id": 11,
        "authors": "D. Kwiatkowski, P. C. B. Phillips, P. Schmidt, and Y. Shin",
        "title": "Testing the null hypothesis of stationarity against the alternative of a unit root",
        "journal": "Journal of Econometrics",
        "year": 1992,
        "vol": "54(1–3)",
        "pages": "159–178",
        "doi": "10.1016/0304-4076(92)90104-Y",
        "used_in": "EDA & §4.1.1 — Kiểm định tính dừng KPSS (đối chứng với ADF)",
        "context": "Kiểm định KPSS với giả thuyết H0 là chuỗi dừng (ngược với ADF). Kết hợp ADF và KPSS giúp kết luận chuỗi PM2.5 dừng xu hướng (trend-stationary) nhưng xuất hiện phương sai cục bộ do ngoại lệ môi trường.",
        "quote": "The KPSS test provides a complementary stationarity test by specifying the null hypothesis as stationarity around a deterministic trend.",
        "location": "Abstract & Section 2",
        "pdf_link": "docs/references/[24] Kwiatkowski 1992_KPSS Test.pdf",
    },
    # ── [12] Peixeiro (2022) ──
    "peixeiro2022": {
        "id": 12,
        "authors": "M. Peixeiro",
        "title": "Time Series Forecasting in Python",
        "journal": "Shelter Island, NY, USA: Manning Publications",
        "year": 2022,
        "vol": "",
        "pages": "",
        "doi": "",
        "used_in": "Pipeline & §3.2 — Quy trình benchmark phân tầng & Kiểm định nhân quả Granger",
        "context": "Cẩm nang hướng dẫn thực hành xây dựng hệ thống dự báo chuỗi thời gian phân tầng: Baseline → Thống kê → Học máy → Học sâu. Đề tài kế thừa quy trình kiểm tra chất lượng phần dư và phân tích nhân quả Granger giữa các biến khí tượng và PM2.5.",
        "quote": "A principled forecasting workflow begins with a naive baseline to establish minimum acceptable performance before deploying complex architectures.",
        "location": "Part 2: Statistical Models & Part 4: Advanced Forecasting",
        "pdf_link": "docs/references/[09] Peixeiro (2022)Time Series Forecasting in Python (Marco Peixeiro) (Z-Library).pdf",
    },
    # ── [13] Ljung & Box (1978) ──
    "ljung1978": {
        "id": 13,
        "authors": "G. M. Ljung and G. E. P. Box",
        "title": "On a measure of lack of fit in time series models",
        "journal": "Biometrika",
        "year": 1978,
        "vol": "65(2)",
        "pages": "297–303",
        "doi": "10.1093/biomet/65.2.297",
        "used_in": "Evaluation & §4.5 — Chẩn đoán phần dư (Residual Diagnostics) & Ljung-Box Test",
        "context": "Kiểm định Ljung-Box Q-statistic đánh giá tính độc lập của chuỗi phần dư (White Noise). Trong đề án, việc Ljung-Box bác bỏ H0 ở mốc ngắn phản ánh tính phi tuyến còn sót lại, tạo cơ sở khoa học để áp dụng Adaptive Conformal Inference (ACI) hiệu chuẩn khoảng tin cậy.",
        "quote": "The proposed statistic provides a sensitive test for detecting autocorrelation among the residuals of an estimated time series model.",
        "location": "Section 2: The Modified Statistic",
        "pdf_link": "docs/references/[25] Ljung 1978_Ljung-Box Test.pdf",
    },
    # ── [14] Makridakis et al. (2020) ──
    "makridakis2020": {
        "id": 14,
        "authors": "S. Makridakis, E. Spiliotis, and V. Assimakopoulos",
        "title": "The M4 Competition: 100,000 time series and 61 forecasting methods",
        "journal": "International Journal of Forecasting",
        "year": 2020,
        "vol": "36(1)",
        "pages": "54–74",
        "doi": "10.1016/j.ijforecast.2019.04.014",
        "used_in": "Evaluation & §2.2 — Bài học M4 Competition & Chuẩn mực đánh giá mô hình lai",
        "context": "Báo cáo tổng kết cuộc thi dự báo toàn cầu M4 với 100.000 chuỗi thời gian. Kết luận cốt lõi: Các mô hình lai (Hybrid) kết hợp mô hình thống kê và học máy/học sâu đạt độ chính xác vượt trội so với các mô hình đơn lẻ thuần túy, định hình giải pháp Stacking/Weighted Ensemble trong đề án.",
        "quote": "The major finding of the M4 Competition was that hybrid and combination approaches outperformed all individual statistical and machine learning methods.",
        "location": "Section 6: Key Findings",
        "pdf_link": "docs/references/[17] Makridakis 2020_M4 Competition.pdf",
    },
    # ── [15] Box et al. (2015) ──
    "box2015": {
        "id": 15,
        "authors": "G. E. P. Box, G. M. Jenkins, G. C. Reinsel, and G. M. Ljung",
        "title": "Time Series Analysis: Forecasting and Control",
        "journal": "5th ed. Hoboken, NJ, USA: John Wiley & Sons",
        "year": 2015,
        "vol": "",
        "pages": "",
        "doi": "10.1002/9781118619193",
        "used_in": "Modeling & §3.2 — Phương pháp luận Box-Jenkins & Mô hình SARIMAX",
        "context": "Tác phẩm nền tảng của phương pháp Box-Jenkins trong xây dựng mô hình ARIMA/SARIMAX: Nhận dạng bậc (p,d,q) qua ACF/PACF, ước lượng tham số hợp lý cực đại, và kiểm định chẩn đoán phần dư để làm mô hình đối chứng thống kê.",
        "quote": "The iterative cycle of model identification, parameter estimation, and diagnostic checking forms the foundation of reliable time series modeling.",
        "location": "Chapter 3: Linear Stationary Models & Chapter 9: Seasonal Models",
        "pdf_link": "docs/references/[48] Time Series Analysis_ Forecasting and Control.pdf",
    },
    # ── [16] Joseph (2022) ──
    "joseph2022": {
        "id": 16,
        "authors": "M. Joseph",
        "title": "Modern Time Series Forecasting with Python",
        "journal": "Birmingham, UK: Packt Publishing",
        "year": 2022,
        "vol": "",
        "pages": "",
        "doi": "",
        "used_in": "EDA & Feature Engineering §4.1 — Đo lường độ khả dự (Forecastability) & Kỹ nghệ biến trễ",
        "context": "Tài liệu chuyên sâu về dự báo hiện đại bằng Python: Đo lường độ phức tạp chuỗi thời gian, tính entropy phổ và phân tích tương quan đa bước trễ (Lag features), định hình cấu trúc 119 đặc trưng phi rò rỉ của đề án.",
        "quote": "Measuring forecastability before selecting model architectures prevents over-engineering and highlights data bottlenecks.",
        "location": "Chapter 3: Exploring and Understanding Time Series",
        "pdf_link": "docs/references/[34] Manu Joseph - Modern Time Series Forecasting with Python_ Explore industry-ready time series forecasting using modern machine learning and dee (2022, Packt Publishing) - libgen.li.pdf",
    },
    # ── [17] Ke et al. (2017) ──
    "ke2017": {
        "id": 17,
        "authors": "G. Ke et al.",
        "title": "LightGBM: A highly efficient gradient boosting decision tree",
        "journal": "Advances in Neural Information Processing Systems (NeurIPS)",
        "year": 2017,
        "vol": "30",
        "pages": "3146–3154",
        "doi": "",
        "used_in": "Modeling & §3.3 — Mô hình LightGBM (GOSS & EFB) dự báo nồng độ bụi mịn",
        "context": "Thuật toán cây quyết định tăng cường độ dốc (GBDT) tối ưu hóa với cơ chế GOSS (Gradient-based One-Side Sampling) và EFB (Exclusive Feature Bundling). LightGBM là mô hình Machine Learning chủ lực trong đề án, đạt tốc độ huấn luyện nhanh gấp 10 lần và xử lý hiệu quả 119 đặc trưng bảng.",
        "quote": "LightGBM speeds up the training process of conventional GBDT by up to over 20 times while achieving almost the same accuracy.",
        "location": "Section 1: Introduction & Section 2: Preliminaries",
        "pdf_link": "docs/references/[06] Ke 2017_LightGBM.pdf",
    },
    # ── [18] Breiman (2001) ──
    "breiman2001": {
        "id": 18,
        "authors": "L. Breiman",
        "title": "Random forests",
        "journal": "Machine Learning",
        "year": 2001,
        "vol": "45(1)",
        "pages": "5–32",
        "doi": "10.1023/A:1010933404324",
        "used_in": "Modeling & §3.3 — Mô hình Random Forest phi tuyến & Cơ chế Bagging",
        "context": "Thuật toán rừng cây ngẫu nhiên kết hợp cơ chế lấy mẫu có hoàn lại (Bootstrap Aggregating - Bagging) và chọn ngẫu nhiên không gian đặc trưng. Mô hình được dùng làm benchmark phi tuyến mạnh mẽ, kiểm chứng năng lực kháng overfit khi so sánh với LightGBM và mạng nơ-ron sâu.",
        "quote": "Random forests are a combination of tree predictors such that each tree depends on the values of a random vector sampled independently.",
        "location": "Section 1: Introduction",
        "pdf_link": "docs/references/[27] Breiman 2001_Random Forests.pdf",
    },
    # ── [19] Huang & Petukhina (2022) ──
    "huang2022": {
        "id": 19,
        "authors": "C. Huang and A. Petukhina",
        "title": "Applied Time Series Analysis and Forecasting with Python",
        "journal": "Cham, Switzerland: Springer",
        "year": 2022,
        "vol": "",
        "pages": "",
        "doi": "10.1007/978-3-031-13971-0",
        "used_in": "Modeling & Evaluation — Thiết kế hàm mất mát và phân tích sai số theo thời gian",
        "context": "Giáo trình ứng dụng toán học và Python trong phân tích chuỗi thời gian: Cung cấp kỹ thuật chuẩn hóa dữ liệu, thiết lập ma trận bước trễ (Lag matrix) và xây dựng pipeline dự báo đa bước thời gian (Multi-horizon direct forecasting).",
        "quote": "Direct multi-step forecasting models avoid error accumulation inherent in recursive strategies.",
        "location": "Chapter 6: Machine Learning for Forecasting",
        "pdf_link": "docs/references/[35] Applied Time Series Analysis and Forecasting with Python (Changquan Huang, Alla Petukhina) (Z-Library).pdf",
    },
    # ── [20] Hochreiter & Schmidhuber (1997) ──
    "hochreiter1997": {
        "id": 20,
        "authors": "S. Hochreiter and J. Schmidhuber",
        "title": "Long short-term memory",
        "journal": "Neural Computation",
        "year": 1997,
        "vol": "9(8)",
        "pages": "1735–1780",
        "doi": "10.1162/neco.1997.9.8.1735",
        "used_in": "Modeling & §3.3 — Mạng nơ-ron hồi quy sâu LSTM (Long Short-Term Memory)",
        "context": "Kiến trúc mạng hồi quy sâu giải quyết triệt để vấn đề triệt tiêu đạo hàm (vanishing gradient) qua cấu trúc bộ nhớ tế bào (Memory Cell) và 3 cổng kiểm soát (Input, Forget, Output Gate). LSTM được triển khai để học các phụ thuộc thời gian dài trong chuỗi PM2.5 ở các mốc 6h và 24h.",
        "quote": "LSTM can solve complex long-time-lag tasks that have never been solved by previous recurrent network algorithms.",
        "location": "Section 1: Introduction & Section 2: Architecture",
        "pdf_link": "docs/references/[07] Hochreiter 1997_LSTM Architecture.pdf",
    },
    # ── [21] Cho et al. (2014) ──
    "cho2014": {
        "id": 21,
        "authors": "K. Cho et al.",
        "title": "Learning phrase representations using RNN encoder-decoder for statistical machine translation",
        "journal": "Proc. Conference on Empirical Methods in Natural Language Processing (EMNLP)",
        "year": 2014,
        "vol": "",
        "pages": "1724–1734",
        "doi": "10.3115/v1/D14-1179",
        "used_in": "Modeling & §3.3 — Mô hình GRU (Gated Recurrent Unit) - Mô hình DL tốt nhất ở phân giải 15m",
        "context": "Kiến trúc GRU tinh giản từ LSTM, gộp cổng Forget và Input thành cổng Update Gate duy nhất. Với số lượng tham số ít hơn 25%, GRU huấn luyện nhanh hơn, ít bị overfit trên dữ liệu vừa và nhỏ, và đạt MASE = 0,667 ở phân giải 15m (mô hình DL duy nhất vượt qua baseline Persistence ở mốc 1h).",
        "quote": "The gated recurrent unit acts similarly to the LSTM unit, but is computationally more efficient while achieving comparable performance.",
        "location": "Section 2.1: Gated Recurrent Unit",
        "pdf_link": "docs/references/[05] Cho 2014_GRU Architecture.pdf",
    },
    # ── [22] Lim et al. (2021) ──
    "lim2021": {
        "id": 22,
        "authors": "B. Lim, S. O. Arik, N. Loeff, and T. Pfister",
        "title": "Temporal Fusion Transformers for interpretable multi-horizon time series forecasting",
        "journal": "International Journal of Forecasting",
        "year": 2021,
        "vol": "37(4)",
        "pages": "1748–1764",
        "doi": "10.1016/j.ijforecast.2021.03.012",
        "used_in": "Modeling & §3.3 — Mô hình Temporal Fusion Transformer (TFT) với Self-Attention",
        "context": "Kiến trúc Transformer chuyên biệt cho chuỗi thời gian kết hợp cơ chế Self-Attention đa đầu và mạng Variable Selection Network (VSN). TFT cho phép dự báo đồng thời nhiều chân trời (1h, 6h, 24h) và trực quan hóa trọng số chú ý thời gian.",
        "quote": "Temporal Fusion Transformers combine high-performance multi-horizon forecasting with interpretable insights into temporal dynamics.",
        "location": "Section 3: Temporal Fusion Transformer Architecture",
        "pdf_link": "docs/references/[08] Lim 2021_TFT Architecture.pdf",
    },
    # ── [23] Wolpert (1992) ──
    "wolpert1992": {
        "id": 23,
        "authors": "D. H. Wolpert",
        "title": "Stacked generalization",
        "journal": "Neural Networks",
        "year": 1992,
        "vol": "5(2)",
        "pages": "241–259",
        "doi": "10.1016/S0893-6080(05)80023-1",
        "used_in": "Modeling & §3.3 — Kỹ thuật xếp chồng mô hình Stacking Ensemble",
        "context": "Cơ sở lý thuyết của phương pháp Stacking: Huấn luyện một mô hình học cấp cao (Meta-learner) để học cách kết hợp tối ưu các dự báo từ các mô hình cơ sở cấp thấp (Base learners), giảm thiểu bias và phương sai của từng mô hình thành phần.",
        "quote": "Stacked generalization works by deducing the biases of the generalizer(s) with respect to a provided learning set.",
        "location": "Section 1: Introduction",
        "pdf_link": "docs/references/[26] Wolpert 1992_Stacked Generalization.pdf",
    },
    # ── [24] Dietterich (2000) ──
    "dietterich2000": {
        "id": 24,
        "authors": "T. G. Dietterich",
        "title": "Ensemble methods in machine learning",
        "journal": "Multiple Classifier Systems, J. Kittler and F. Roli, Eds. Berlin, Germany: Springer",
        "year": 2000,
        "vol": "LNCS 1857",
        "pages": "1–15",
        "doi": "10.1007/3-540-45014-9_1",
        "used_in": "Modeling & §3.3 — Lý thuyết kết hợp mô hình học máy (Ensemble Diversity)",
        "context": "Giải thích 3 lý do nền tảng giúp phương pháp kết hợp mô hình (Ensemble) vượt trội hơn mô hình đơn: Lý do thống kê (Statistical), lý do tính toán (Computational), và lý do biểu diễn (Representational). Đây là cơ sở khoa học để đề án xây dựng Ensemble Weighted và Voting kết hợp ML và DL.",
        "quote": "A committee of learning machines can often achieve higher accuracy than any of the individual members.",
        "location": "Section 2: Why Ensembles Work",
        "pdf_link": "docs/references/[49] Dietterich 2000_Ensemble Methods.pdf",
    },
    # ── [25] Zhang & Li (2022) ──
    "zhang2022": {
        "id": 25,
        "authors": "J. Zhang and S. Li",
        "title": "Air quality index forecast in Beijing based on CNN-LSTM multi-model",
        "journal": "Chemosphere",
        "year": 2022,
        "vol": "308",
        "pages": "136180",
        "doi": "10.1016/j.chemosphere.2022.136180",
        "used_in": "Literature Review §2.3 & Đối chuẩn benchmark mô hình Deep Learning",
        "context": "Nghiên cứu ứng dụng kiến trúc kết hợp CNN trích xuất đặc trưng không gian cục bộ và LSTM nắm bắt phụ thuộc thời gian dài trong dự báo chỉ số AQI tại Bắc Kinh (MAE = 8,12 µg/m³). Đề án kế thừa mô hình lai nhưng bổ sung cơ chế xử lý chuỗi thực tế bị gián đoạn tín hiệu tại Sa Đéc.",
        "quote": "The proposed CNN-LSTM hybrid model exhibits superior forecasting performance compared to single deep learning architectures for air quality index.",
        "location": "Abstract & Section 3",
        "pdf_link": "https://doi.org/10.1016/j.chemosphere.2022.136180",
    },
    # ── [26] Patel et al. (2025) ──
    "patel2025": {
        "id": 26,
        "authors": "P. Patel, S. Bhatt, and B. Shah",
        "title": "A systematic study on PM2.5 and PM10 concentration prediction in air pollution using machine learning and deep learning models",
        "journal": "Environmental Challenges",
        "year": 2025,
        "vol": "18",
        "pages": "101062",
        "doi": "10.1016/j.envc.2024.101062",
        "used_in": "Literature Review §2.3 — Tổng quan hệ thống về ML/DL trong dự báo bụi mịn PM2.5/PM10",
        "context": "Nghiên cứu tổng quan hệ thống mới nhất năm 2025 về các phương pháp ML/DL dự báo bụi mịn. Đề án sử dụng làm cơ sở đối chuẩn và so sánh hiệu năng các kiến trúc tiên tiến trên thế giới.",
        "quote": "Deep learning models, specifically hybrid configurations, demonstrate consistently lower error rates in short-term ambient particulate prediction.",
        "location": "Section 3: Comparative Analysis",
        "pdf_link": "https://doi.org/10.1016/j.envc.2024.101062",
    },
    # ── [27] Houdou et al. (2024) ──
    "houdou2024": {
        "id": 27,
        "authors": "A. Houdou, A. S. O. Lundervold, and A. Lundervold",
        "title": "Interpretable machine learning approaches for forecasting and predicting air pollution: A systematic review",
        "journal": "Aerosol and Air Quality Research",
        "year": 2024,
        "vol": "24(5)",
        "pages": "230151",
        "doi": "10.4209/aaqr.230151",
        "used_in": "Explainability Hub & §2.4, §4.6 — Tổng quan có hệ thống về XAI trong chất lượng không khí",
        "context": "Nghiên cứu tổng quan chỉ ra rằng SHAP chiếm 46,4% các ứng dụng giải thích mô hình chất lượng không khí, trở thành phương pháp diễn giải phổ biến và tin cậy nhất hiện nay.",
        "quote": "SHAP has emerged as the most widely used interpretation technique, effectively bridging high performance with policy-oriented transparency.",
        "location": "Section 4: Explainability Methods",
        "pdf_link": "docs/references/[52] Houdou et al. (2024).pdf",
    },
    # ── [28] Tuyet et al. (2024) ──
    "tuyet2024": {
        "id": 28,
        "authors": "N. T. N. Tuyet, T. H. Quan, and N. T. K. Oanh",
        "title": "Statistical and machine learning approaches for estimating pollution of fine particulate matter (PM2.5) in Vietnam",
        "journal": "Journal of Environmental Engineering and Landscape Management",
        "year": 2024,
        "vol": "32(4)",
        "pages": "292–304",
        "doi": "10.3846/jeelm.2024.21852",
        "used_in": "Literature Review §2.3 — Thực trạng và đặc thù quan trắc bụi mịn PM2.5 tại Việt Nam",
        "context": "Công trình của các tác giả Việt Nam khảo sát nồng độ bụi mịn PM2.5 trong điều kiện khí hậu nhiệt đới gió mùa tại Việt Nam, cung cấp cơ sở dữ liệu đối chứng địa phương cho đề án tại Đồng bằng sông Cửu Long.",
        "quote": "Machine learning algorithms significantly improve ambient PM2.5 estimation accuracy across Vietnamese monitoring networks.",
        "location": "Section 2: Methodology & Results",
        "pdf_link": "https://doi.org/10.3846/jeelm.2024.21852",
    },
    # ── [29] Rakholia et al. (2022) ──
    "rakholia2022": {
        "id": 29,
        "authors": "R. Rakholia, P. Sharma, and T. N. Quang",
        "title": "AI-based air quality PM2.5 forecasting models for developing countries: A case study of Ho Chi Minh City, Vietnam",
        "journal": "Urban Climate",
        "year": 2022,
        "vol": "44",
        "pages": "101315",
        "doi": "10.1016/j.uclim.2022.101315",
        "used_in": "Literature Review §2.3 — Nghiên cứu điển hình về AI dự báo PM2.5 tại miền Nam Việt Nam",
        "context": "Nghiên cứu điển hình tại TP. Hồ Chí Minh chứng minh hiệu quả của các mô hình AI trong dự báo bụi mịn PM2.5, làm mốc mỏ neo đối chiếu điều kiện khí hậu Nam Bộ tương đồng với Sa Đéc.",
        "quote": "AI-based models demonstrate strong adaptability for capturing complex urban air quality variations in developing countries.",
        "location": "Section 4: Case Study Findings",
        "pdf_link": "https://doi.org/10.1016/j.uclim.2022.101315",
    },
    # ── [30] Shetty et al. (2024) ──
    "shetty2024": {
        "id": 30,
        "authors": "S. Shetty et al.",
        "title": "Daily high-resolution surface PM2.5 estimation over Europe by ML-based downscaling",
        "journal": "Environmental Research",
        "year": 2024,
        "vol": "252",
        "pages": "120363",
        "doi": "10.1016/j.envres.2024.120363",
        "used_in": "Literature Review §2.3 — Kỹ thuật thu hẹp tỷ lệ (Downscaling) và phân giải thời gian cao",
        "context": "Ứng dụng Machine Learning để thu hẹp tỷ lệ (downscaling) và tái lấy mẫu thời gian, là cơ sở đối chứng cho việc xây dựng dữ liệu đa phân giải (15m, 30m, 1h) trong đề án.",
        "quote": "Machine-learning downscaling allows fine-grained temporal and spatial representation of particulate matter concentrations.",
        "location": "Section 3: Downscaling Architecture",
        "pdf_link": "https://doi.org/10.1016/j.envres.2024.120363",
    },
    # ── [31] Tian et al. (2024) ──
    "tian2024": {
        "id": 31,
        "authors": "H. Tian, S. Li, and J. Wu",
        "title": "A novel stacking ensemble learning approach for predicting PM2.5 concentration",
        "journal": "Applied Sciences",
        "year": 2024,
        "vol": "14(12)",
        "pages": "5062",
        "doi": "10.3390/app14125062",
        "used_in": "Modeling & §3.3 — Kỹ thuật Stacking Ensemble mới trong dự báo PM2.5",
        "context": "Đề xuất phương pháp kết hợp Stacking Ensemble mới, kiểm chứng việc kết hợp các mô hình đa dạng giúp giảm sai số dự báo vượt bậc so với các mô hình đơn.",
        "quote": "The novel stacking ensemble framework significantly mitigates individual model biases and enhances predictive stability.",
        "location": "Abstract & Methodology",
        "pdf_link": "https://doi.org/10.3390/app14125062",
    },
    # ── [32] Inam et al. (2024) ──
    "inam2024": {
        "id": 32,
        "authors": "S. A. Inam et al.",
        "title": "PR-FCNN: A data-driven hybrid approach for predicting PM2.5 concentration",
        "journal": "Earth Science Informatics",
        "year": 2024,
        "vol": "17(2)",
        "pages": "1251–1264",
        "doi": "10.1007/s12145-024-01254-0",
        "used_in": "Literature Review §2.3 — Mạng nơ-ron tích chập lai dự báo ô nhiễm không khí",
        "context": "Nghiên cứu mô hình nơ-ron lai kết hợp trích xuất đặc trưng tự động và mạng fully-connected, làm nguồn tham khảo cho kiến trúc học sâu trong đề án.",
        "quote": "Data-driven hybrid approaches successfully learn high-dimensional interactions from multi-sensor air quality streams.",
        "location": "Section 3: Architecture",
        "pdf_link": "https://doi.org/10.1007/s12145-024-01254-0",
    },
    # ── [33] Kim et al. (2023) ──
    "kim2023": {
        "id": 33,
        "authors": "B. Kim, S. Park, and J. Lee",
        "title": "PM2.5 concentration forecasting using weighted Bi-LSTM and random forest feature importance",
        "journal": "Atmosphere",
        "year": 2023,
        "vol": "14(6)",
        "pages": "968",
        "doi": "10.3390/atmos14060968",
        "used_in": "Modeling & §3.3 — Kết hợp trọng số Bi-LSTM và xếp hạng đặc trưng Random Forest",
        "context": "Ứng dụng cơ chế trọng số kết hợp Bi-LSTM và Random Forest để lọc đặc trưng quan trọng, là tiền đề cho mô hình Ensemble_Weighted_30m đạt MASE = 0,382 tại mốc 6h trong đề án.",
        "quote": "Weighting mechanisms based on feature importance enhance the predictive capabilities of bidirectional recurrent architectures.",
        "location": "Section 3: Model Pipeline",
        "pdf_link": "https://doi.org/10.3390/atmos14060968",
    },
    # ── [34] Willmott & Matsuura (2005) ──
    "willmott2005": {
        "id": 34,
        "authors": "C. J. Willmott and K. Matsuura",
        "title": "Advantages of the mean absolute error (MAE) over the root mean square error (RMSE) in assessing average model performance",
        "journal": "Climate Research",
        "year": 2005,
        "vol": "30(1)",
        "pages": "79–82",
        "doi": "10.3354/cr030079",
        "used_in": "Evaluation & §3.5 — Biện luận lý thuyết ưu tiên MAE hơn RMSE trong đánh giá sai số trung bình",
        "context": "Chứng minh MAE phản ánh sai số trung bình thực tế tự nhiên hơn RMSE. RMSE phạt bình phương quá nặng các ngoại lệ đỉnh nhọn vốn xuất hiện thường xuyên trong chuỗi PM2.5.",
        "quote": "MAE is the most natural measure of average error magnitude, whereas RMSE is ambiguously related to average error and heavily influenced by outliers.",
        "location": "Abstract & Discussion",
        "pdf_link": "docs/references/[02] Willmott 2005.pdf",
    },
    # ── [35] Gneiting & Raftery (2007) ──
    "gneiting2007": {
        "id": 35,
        "authors": "T. Gneiting and A. E. Raftery",
        "title": "Strictly proper scoring rules, prediction, and estimation",
        "journal": "Journal of the American Statistical Association",
        "year": 2007,
        "vol": "102(477)",
        "pages": "359–378",
        "doi": "10.1198/016214506000001437",
        "used_in": "Evaluation & §3.5 — Thước đo chấm điểm chuẩn tắc (CRPS) cho dự báo phân phối xác suất",
        "context": "Lý thuyết nền tảng về Strictly Proper Scoring Rules. CRPS đánh giá đồng thời độ chính xác điểm và độ tin cậy của phân phối xác suất, khuyến khích mô hình đưa ra dự báo trung thực.",
        "quote": "Strictly proper scoring rules encourage the forecaster to make careful assessments and to be honest.",
        "location": "Abstract & Section 1",
        "pdf_link": "docs/references/[03] Gneiting 2007.pdf",
    },
    # ── [36] Rosner (1983) ──
    "rosner1983": {
        "id": 36,
        "authors": "B. Rosner",
        "title": "Percentage points for a generalized ESD many-outlier procedure",
        "journal": "Technometrics",
        "year": 1983,
        "vol": "25(2)",
        "pages": "165–172",
        "doi": "10.1080/00401706.1983.10487848",
        "used_in": "Pipeline ② — Thuật toán Generalized ESD phát hiện đa dị biệt (Seasonal-ESD)",
        "context": "Thủ tục Generalized ESD phát hiện đồng thời nhiều ngoại lệ mà không bị hiện tượng che giấu (masking effect). Đề tài áp dụng Seasonal-ESD kết hợp STL để loại bỏ dị biệt cảm biến mà vẫn giữ nguyên các đỉnh ô nhiễm thực.",
        "quote": "A generalized ESD procedure controls the type I error rate under both the hypothesis of no outliers and alternative outlier hypotheses.",
        "location": "Abstract & Section 2",
        "pdf_link": "docs/references/[14] Rosner 1983_S-ESD Outliers.pdf",
    },
    # ── [37] Cleveland et al. (1990) ──
    "cleveland1990": {
        "id": 37,
        "authors": "R. B. Cleveland, W. S. Cleveland, J. E. McRae, and I. Terpenning",
        "title": "STL: A seasonal-trend decomposition procedure based on loess",
        "journal": "Journal of Official Statistics",
        "year": 1990,
        "vol": "6(1)",
        "pages": "3–73",
        "doi": "",
        "used_in": "EDA & Pipeline ② — Phân rã mùa - xu hướng STL (Loess-based Seasonal-Trend Decomposition)",
        "context": "Phương pháp STL phân tách chuỗi thời gian thành 3 thành phần: Trend (xu hướng), Seasonal (chu kỳ mùa), và Remainder (phần dư). Đề án dùng STL để phân tích nhịp ngày đêm 24h và hỗ trợ thuật toán Seasonal-ESD.",
        "quote": "STL is a versatile and robust procedure for decomposing time series into trend, seasonal, and irregular components.",
        "location": "Section 1: Introduction",
        "pdf_link": "docs/references/[21] Robert B. Cleveland 1990_stl-a-seasonal-trend-decomposition-procedure-based-on-loess.pdf",
    },
    # ── [38] Troyanskaya et al. (2001) ──
    "troyanskaya2001": {
        "id": 38,
        "authors": "O. Troyanskaya et al.",
        "title": "Missing value estimation methods for DNA microarrays",
        "journal": "Bioinformatics",
        "year": 2001,
        "vol": "17(6)",
        "pages": "520–525",
        "doi": "10.1093/bioinformatics/17.6.520",
        "used_in": "Pipeline ③ & §4.1.3, Bảng 4.17 — Thuật toán KNN Imputation xử lý dữ liệu khuyết khoảng vừa (6–24 bước)",
        "context": "Thuật toán KNN Imputation nội suy giá trị khuyết dựa trên k mẫu quan sát lân cận có đặc tính vi khí hậu tương đồng nhất. Đề án kiểm định độ nhạy k và chọn k=5 tối ưu với MAE thấp nhất (32,25 µg/m³).",
        "quote": "KNN-based estimation consistently outperforms row-average and singular value decomposition imputation methods across diverse patterns of missingness.",
        "location": "Abstract & Results",
        "pdf_link": "docs/references/[22] Troyanskaya 2001_KNN Imputation.pdf",
    },
    # ── [39] Akiba et al. (2019) ──
    "akiba2019": {
        "id": 39,
        "authors": "T. Akiba, S. Sano, T. Yanase, T. Ohta, and M. Koyama",
        "title": "Optuna: A next-generation hyperparameter optimization framework",
        "journal": "Proc. 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining",
        "year": 2019,
        "vol": "",
        "pages": "2623–2631",
        "doi": "10.1145/3292500.3330701",
        "used_in": "Training & §3.3 — Tối ưu hóa siêu tham số tự động bằng thuật toán TPE Bayes",
        "context": "Framework tối ưu hóa siêu tham số thế hệ mới với thuật toán TPE (Tree-structured Parzen Estimator) và cơ chế cắt tỉa sớm (pruning). Tự động tìm kiếm siêu tham số tối ưu cho LightGBM, LSTM, GRU, TFT.",
        "quote": "Optuna features an define-by-run API that enables users to dynamically construct parameter search spaces with state-of-the-art pruning.",
        "location": "Section 1: Introduction",
        "pdf_link": "docs/references/[12] Akiba 2019_Optuna (TPE Sampler).pdf",
    },
    # ── [40] Diebold & Mariano (1995) ──
    "diebold1995": {
        "id": 40,
        "authors": "F. X. Diebold and R. S. Mariano",
        "title": "Comparing predictive accuracy",
        "journal": "Journal of Business & Economic Statistics",
        "year": 1995,
        "vol": "13(3)",
        "pages": "253–263",
        "doi": "10.1080/07350015.1995.10524599",
        "used_in": "Evaluation & §3.5, §4.4 — Kiểm định thống kê Diebold-Mariano (DM test) so sánh độ chính xác",
        "context": "Kiểm định DM kiểm tra xem sự chênh lệch hiệu năng giữa hai mô hình (vd: GRU vs LightGBM) có ý nghĩa thống kê hay chỉ do ngẫu nhiên, giúp khẳng định tính khoa học của kết luận đề án.",
        "quote": "We propose tests of the null hypothesis of no difference in the accuracy of two competing forecasts that are valid under wide conditions.",
        "location": "Abstract & Section 2",
        "pdf_link": "docs/references/[11] Diebold 1995_Diebold-Mariano Test.pdf",
    },
    # ── [41] Lundberg & Lee (2017) ──
    "lundberg2017": {
        "id": 41,
        "authors": "S. M. Lundberg and S.-I. Lee",
        "title": "A unified approach to interpreting model predictions",
        "journal": "Advances in Neural Information Processing Systems (NeurIPS)",
        "year": 2017,
        "vol": "30",
        "pages": "4765–4774",
        "doi": "",
        "used_in": "Explainability Hub & §4.6 — Giải thích mô hình bằng SHAP (SHapley Additive exPlanations)",
        "context": "Khung lý thuyết thống nhất giải thích mô hình dựa trên giá trị Shapley từ lý thuyết trò chơi hợp tác. Áp dụng Tree SHAP để giải thích vai trò của 119 đặc trưng và xác định điểm bùng phát ô nhiễm 14–17 µg/m³.",
        "quote": "SHAP values represent the only additive feature attribution method that satisfies both local accuracy and consistency.",
        "location": "Section 1: Introduction & Section 3: Additive Feature Attribution Methods",
        "pdf_link": "docs/references/[20] Lundberg 2017_SHAP.pdf",
    },
    # ── [42] Romano et al. (2019) ──
    "romano2019": {
        "id": 42,
        "authors": "Y. Romano, E. Patterson, and E. J. Candès",
        "title": "Conformalized quantile regression",
        "journal": "Advances in Neural Information Processing Systems (NeurIPS)",
        "year": 2019,
        "vol": "32",
        "pages": "3543–3553",
        "doi": "",
        "used_in": "Prediction Intervals & §3.6, §4.7 — Conformalized Quantile Regression (CQR) sinh khoảng dự báo bất định (PIs)",
        "context": "CQR kết hợp hồi quy phân vị với Conformal Prediction để tạo ra các khoảng dự báo có bảo đảm toán học về độ phủ biên (Marginal Coverage) 90% không phụ thuộc giả định phân phối.",
        "quote": "Conformalized quantile regression is a generic method for constructing predictive intervals that attain valid marginal coverage without any distributional assumptions.",
        "location": "Abstract & Section 2: Conformalized Quantile Regression",
        "pdf_link": "docs/references/[04] Romano 2019_CQR (Conformal Quantile).pdf",
    },
    # ── [43] Box & Cox (1964) ──
    "boxcox1964": {
        "id": 43,
        "authors": "G. E. P. Box and D. R. Cox",
        "title": "An analysis of transformations",
        "journal": "Journal of the Royal Statistical Society: Series B (Methodological)",
        "year": 1964,
        "vol": "26(2)",
        "pages": "211–243",
        "doi": "10.1111/j.2517-6161.1964.tb00553.x",
        "used_in": "Pipeline & §4.1.1 — Biến đổi Box-Cox ổn định phương sai cho chuỗi phân phối lệch phải",
        "context": "Phép biến đổi lũy thừa Box-Cox đưa chuỗi phân phối lệch về gần phân phối chuẩn và ổn định phương sai. Với PM2.5 tại Sa Đéc, λ tối ưu gần 0, biện luận cho việc áp dụng Log Transform trên mô hình Deep Learning.",
        "quote": "In this paper we make the assumption that a normal, homoscedastic linear model is appropriate after some suitable power transformation.",
        "location": "Section 1: Introduction",
        "pdf_link": "docs/references/[13] Box & Cox 1964_Box-Cox Transform.pdf",
    },
    # ── [44] Gal & Ghahramani (2016) ──
    "gal2016": {
        "id": 44,
        "authors": "Y. Gal and Z. Ghahramani",
        "title": "Dropout as a Bayesian approximation: Representing model uncertainty in deep learning",
        "journal": "Proc. 33rd International Conference on Machine Learning (ICML)",
        "year": 2016,
        "vol": "",
        "pages": "1050–1059",
        "doi": "",
        "used_in": "Uncertainty Estimation & §3.6 — Monte Carlo Dropout ước lượng độ bất định mô hình",
        "context": "Chứng minh Dropout trong giai đoạn suy luận (Inference time) tương đương với xấp xỉ biến phân trong mô hình Quá trình Gaussian Bayes, dùng làm mô hình đối chứng cho Conformal Prediction.",
        "quote": "Dropout in deep neural networks can be cast as an approximate Bayesian inference in deep Gaussian processes.",
        "location": "Abstract & Section 2",
        "pdf_link": "docs/references/[18] Gal 2016_ MC Dropout.pdf",
    },
    # ── [45] Lakshminarayanan et al. (2017) ──
    "lakshminarayanan2017": {
        "id": 45,
        "authors": "B. Lakshminarayanan, A. Pritzel, and C. Blundell",
        "title": "Simple and scalable predictive uncertainty estimation using deep ensembles",
        "journal": "Advances in Neural Information Processing Systems (NeurIPS)",
        "year": 2017,
        "vol": "30",
        "pages": "6402–6413",
        "doi": "",
        "used_in": "Uncertainty Estimation & §3.6 — Deep Ensembles ước lượng độ bất định ngẫu nhiên & mô hình",
        "context": "Phương pháp đơn giản và có khả năng mở rộng để ước lượng độ bất định dự báo bằng cách kết hợp tập hợp các mạng nơ-ron sâu huấn luyện với các khởi tạo ngẫu nhiên khác nhau.",
        "quote": "Deep ensembles are simple to implement, readily parallelizable, require very little hyperparameter tuning, and yield high-quality uncertainty estimates.",
        "location": "Abstract & Introduction",
        "pdf_link": "docs/references/[19] Lakshminarayanan_Deep Ensembles.pdf",
    },
    # ── [46] Armstrong (2001) ──
    "armstrong2001": {
        "id": 46,
        "authors": "J. S. Armstrong",
        "title": "Principles of Forecasting: A Handbook for Researchers and Practitioners",
        "journal": "Boston, MA, USA: Kluwer Academic Publishers",
        "year": 2001,
        "vol": "",
        "pages": "",
        "doi": "10.1007/978-0-306-47630-3",
        "used_in": "Methodology §3.4 — Nguyên lý kiểm định ngoài mẫu (Out-of-sample Evaluation) & R² âm",
        "context": "Cẩm nang kinh điển về các nguyên lý dự báo: Khẳng định tính chuẩn tắc của việc đánh giá mô hình hoàn toàn ngoài mẫu (Out-of-sample). Giải thích hiện tượng R² có thể nhận giá trị âm khi dự báo ngoài mẫu trên dữ liệu có biến động mạnh và ngoại lệ đuôi dài.",
        "quote": "Evaluating forecasts on out-of-sample data is essential to avoid overfitting and ensure genuine predictive validity.",
        "location": "Chapter 13: Evaluating Forecasting Methods",
        "pdf_link": "https://doi.org/10.1007/978-0-306-47630-3",
    },
    # ── [47] Cleveland (1993) ──
    "cleveland1993": {
        "id": 47,
        "authors": "W. S. Cleveland",
        "title": "Visualizing Data",
        "journal": "Summit, NJ, USA: Hobart Press",
        "year": 1993,
        "vol": "",
        "pages": "",
        "doi": "",
        "used_in": "EDA & §4.1 — Nguyên lý trực quan hóa dữ liệu thống kê (Q-Q plot, Residual plot, Scatter)",
        "context": "Cuốn sách nền tảng về trực quan hóa dữ liệu thống kê, cung cấp cơ sở phương pháp luận cho biểu đồ Q-Q phân vị, phân tích phần dư, biểu đồ tán xạ đa biến và chẩn đoán phân phối đuôi dài của PM2.5.",
        "quote": "Visualizing data is a vital step in data analysis, revealing patterns, outliers, and subtle relationships that summary statistics fail to capture.",
        "location": "Chapter 2: Univariate Data & Chapter 3: Bivariate Data",
        "pdf_link": "docs/references/[38] Cleveland-1993-Visualizing-Data.pdf",
    },
    # ── [48] Vishwas & Patel (2020) ──
    "vishwas2020": {
        "id": 48,
        "authors": "B. V. Vishwas and A. Patel",
        "title": "Hands-on Time Series Analysis with Python: From Basics to Bleeding Edge Techniques",
        "journal": "New York, NY, USA: Apress",
        "year": 2020,
        "vol": "",
        "pages": "",
        "doi": "10.1007/978-1-4842-5992-4",
        "used_in": "Pipeline & Feature Engineering §4.2 — Kỹ thuật trích xuất đặc trưng chuỗi thời gian bằng Python",
        "context": "Hướng dẫn thực hành các kỹ thuật dự báo chuỗi thời gian hiện đại: Xây dựng cửa sổ trượt (Rolling windows), biến đổi làm mịn hàm mũ (EWM), và mã hóa chu kỳ hàm lượng giác.",
        "quote": "Feature engineering transforms sequential time series into expressive tabular representations suitable for tree-based boosting.",
        "location": "Chapter 4: Feature Engineering for Time Series",
        "pdf_link": "docs/references/[36] Hands-on Time Series Analysis With Python From Basics To Bleeding Edge Techniques (B. V. Vishwas, Ashish Patel) (Z-Library).pdf",
    },
    # ── [49] Christ et al. (2018) ──
    "christ2018": {
        "id": 49,
        "authors": "M. Christ, N. Braun, J. Neuffer, and A. W. Kempa-Liehr",
        "title": "Time series FeatuRe Extraction on basis of Scalable Hypothesis tests (tsfresh -- A Python package)",
        "journal": "Neurocomputing",
        "year": 2018,
        "vol": "307",
        "pages": "72–77",
        "doi": "10.1016/j.neucom.2018.03.067",
        "used_in": "Feature Engineering & §4.2, Phụ lục 1 — Kỹ nghệ đặc trưng tự động và kiểm định giả thuyết",
        "context": "Thư viện tsfresh trích xuất hàng trăm đặc trưng thời gian và kiểm định ý nghĩa thống kê với biến mục tiêu, cung cấp cơ sở cho việc thiết kế 119 đặc trưng phi rò rỉ của đề án.",
        "quote": "tsfresh automates the extraction of comprehensive time series feature representations combined with rigorous feature selection tests.",
        "location": "Abstract & Section 2",
        "pdf_link": "docs/references/[33] M. Christ 2018_Time Series FeatuRe Extraction on basis of Scalable Hypothesis tests.pdf",
    },
    # ── [50] Fisher et al. (2019) ──
    "fisher2019": {
        "id": 50,
        "authors": "A. Fisher, C. Rudin, and F. Dominici",
        "title": "All models are wrong, but many are useful: Learning a variable's importance by considering all support models simultaneously",
        "journal": "Journal of Machine Learning Research",
        "year": 2019,
        "vol": "20(177)",
        "pages": "1–81",
        "doi": "",
        "used_in": "Explainability Hub & §4.6, Bảng 4.5 — Permutation Importance (Model Reliance) cho Deep Learning",
        "context": "Cơ sở lý thuyết của Permutation Importance (Model Reliance): Đánh giá mức tăng sai số khi hoán vị ngẫu nhiên một đặc trưng để đo lường tầm quan trọng độc lập mô hình (model-agnostic) cho GRU/LSTM/TFT.",
        "quote": "We propose reliance as a measure of variable importance: the increase in prediction error when the variable is scrambled.",
        "location": "Abstract & Section 2",
        "pdf_link": "docs/references/[50] Fisher et al. (2019) - Permutation Importance.pdf",
    },
    # ── [51] Gu et al. (2021) ──
    "gu2021": {
        "id": 51,
        "authors": "Y. Gu, Q. Li, and Y. Du",
        "title": "Hybrid interpretable predictive machine learning model for air pollution prediction",
        "journal": "Neurocomputing",
        "year": 2021,
        "vol": "466",
        "pages": "341–355",
        "doi": "10.1016/j.neucom.2021.09.041",
        "used_in": "Explainability Hub & §4.6 — Mô hình máy học lai kết hợp XAI giải thích dự báo ô nhiễm không khí",
        "context": "Nghiên cứu ứng dụng thực nghiệm chứng minh việc kết hợp mô hình học máy (LightGBM/XGBoost) với SHAP giúp hiểu rõ cơ chế tương tác phức tạp của khí tượng đến nồng độ bụi mịn.",
        "quote": "The integration of predictive machine learning with interpretability methods like SHAP facilitates understanding variable contributions in air pollution forecasting.",
        "location": "Abstract & Section 4",
        "pdf_link": "docs/references/[51] Gu et al. (2021) _Hybrid interpretable predictive machine learning model for air pollution prediction.pdf",
    },
    # ── [52] Kaveh et al. (2025) ──
    "kaveh2025": {
        "id": 52,
        "authors": "M. Kaveh, M. S. Mesgari, and M. Pal",
        "title": "A novel evolutionary deep learning approach for PM2.5 prediction using remote sensing and meteorological data",
        "journal": "ISPRS International Journal of Geo-Information",
        "year": 2025,
        "vol": "14(2)",
        "pages": "42",
        "doi": "10.3390/ijgi14020042",
        "used_in": "Literature Review §2.3 — Học sâu tiến hóa kết hợp dữ liệu khí tượng và viễn thám dự báo PM2.5",
        "context": "Công trình mới nhất (2025) áp dụng thuật toán tiến hóa kết hợp học sâu và dữ liệu khí tượng đa nguồn để dự báo PM2.5, làm cơ sở đối chuẩn quốc tế cập nhật cho đề án.",
        "quote": "Evolutionary deep learning frameworks optimize hyperparameter configurations while effectively integrating heterogeneous meteorological streams.",
        "location": "Section 3: Methodology",
        "pdf_link": "https://doi.org/10.3390/ijgi14020042",
    },
    # ── [53] Bhardwaj et al. (2023) ──
    "bhardwaj2023": {
        "id": 53,
        "authors": "P. Bhardwaj, P. Sharma, and T. N. Quang",
        "title": "Machine learning and explainable AI for ambient PM2.5 forecasting",
        "journal": "Environmental Science and Pollution Research",
        "year": 2023,
        "vol": "30",
        "pages": "68210–68225",
        "doi": "10.1007/s11356-023-27083-9",
        "used_in": "Explainability Hub & §2.4, §4.6 — XAI giải thích mô hình cây kết hợp SHAP (R² = 0,87)",
        "context": "Nghiên cứu ứng dụng XGBoost và Tree SHAP trong dự báo PM2.5 (đạt R² = 0,87). Đề án đối chứng trực tiếp với kết quả của Bhardwaj để phân tích điểm bùng phát (Tipping Point 14–17 µg/m³) tại Sa Đéc.",
        "quote": "Explainable AI techniques provide critical insights into the non-linear interaction effects of meteorological factors on PM2.5 concentrations.",
        "location": "Abstract & Section 4",
        "pdf_link": "https://doi.org/10.1007/s11356-023-27083-9",
    },
    # ── [54] Gibbs & Candès (2021) ──
    "gibbs2021": {
        "id": 54,
        "authors": "I. Gibbs and E. Candès",
        "title": "Adaptive conformal inference under distribution shift",
        "journal": "Advances in Neural Information Processing Systems (NeurIPS)",
        "year": 2021,
        "vol": "34",
        "pages": "1660–1672",
        "doi": "",
        "used_in": "Prediction Intervals & §3.6, §4.7, Bảng 4.18 — Adaptive Conformal Inference (ACI) cập nhật thích ứng dưới dịch chuyển phân phối",
        "context": "Thuật toán ACI cập nhật mức độ bao phủ thích ứng trực tuyến với tham số tốc độ học γ. Giúp khoảng tin cậy của đề án duy trì độ phủ thực nghiệm 91,0% (vượt mục tiêu 90%) ngay cả khi có biến động thời tiết đột ngột.",
        "quote": "Adaptive conformal inference achieves asymptotic coverage guarantees even in the presence of arbitrary, unknown distribution shifts.",
        "location": "Abstract & Section 2: Adaptive Conformal Inference",
        "pdf_link": "https://arxiv.org/pdf/2106.00170.pdf",
    },
}

# Backward compatibility aliases
IEEE_REFS["hyndman2021"] = IEEE_REFS["hyndman2006"]
IEEE_REFS["zhang2017"] = IEEE_REFS["zhang2011"]
IEEE_REFS["nguyen2024"] = IEEE_REFS["tuyet2024"]
IEEE_REFS["kang2017"] = IEEE_REFS["christ2018"]


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


def render_references_section(title: str = "📚 Tài Liệu Tham Khảo (IEEE)", filter_ids: list | None = None):
    """Render a full IEEE-formatted references list at the bottom of a page."""
    _ensure_css()

    refs = list(IEEE_REFS.values())
    if filter_ids is not None:
        refs = [r for r in refs if r["id"] in filter_ids]

    # De-duplicate by reference ID to ensure exactly 54 unique canonical items even with aliases
    unique_by_id = {r["id"]: r for r in refs}
    sorted_refs = sorted(unique_by_id.values(), key=lambda r: r["id"])

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
