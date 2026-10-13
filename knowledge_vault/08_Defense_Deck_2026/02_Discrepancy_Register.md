---
title: "Bảng Đăng Ký Sai Lệch & Chuẩn Hóa Số Liệu Trình Bày (Discrepancy Register)"
tags: [defense-2026, discrepancy, audit, reconciliation, speaker-notes, alignment]
aliases: ["Discrepancy Register 2026", "Bảng Đối Soát Sai Lệch", "Chuẩn Hóa Số Liệu Bảo Vệ"]
domain: defense
created: "2026-10-04"
status: completed
---

# ⚖️ Bảng Đăng Ký Sai Lệch (Discrepancy Register) & Chuẩn Hóa Phát Biểu Hội Đồng 2026

> **Mục tiêu:** Nhận diện toàn bộ các điểm không đồng nhất về số liệu giữa **Báo cáo Luận văn chính thức đã nộp** (`docs/THESIS_DRAFT_CTU_1799.md`), **Playbook cũ 24 slide** (`docs/SLIDES_DEFENSE_PLAYBOOK.md`), và **Mã nguồn Dashboard/Test suite hiện hành** (`app.py`, `tests/`).  
> **Nguyên tắc phân xử:** Ưu tiên số 1 tuyệt đối cho Báo cáo Luận văn chính thức (`THESIS_DRAFT_CTU_1799.md`). Với mỗi điểm lệch, tài liệu này cung cấp **Số liệu chuẩn**, **Cơ sở đối chiếu** và **Kịch bản giải trình mẫu cho học viên Hoàng Xuân Trí**.

---

## 📊 1. Bảng Tổng Hợp Đối Soát Chi Tiết 8 Điểm Sai Lệch

| # | Hạng mục đối soát | Báo cáo chính thức (`THESIS_DRAFT`) | Playbook cũ (`SLIDES_PLAYBOOK`) | Codebase / Test Suite thực tế | QUYẾT ĐỊNH KHUYẾN NGHỊ |
| :-: | :--- | :--- | :--- | :--- | :--- |
| **1** | **Số lượng Automated Tests** | **193 tests** nghiệp vụ lõi (Bảng PL4.1); **204 tests** mở rộng CI/CD (dòng 1559) | **248 tests** (Slide 19, kịch bản cũ) | **462 tests** thu thập bởi `pytest --collect-only` | **Dùng 193 tests làm số liệu chính thức; 462 tests cho minh chứng CI/CD thực tế.** |
| **2** | **Số lượng Đặc trưng Kỹ nghệ** | **119 đặc trưng** thuộc 7 nhóm (Bảng 3.3 & Phụ lục 1) | **57 đặc trưng** (Slide 08) | **119 đặc trưng** trên UI `app.py` và module `builder.py` | **119 đặc trưng (Chuẩn tuyệt đối).** |
| **3** | **Số lượng Mô hình Thực nghiệm** | **41 cấu hình** thực nghiệm (dòng 693, 1299); **11 mô hình** đối chuẩn Bảng 4.3 | **11 mô hình** đối chuẩn | 41 cấu hình trong `experiments.py` & `conclusion_page.py` | **41 cấu hình thực nghiệm, 11 mô hình tiêu biểu.** |
| **4** | **Số trang / Module Dashboard** | Kiến trúc 3 tầng (Hình 3.1) | 11 trang phân tích | **17 trang / module** (15 trang khoa học + 2 công cụ) | **17 trang hoàn chỉnh trên Dashboard.** |
| **5** | **Tỷ lệ Khuyết Thiếu Dữ Liệu** | **74% tích lũy** qua 38 tháng (89 ngày/năm; T2 64%, T9 73%) | 74% tích lũy | 74% tính từ 209.594 bản ghi thô | **74% lũy kế toàn bộ 38 tháng.** |
| **6** | **Hệ số $R^2$ Ngoài Mẫu Tại 6h** | **$R^2 = -0,044$** ở Ensemble 30m; **$R^2 = -0,774$** ở Persistence (Bảng 4.3) | Tập trung MASE thay vì $R^2$ | $R^2 = -0,044$ trên tập Anchor Test | **Báo cáo chuẩn: $R^2 = -0,044$ kèm MASE = 0,382.** |
| **7** | **Mẫu số Chuẩn Hóa MASE (BẪY #1)** | §3.6.2 + code `naive_mae_train`: mẫu số = MAE naive in-sample trên tập huấn luyện (theo từng tầm). §2.4.3 dòng 517 ghi nhầm mẫu số = Persistence test (2,596 / 6,932 / 6,327) | Mẫu số đồng nhất $MAE_{\text{naive}} = 1,821\,\mu\text{g/m}^3$ | `scripts/v9_standardize_evaluation.py` (`naive_mae_train`) | **Chuẩn: §3.6.2 + code. Giảm 49,6%/46,0% MAE là so sánh riêng trên test; CẤM quy đổi MASE thành tỷ lệ so với Persistence test.** |
| **8** | **Số đợt Ô Nhiễm Nguy Hại (WHO)** | **57 đợt** thực tế (Ground Truth Positives) trong tập Test (Bảng 4.8) | 57 đợt | Khớp 100% cột `is_alert` trên Test Set | **57 đợt mỏ neo cố định.** |

---

## 🔍 2. Phân Tích Cơ Sở Khoa Học & Kịch Bản Phát Biểu Chi Tiết

### Sai lệch 1: Số lượng Automated Tests (193 vs 204 vs 248 vs 462)
* **Bối cảnh lệch:**
  - Báo cáo luận văn niêm phong Bảng PL4.1 gồm đúng **193 tests** chia làm 5 nhóm chuyên trách (39 Anti-Leakage, 29 Preprocessing & Tiered Imputation, 29 Data Validation, 60 Anchor Metrics, 36 DL & API). Dòng 1559 ghi nhận bộ CI/CD mở rộng là **204 tests** (bổ sung 8 test kiểm định IEEE citations và 3 test kiểm tra hồi quy UI).
  - Bản thảo slide cũ ghi 248 tests (con số ở giai đoạn giữa dự án).
  - Hiện tại, khi kiểm thử toàn diện cả chatbot RAG, security duel, XAI audit và thesis alignment, pytest collect được **462 tests**.
* **Khuyến nghị phát biểu:**
  > *"Kính thưa Hội đồng, trong quyển báo cáo đề án chính thức nộp lên nhà trường, toàn bộ logic nghiệp vụ cốt lõi và kiểm định chống rò rỉ dữ liệu được niêm phong bất biến qua **193 bài kiểm thử tự động (100% Pass)**. Trong quá trình phát triển hoàn thiện hệ thống phần mềm và tích hợp quy chuẩn bảo mật DevSecOps/CI-CD, hệ thống hiện tại đã mở rộng lên **462 bài kiểm thử**, bảo đảm toàn bộ mã nguồn không phát sinh lỗi hồi quy."*

### Sai lệch 2: Số lượng Đặc trưng Kỹ nghệ (57 vs 119)
* **Bối cảnh lệch:**
  - Slide playbook cũ ghi 57 đặc trưng (phiên bản v5 rút gọn ban đầu).
  - Báo cáo chính thức Bảng 3.3 và Phụ lục 1 đã hoàn thiện đầy đủ **119 đặc trưng** thuộc 7 nhóm (4 Raw, 13 Calendar, 40 Lag, 24 Rolling, 6 EWM, 4 Diff, 28 Domain). Toàn bộ UI Dashboard hiện tại đều hiển thị 119 features.
* **Khuyến nghị phát biểu:**
  > *"Hệ thống xây dựng Feature Store chuẩn hóa gồm đúng **119 đặc trưng thời gian** thuộc 7 nhóm, bao phủ từ biến quan trắc vi khí tượng, chu kỳ lịch, biến trễ đa khung, trung bình trượt và sóng điều hòa Fourier. Toàn bộ 119 đặc trưng này đều tuân thủ nghiêm ngặt kỷ luật `shift(1)` chống rò rỉ dữ liệu."*

### Sai lệch 3: Số lượng Mô hình Thực nghiệm (11 vs 41)
* **Bối cảnh lệch:**
  - Bảng 4.3 chỉ trình bày 11 mô hình đại diện tiêu biểu nhất (Persistence, ARIMA, LightGBM, RF, ElasticNet, GRU, LSTM, TFT, Weighted Ensemble qua các mốc).
  - Mục 3.4 (dòng 693) và Mục 4.9 (dòng 1299) khẳng định đề tài thực nghiệm **41 cấu hình mô hình** (bao gồm các biến thể tham số, độ phân giải 15m/30m/1h và 3 chân trời 1h/6h/24h).
* **Khuyến nghị phát biểu:**
  > *"Đề tài đã thiết lập và thực nghiệm tổng cộng **41 cấu hình mô hình** thuộc 5 họ thuật toán lớn (Thống kê, Máy học dạng bảng, Học sâu hồi quy, Transformer và Ensemble). Để báo cáo súc tích trước Hội đồng, Bảng 4.3 trích lọc **11 cấu hình mô hình đại diện** xuất sắc nhất để đối chuẩn khoa học."*

### Sai lệch 4: Số trang Dashboard (11 vs 17)
* **Bối cảnh lệch:** Slide cũ ghi 11 trang; hiện tại file `app.py` có **17 trang/module** (15 trang khoa học phân theo 4 phase + 2 module Trợ lý AI và Quản lý nội dung).
* **Khuyến nghị phát biểu:**
  > *"Dashboard được xây dựng hoàn chỉnh với **17 trang chức năng**, bám sát dòng chảy nghiên cứu từ Giới thiệu, Khám phá dữ liệu EDA, Huấn luyện mô hình, Đánh giá đa mốc, Giải thích XAI, Định lượng khoảng tin cậy cho tới Cảnh báo điều hành và Trợ lý AI tích hợp."*

### Sai lệch 5: Tỷ lệ Khuyết thiếu (74% chung vs Từng tháng)
* **Bối cảnh lệch:** Tỷ lệ khuyết thiếu toàn chuỗi 38 tháng là 74% (do Drop 19.810 giờ). Tuy nhiên, độ phủ biến động mạnh theo mùa: Tháng 11 đạt 93% độ phủ (chỉ khuyết 7%), trong khi Tháng 2 khuyết 64% (bảo trì) và Tháng 9 khuyết 73% (mưa bão).
* **Khuyến nghị phát biểu:**
  > *"Tỷ lệ khuyết thiếu trung bình tích lũy là **74% qua 38 tháng** (tương ứng 89 ngày gián đoạn mỗi năm). Sự cố tập trung vào tháng 2 (do bảo trì trạm định kỳ) và tháng 9 (do ngập lụt mưa bão lớn ĐBSCL). Đây là căn cứ khoa học thôi thúc đề tài đề xuất chiến lược Nội suy phân tầng Tiered Imputation."*

### Sai lệch 6: Hệ số $R^2$ Nhận Giá Trị Âm Ngoài Mẫu (-0,044 ở 6h)
* **Bối cảnh lệch:** Một số thành viên hội đồng có thể thắc mắc tại sao $R^2 < 0$ ở Bảng 4.3 (Ensemble 30m ở 6h có $R^2 = -0,044$, Persistence ở 6h có $R^2 = -0,774$).
* **Khuyến nghị phát biểu:**
  > *"Theo lý thuyết kinh tế lượng chuỗi thời gian của GS. Rob Hyndman, trên tập kiểm thử độc lập ngoài mẫu (Out-of-sample Test), $R^2$ có miền giá trị từ âm vô cùng đến 1. Khi chuỗi xuất hiện các xung đỉnh ô nhiễm nhọn (spikes), nếu mô hình dự báo bị lệch pha nhẹ 1–2 bước thời gian, tổng bình phương sai số $SS_{\text{res}}$ sẽ vượt phương sai mẫu $SS_{\text{tot}}$, khiến $R^2$ bị âm nhẹ. Tuy nhiên, sai số tuyệt đối MAE của Ensemble giảm tới 49,6% so với Persistence và chỉ số chuẩn hóa MASE = 0,382 (< 1,0) khẳng định năng lực dự báo vượt trội thực tế của mô hình."*

### Sai lệch 7: [BẪY PHẢN BIỆN #1 - CRITICAL TRAP] Mẫu số Chuẩn Hóa MASE (In-sample Train Naive vs Out-of-sample Test Persistence) & Xung Đột Nội Bộ Văn Bản
* **Bẫy toán học & Sai lầm đồng nhất khái niệm:**
  - Tuyệt đối KHÔNG đồng nhất chỉ số chuẩn hóa $\text{MASE} = 0,382$ với tỷ lệ giảm sai số MAE trên tập Test ngoài mẫu ($49,6\%$).
  - Tại mốc 6 giờ: Sai số MAE của Ensemble là $3,493\,\mu\text{g/m}^3$, sai số MAE của Persistence trên tập Anchor Test là $6,932\,\mu\text{g/m}^3$. Tỷ lệ sai số thực tế trên tập Test là $\frac{3,493}{6,932} = 0,50389 \approx 50,4\%$, tương ứng với mức giảm **$49,6\%$ MAE** so với Persistence. Con số này **KHÔNG PHẢI là $0,382$!**
  - Tương tự tại 1 giờ: $\frac{\text{MAE}_{\text{GRU}}}{\text{MAE}_{\text{Pers\_test}}} = \frac{2,944}{2,596} = 1,134$, hoàn toàn khác con số $\text{MASE} = 0,667$.
* **Mâu thuẫn nội bộ trong văn bản Luận văn chính thức (`THESIS_DRAFT_CTU_1799.md`):**
  1. **Định nghĩa chuẩn tắc tại §3.6.2 (dòng 787):** MASE tuân thủ đúng khuyến nghị của Hyndman & Koehler (2006) và chuẩn thi M4 Competition: mẫu số là sai số naive 1 bước trên tập huấn luyện in-sample (`naive_mae_train`): $\text{MASE} = \frac{\text{MAE}_{\text{model}}}{\frac{1}{T-1}\sum_{t=2}^T |y_t - y_{t-1}|}$. Đây là công thức được hiện thực hóa 100% trong mã nguồn thực nghiệm (`scripts/v9_standardize_evaluation.py:58`), sinh ra giá trị $\text{MASE} = 0,382$ ở 6h và $\text{MASE} = 0,469$ ở 24h trong Bảng 4.3.
  2. **Sai sót mô tả tại §2.4.3 (dòng 517):** Văn bản ghi nhầm mẫu số MASE là sai số của Baseline Persistence trên tập Anchor Test Set ($2,596$ ở 1h; $6,932$ ở 6h; $6,327\,\mu\text{g/m}^3$ ở 24h).
* **Nguồn chuẩn mực (Single Source of Truth - SSOT):**
  - **SSOT toán học & mã nguồn:** §3.6.2 kết hợp code `scripts/v9_standardize_evaluation.py:58` (Mẫu số in-sample `naive_mae_train`).
  - **So sánh đối đầu trực tiếp ngoài mẫu:** Sử dụng tỷ lệ phần trăm giảm sai số MAE: giảm **$49,6\%$ tại 6h** ($3,493$ vs $6,932$) và **$46,0\%$ tại 24h** ($3,418$ vs $6,327$).
* **ĐIỀU CẤM KỴ KHI BÁO CÁO:**
  - ❌ **CẤM** phát biểu: *"MASE = 0,382 nghĩa là sai số của mô hình bằng 38,2% so với Persistence trên tập test"*.
  - ❌ **CẤM** phát biểu: *"Chỉ số MASE 0,382 chứng minh mô hình giảm 49,6% sai số so với Persistence"*.
* **Khuyến nghị phát biểu chuẩn xác trước Hội đồng:**
  > *"Kính thưa Quý Thầy/Cô, để đánh giá năng lực mô hình một cách toàn diện, đề tài tiếp cận theo hai góc độ độc lập:
  > 1. Về mặt chuẩn hóa độc lập thang đo (Scale-free), chỉ số MASE được tính chuẩn xác theo định nghĩa của Hyndman & Koehler (§3.6.2) với mẫu số là sai số naive 1 bước trên tập huấn luyện. Đạt $\text{MASE} = 0,382$ (< 1,0) tại 6 giờ khẳng định mô hình có năng lực học thực thụ, vượt trội so với quán tính in-sample.
  > 2. Về mặt đối đầu thực nghiệm ngoài mẫu trên tập Anchor Test: Mô hình Ensemble đạt MAE = 3,493 µg/m³, giảm tới 49,6% sai số so với mô hình cơ sở Persistence (6,932 µg/m³), và giảm 46,0% ở tầm 24 giờ."*
* **Kịch bản ứng đối nếu Hội đồng chỉ ra xung đột dòng 517 (§2.4.3):**
  > *"Dạ kính thưa Thầy/Cô, tác giả xin cảm ơn nhận xét cực kỳ sâu sắc của Thầy/Cô. Trong phần tổng quan lý thuyết tại Mục 2.4.3, tác giả đã viết ghép phép đối chiếu sai số tương đối trực tiếp trên tập kiểm thử ngoài mẫu (tỷ lệ $3,493 / 6,932 = 50,4\%$, tức giảm $49,6\%$ MAE). Trong khi đó, tại Mục 3.6.2 và toàn bộ mã nguồn thực nghiệm chuẩn hóa Bảng 4.3, đề tài hoàn toàn tuân thủ công thức MASE chuẩn của GS. Hyndman lấy mẫu số là naive sai phân trên tập huấn luyện in-sample. Tác giả xin được tiếp thu và hiệu chỉnh lại câu chữ ở Mục 2.4.3 trong bản in hoàn thiện cuối cùng để tường minh hóa sự khác biệt giữa hai thước đo này."*

### Sai lệch 8: Con số 57 Đợt Ô Nhiễm Trong Bảng 4.8
* **Bối cảnh lệch:** Tại sao cột cuối Bảng 4.8 mọi mô hình đều ghi số 57?
* **Khuyến nghị phát biểu:**
  > *"Con số 57 đợt là số quan sát thực tế (Ground Truth Positives: $TP + FN = 57$) có nồng độ PM2.5 vượt ngưỡng khuyến nghị $45\,\mu\text{g/m}^3$ trên tập kiểm thử mỏ neo độc lập. Cột này là hằng số mỏ neo thực tế cố định để so sánh khả năng bắt trúng đợt ô nhiễm giữa các mô hình."*

---
*Liên kết liên quan: [[01_Deck_Fact_Sheet_Index]] | [[00_Index_MOC]] | [`01_SECOND_BRAIN_FACTSHEET.md`](file:///Users/trihx/Desktop/time-series-forecasting/docs/defense-2026/01_SECOND_BRAIN_FACTSHEET.md)*
