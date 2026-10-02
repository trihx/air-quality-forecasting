---
title: "Bộ 20 câu hỏi - đáp chất vấn của Hội đồng bảo vệ luận văn thạc sĩ"
tags: [defense, faq, master-thesis, qna, chair, reviewers, playbook]
aliases: ["Bộ 20 câu hỏi Hội đồng", "Master Defense Q&A", "Hội đồng chất vấn"]
domain: defense
created: "2026-10-02"
status: completed
---

# ⚔️ Bộ 20 Câu Hỏi - Đáp Chất Vấn Của Hội Đồng Bảo Vệ Luận Văn Thạc Sĩ

Bộ câu hỏi được xây dựng từ thực tiễn phản biện học thuật, phân bổ theo 3 góc nhìn của Hội đồng: **Chủ tịch Hội đồng (Methodology)**, **Phản biện 1 (Data & Engineering)**, và **Phản biện 2 (AI/DL Depth & Ứng dụng)**.

---

## 🏛️ Nhóm 1: Phương Pháp Luận & Thống Kê (Chủ Tịch Hội Đồng)

### Q1: Tại sao tác giả dùng MASE thay vì RMSE hoặc MAE làm thước đo trung tâm?
**Trả lời:**
1. RMSE và MAE là các chỉ số phụ thuộc thang đo (scale-dependent). Nồng độ $PM_{2.5}$ Sa Đéc trung bình $\approx 10\,\mu g/m^3$, sai số $MAE=3.5$ ở đây có ý nghĩa hoàn toàn khác với $MAE=3.5$ tại nơi có nồng độ $150\,\mu g/m^3$.
2. MASE (Hyndman & Koehler, 2006) chuẩn hóa sai số mô hình bằng sai số của mô hình cơ sở ngây thơ In-sample Naive Persistence ($MAE_{\text{naive}} = 1.821\,\mu g/m^3$).
3. $MASE < 1.0$ là bằng chứng toán học duy nhất chứng minh mô hình có "kỹ năng dự báo thực chất" (True Skill).
4. $R^2$ không phù hợp với chuỗi thời gian tự tương quan cao vì dùng đường trung bình ngang làm baseline.

### Q2: Tại sao thống kê Diebold-Mariano ở Bảng 4.7 lại mang giá trị âm?
**Trả lời:**
Hàm tổn thất vi phân được định nghĩa là $d_t = |e_{\text{Proposed}}| - |e_{\text{Baseline}}|$. Khi mô hình đề xuất có sai số nhỏ hơn mô hình cơ sở, trung bình $\b\bar{d} < 0$. Do đó, thống kê $DM < 0$ và mang dấu âm lớn (ví dụ $-8.452$) cùng giá trị $p < 0.001$ chính là minh chứng khẳng định mô hình đề xuất vượt trội có ý nghĩa thống kê so với baseline.

### Q3: Cơ sở nào tác giả phân chia 80:10:10 mà không dùng K-Fold Cross Validation?
**Trả lời:**
K-Fold ngẫu nhiên xáo trộn thời gian, đưa dữ liệu tương lai vào huấn luyện quá khứ, vi phạm nguyên lý nhân quả (causality). Đề án dùng Anchor Split tuyến tính thời gian 80:10:10 (Train 30 tháng, Val 3 tháng để tinh chỉnh tham số, Test 4.5 tháng để đánh giá mù độc lập).

### Q4: Kiểm định ADF dừng nhưng KPSS không dừng có mâu thuẫn không?
**Trả lời:**
Không hề mâu thuẫn. Đây là trường hợp kinh điển trong kinh tế lượng chỉ ra rằng chuỗi có tính chất **Dừng cục bộ & Biến thiên theo mùa (Locally Stationary with Strong Seasonality)**. Chuỗi không bị trôi dạt vô hạn (no unit root), nhưng phương sai thay đổi tuần hoàn theo chu kỳ ngày đêm và mùa vụ.

### Q5: Conformal Prediction khác gì so với khoảng tin cậy Gauss truyền thống?
**Trả lời:**
Khoảng tin cậy Gauss $\hat{y} \pm 1.96 \hat{\sigma}$ bắt buộc giả định phần dư tuân theo phân phối chuẩn. Dữ liệu $PM_{2.5}$ có độ lệch Skewness $2.0046$ và Kurtosis $6.1458$ (đuôi rất nặng). Conformal Prediction (CQR) là phương pháp không phụ thuộc phân phối (Distribution-free), đảm bảo độ phủ toán học chính xác 90% ngay cả khi dữ liệu bị lệch cực đoan.

### Q6: Tại sao ACI thích ứng lại khôi phục được độ phủ 90% khi có Concept Drift?
**Trả lời:**
CQR tĩnh giả định dữ liệu có tính chất trao đổi được (Exchangeability). Khi thời tiết giao mùa (Concept Drift), giả định này bị vi phạm khiến độ phủ tụt xuống 76%. ACI (Gibbs & Candès, 2021) bổ sung cơ chế phản hồi động: khi bước trước bị trượt khoảng dự báo, ngưỡng tin cậy sẽ tự động nới rộng ra ở bước tiếp theo với tốc độ $\gamma=0.01$, khôi phục độ phủ thực nghiệm đạt $90.2\%$.

### Q7: Mẫu số chuẩn hóa MASE có bị thay đổi giữa các mô hình không?
**Trả lời:**
Tuyệt đối không. Đề án áp dụng mẫu số chuẩn hóa đồng nhất $MAE_{\text{Persistence\_1h}} = 1.821\,\mu g/m^3$ cho toàn bộ 11 mô hình và tất cả các chân trời dự báo, bảo đảm tính công bằng tuyệt đối.

---

## 🛠️ Nhóm 2: Kỹ Nghệ Dữ Liệu & Xử Lý Bất Thường (Phản Biện 1)

### Q8: Tại sao dũng cảm loại bỏ 19.810 giờ khuyết dài thay vì dùng GAN hay MICE?
**Trả lời:**
19.810 giờ khuyết thiếu tương đương hơn 2 năm tích lũy ngắt quãng do trạm bảo trì. Dữ liệu này mang bản chất MNAR (Missing Not At Random). Việc dùng GAN hay MICE để tự vẽ ra 19.810 giờ là hành vi tạo ảo giác dữ liệu (Data Hallucination). Đề án thà chấp nhận số lượng mẫu ít hơn nhưng 100% sạch và thật, chia thành các đoạn liên tục độc lập để huấn luyện.

### Q9: Bẫy xóa ngoại lai bằng IQR 3.0 đã gây hậu quả gì trong thử nghiệm?
**Trả lời:**
IQR 3.0 đã gọt nhầm 66 đỉnh ô nhiễm thực tế ($> 54\,\mu g/m^3$). Khi đó RMSE tập kiểm tra giảm giả tạo, nhưng mô hình hoàn toàn mù trước các đợt ô nhiễm nặng thực tế, chỉ đạt $F_1 < 0.35$ khi cảnh báo sớm. Đề án chuyển sang Domain Bounds [0, 500] µg/m³ theo chuẩn WHO AQI, khôi phục $F_1 = 0.782$.

### Q10: Kỷ luật shift(1) đã giải quyết việc $R^2=1.000$ ảo ra sao?
**Trả lời:**
Trước khi áp dụng `shift(1)`, các biến rolling mean chứa chính giá trị $y_t$, dẫn đến việc mô hình chỉ giải phương trình đại số để lấy $y_t$ đạt $R^2=1.000$. Kỷ luật bắt buộc `.shift(1)` trước mọi hàm trượt/sai phân đã triệt tiêu hoàn toàn rò rỉ dữ liệu, đưa $R^2$ về giá trị thực $0.267$.

### Q11: Tại sao tập Test chỉ được tính điểm trên dữ liệu thực (`is_imputed == 0`)?
**Trả lời:**
Để bảo đảm tính liêm chính khoa học. Đánh giá trên dữ liệu nội suy chỉ kiểm tra xem mô hình có học được hàm Spline/KNN hay không, chứ không phản ánh khả năng dự báo thế giới thực.

### Q12: KNN Imputation có gây rò rỉ dữ liệu tương lai không?
**Trả lời:**
Không. Thuật toán KNN Imputation trong đề án tuân thủ kỷ luật nghiêm ngặt: các mẫu donor chỉ được tìm kiếm ở các mốc thời gian quá khứ ($t' < t$).

### Q13: Tại sao lại chọn khung đa độ phân giải 15m, 30m, 1h?
**Trả lời:**
15m để bắt kịp vi xung phát thải và phá bẫy tự tương quan; 30m là điểm ngọt Pareto cân bằng tín hiệu - nhiễu (chiếm 10/15 top-5 ranks); 1h để đối chuẩn với các trạm quy chuẩn quốc gia.

### Q14: Việc ép Domain Bounds [0, 500] có vi phạm tính khách quan?
**Trả lời:**
Không. Nồng độ khối lượng vật lý không thể âm ($<0$), và mức $>500\,\mu g/m^3$ là ngưỡng kịch khung nguy hại khẩn cấp của thang đo WHO AQI.

---

## 🤖 Nhóm 3: Học Máy, Học Sâu & Ứng Dụng Thực Tiễn (Phản Biện 2)

### Q15: Tại sao chuỗi giờ 1h lại mắc bẫy tự tương quan Persistence?
**Trả lời:**
Do hệ số tự tương quan $r=0.86$, nồng độ giờ này gần như bằng giờ trước. Mô hình học máy dự báo bị trễ pha 1 bước thời gian sẽ có sai số lớn hơn việc chỉ lấy lại giá trị cũ.

### Q16: Cơ chế nào giúp GRU 15m phá vỡ bẫy tự tương quan đạt MASE=0.667?
**Trả lời:**
Ở độ phân giải 15m, tầm 1h tương ứng 4 bước ($h=4$). GRU nắm bắt được đạo hàm biến thiên bậc 1 và gia tốc tích tụ hạt bụi trong 4 nhịp 15 phút trước khi đạt đỉnh, khắc phục hoàn toàn hiện tượng trễ pha.

### Q17: Tại sao Weighted Ensemble 30m lại là mô hình quán quân?
**Trả lời:**
Nó kết hợp tối ưu trọng số lồi giữa LightGBM (phân tách ngưỡng phi tuyến), GRU (quán tính chuỗi liên tục) và Random Forest (ổn định phương sai), giúp triệt tiêu phần dư lỗi chéo.

### Q18: Tại sao phân luồng Tree SHAP cho LightGBM và Permutation cho Deep Learning?
**Trả lời:**
Tree SHAP tận dụng cấu trúc cây tính toán chính xác 100% trong <100ms. Với Deep Learning, Kernel SHAP mất 16 giờ và dễ tràn RAM; Permutation Importance mang lại kết quả trực quan trong vài chục giây mà không phụ thuộc cấu trúc mô hình.

### Q19: Điểm chuyển pha khí quyển 14-17 µg/m³ có ý nghĩa gì với chính quyền Sa Đéc?
**Trả lời:**
Đây là ngưỡng bão hòa tự làm sạch của khí quyển. Khuyến nghị chính quyền phát cảnh báo sớm Mức 1 khi nồng độ chạm $14\,\mu g/m^3$ và kích hoạt điều tiết giao thông/phun sương tưới đường trước khi nồng độ vượt $17\,\mu g/m^3$.

### Q20: Hệ thống có bảo đảm khả năng suy luận thời gian thực trên Cloud hạn chế tài nguyên?
**Trả lời:**
Có. Hệ thống lưu trữ Feature Store dạng Marts tải trong vài chục mili-giây, ép luồng OpenMP $n=1$, bộ nhớ zero-dependency knowledge store tiêu tốn 0 MB RAM, vận hành mượt mà trên Render Free 512MB RAM.

---
*Liên kết liên quan: [[00_Index_MOC]] | [[01_Pipeline_7_Steps]] | [[02_Slides_Narrative_and_Key_Arguments]]*\n