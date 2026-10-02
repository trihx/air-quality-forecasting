---
title: "Quy trình tinh chỉnh siêu tham số bằng Optuna TPE trên tập Validation"
tags: [models, hyperparameter-tuning, optuna, tpe, validation, optimization]
aliases: ["Tinh chỉnh Optuna", "Optuna TPE", "Hyperparameter Tuning"]
domain: models
created: "2026-10-02"
status: completed
---

# ⚙️ Quy Trình Tinh Chỉnh Siêu Tham Số Bằng Optuna TPE Trên Tập Validation

## 1. Phương Pháp Luận Tinh Chỉnh (Methodology)
Để đảm bảo tính khách quan và khai thác tối đa tiềm năng của từng kiến trúc, đề án áp dụng thuật toán **Tree-structured Parzen Estimator (TPE)** thông qua thư viện Optuna:
- **Số lượng thử nghiệm (Trials):** 50 trials độc lập cho mỗi mô hình tham số.
- **Tập dữ liệu tối ưu:** Chỉ thực hiện trên **Tập Xác Thực (Validation Set 10%)** từ 10/2024 đến 12/2024. Cấm tuyệt đối tối ưu trên tập Test.
- **Hàm mục tiêu (Loss Function):** Tối thiểu hóa Mean Absolute Error ($MAE$) trên tập Validation, bảo đảm sự nhất quán tuyệt đối với hàm mục tiêu đánh giá MASE.

---

## 2. Không Gian Tìm Kiếm & Cấu Hình Tối Ưu Từng Mô Hình

### LightGBM Regressor:
- `learning_rate`: Log-uniform $[0.01, 0.15] \implies$ **Tối ưu: $0.042$**
- `num_leaves`: Int $[15, 63] \implies$ **Tối ưu: $31$**
- `max_depth`: Int $[3, 9] \implies$ **Tối ưu: $6$**
- `feature_fraction`: Uniform $[0.6, 0.95] \implies$ **Tối ưu: $0.78$**
- `objective`: Ép cứng `"regression_l1"` (MAE Loss).

### Mạng Nơ-ron GRU:
- `hidden_dim`: Choice $[32, 64, 128] \implies$ **Tối ưu: $64$**
- `num_layers`: Int $[1, 3] \implies$ **Tối ưu: $2$**
- `dropout`: Uniform $[0.1, 0.35] \implies$ **Tối ưu: $0.20$**
- `learning_rate`: Log-uniform $[1e-4, 5e-3] \implies$ **Tối ưu: $0.0012$**
- `batch_size`: $64$ với thuật toán AdamW và Cosine Annealing scheduler.

### ElasticNet:
- `alpha`: Log-uniform $[1e-4, 10.0] \implies$ **Tối ưu: $0.085$**
- `l1_ratio`: Uniform $[0.0, 1.0] \implies$ **Tối ưu: $0.45$**

---

## 3. Cơ Chế Dừng Sớm (Early Stopping)
Để ngăn chặn tình trạng học vẹt (overfitting) trên tập Validation:
- Thiết lập cơ chế **Early Stopping** với `patience = 10 epochs` (đối với GRU/LSTM) và `early_stopping_rounds = 30` (đối với LightGBM).
- Trọng số mô hình được lưu tại epoch có Validation MAE thấp nhất, đảm bảo khả năng tổng quát hóa cao nhất khi đưa sang tập Anchor Test.

---
*Liên kết liên quan: [[02_Temporal_Split_and_Test_on_Real_Only]] | [[01_Benchmark_Models_Overview]] | [[03_Sweet_Spot_30m_Ensemble]]*\n