#!/bin/bash
# ══════════════════════════════════════════════════════════════════
# CQR Upgrade — Complete Pipeline
# ══════════════════════════════════════════════════════════════════
# Chạy toàn bộ quy trình CQR upgrade trong 1 lệnh:
#   1. Tạo snapshot v7 (lưu bản cũ)
#   2. Train GRU Quantile + CQR calibration (h=1, 6, 24)
#   3. Tạo snapshot v8 (kết quả mới)
#
# Usage:
#   cd /Users/trihx/Desktop/time-series-forecasting
#   bash research/scripts/run_cqr_upgrade.sh
#
# Thời gian ước tính: ~10-15 phút (MPS GPU)
# ══════════════════════════════════════════════════════════════════

set -e
cd /Users/trihx/Desktop/time-series-forecasting

echo "══════════════════════════════════════════════════════════"
echo "🔄 CQR Upgrade Pipeline — Conformalized Quantile Regression"
echo "══════════════════════════════════════════════════════════"
echo ""

# ── Step 1: Snapshot v7 (preserve pre-CQR state) ──
echo "📸 Step 1/3: Creating v7 snapshot (pre-CQR backup)..."
uv run python research/scripts/create_snapshot_v7.py
echo ""

# ── Step 2: Train GRU Quantile + CQR ──
echo "🧪 Step 2/3: Training GRU Quantile + CQR calibration..."
echo "   Horizons: 1h, 6h, 24h"
echo "   Method: Pinball Loss (q=0.05, 0.50, 0.95) + Conformal"
echo "   Reference: Romano et al. (2019), NeurIPS"
echo ""
uv run python research/scripts/train_gru_cqr.py 2>&1 | tee research/logs/cqr_training.log
echo ""

# ── Step 3: Snapshot v8 (CQR results) ──
echo "📸 Step 3/3: Creating v8 snapshot (CQR results)..."
uv run python research/scripts/create_snapshot_v8.py
echo ""

echo "══════════════════════════════════════════════════════════"
echo "✅ CQR Upgrade Complete!"
echo "══════════════════════════════════════════════════════════"
echo ""
echo "📂 Outputs:"
echo "   - Models:    models/exported/gru_quantile_*h.pt"
echo "   - Configs:   models/exported/gru_quantile_*h_config.json"
echo "   - Results:   research/experiments/prediction_intervals/"
echo "   - Snapshots: research/experiments/dashboard_runs/v7, v8"
echo "   - Log:       research/logs/cqr_training.log"
echo ""
echo "🔄 Restart dashboard để thấy kết quả mới:"
echo "   lsof -ti :8501 | xargs kill -9 && uv run streamlit run app.py"
