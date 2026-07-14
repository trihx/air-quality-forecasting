#!/bin/bash
# Run CQR training experiment using uv
# Redirects output to log file

set -e

cd /Users/trihx/Desktop/time-series-forecasting

echo "🚀 Starting CQR Training..."
echo "Horizons: 1h, 6h, 24h"
echo "Method: Conformalized Quantile Regression (Romano et al., 2019)"
echo ""

uv run python research/scripts/train_gru_cqr.py 2>&1 | tee research/logs/cqr_training.log

echo ""
echo "✅ Done! Results saved to research/logs/cqr_training.log"
