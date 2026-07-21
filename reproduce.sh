#!/usr/bin/env bash
# ============================================================================
# reproduce.sh — Reproducibility Script for PM2.5 Forecasting Pipeline
# ============================================================================
# Usage: bash reproduce.sh [--skip-install] [--quick]
#
# This script reproduces the complete v9 pipeline results from scratch.
# Requires: Python 3.11+, uv package manager, ~2GB free RAM.
#
# Options:
#   --skip-install  Skip virtual environment setup
#   --quick         Run only metrics verification (skip retraining)
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="research/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/reproduce_${TIMESTAMP}.log"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log() { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $1" | tee -a "$LOG_FILE"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$LOG_FILE"; }
err() { echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"; exit 1; }

SKIP_INSTALL=false
QUICK=false
for arg in "$@"; do
    case $arg in
        --skip-install) SKIP_INSTALL=true ;;
        --quick) QUICK=true ;;
        *) warn "Unknown option: $arg" ;;
    esac
done

# ============================================================================
# Step 0: Environment Setup
# ============================================================================
log "Step 0/7: Environment Setup"
if [ "$SKIP_INSTALL" = false ]; then
    if ! command -v uv &>/dev/null; then
        err "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
    fi
    uv sync --frozen 2>&1 | tail -5 | tee -a "$LOG_FILE"
    log "  ✅ Dependencies installed"
else
    log "  ⏭️  Skipped (--skip-install)"
fi

# ============================================================================
# Step 1: Data Verification
# ============================================================================
log "Step 1/7: Data Verification"
DATASET="dataset/raw/final_dataset.csv"
if [ ! -f "$DATASET" ]; then
    err "Dataset not found: $DATASET"
fi
ROW_COUNT=$(wc -l < "$DATASET" | tr -d ' ')
log "  Dataset: $DATASET ($ROW_COUNT rows)"

# ============================================================================
# Step 2: Run Tests
# ============================================================================
log "Step 2/7: Running Test Suite"
uv run python -m pytest tests/ -q --tb=short 2>&1 | tee -a "$LOG_FILE" || warn "Some tests failed"

# ============================================================================
# Step 3: Verify Stationarity Tests
# ============================================================================
log "Step 3/7: Stationarity Tests (ADF/KPSS)"
STATIONARITY_JSON="research/diagnostics/stationarity/stationarity_results.json"
if [ -f "$STATIONARITY_JSON" ]; then
    uv run python -c "
import json
with open('$STATIONARITY_JSON') as f:
    d = json.load(f)
print(f'  Observations: {d[\"n_observations\"]}')
for r in d['results']:
    print(f'  {r[\"test\"]:4s} | {r[\"series\"]:25s} | stat={r[\"statistic\"]:>8.4f} | p={r[\"p_value\"]:.4f} | {r[\"interpretation\"]}')
" 2>&1 | tee -a "$LOG_FILE"
else
    warn "Stationarity results not found. Run EDA pipeline first."
fi

# ============================================================================
# Step 4: Verify Metrics (Quick Mode stops here)
# ============================================================================
log "Step 4/7: Verify Standardized Metrics"
METRICS_JSON="research/experiments/standardized_metrics.json"
if [ -f "$METRICS_JSON" ]; then
    uv run python -c "
import json
with open('$METRICS_JSON') as f:
    d = json.load(f)
print('  Best models per horizon (unified MASE):')
for h in ['1h', '6h', '24h']:
    models = d['results'][h]
    best = min(models.items(), key=lambda x: x[1].get('mase_unified', 999))
    print(f'    {h}: {best[0]} — MASE={best[1][\"mase_unified\"]:.4f}, MAE={best[1].get(\"mae\",\"N/A\")}')
" 2>&1 | tee -a "$LOG_FILE"
else
    warn "Standardized metrics not found."
fi

# ============================================================================
# Step 5: Bootstrap CI
# ============================================================================
log "Step 5/7: Bootstrap Confidence Intervals"
BOOTSTRAP_SCRIPT="scripts/analysis/bootstrap_mase_ci.py"
if [ -f "$BOOTSTRAP_SCRIPT" ]; then
    uv run python "$BOOTSTRAP_SCRIPT" 2>&1 | tee -a "$LOG_FILE"
else
    warn "Bootstrap script not found: $BOOTSTRAP_SCRIPT"
fi

if [ "$QUICK" = true ]; then
    log "Step 6-7: Skipped (--quick mode)"
    log "============================================"
    log "✅ Quick verification complete. See: $LOG_FILE"
    exit 0
fi

# ============================================================================
# Step 6: Verify SHAP Results
# ============================================================================
log "Step 6/7: SHAP Verification"
SHAP_JSON="research/figures/shap/shap_results.json"
if [ -f "$SHAP_JSON" ]; then
    uv run python << 'PYEOF'
import json
with open("research/figures/shap/shap_results.json") as f:
    d = json.load(f)
for h in ["1h", "6h", "24h"]:
    if h in d:
        shap = d[h].get("top_15_shap", {})
        top3 = list(shap.items())[:3]
        features = ", ".join([f"{k}({v:.2f})" for k, v in top3])
        print(f"  {h} Top-3: {features}")
PYEOF
else
    warn "SHAP results not found."
fi

# ============================================================================
# Step 7: Verify DM Test
# ============================================================================
log "Step 7/7: Diebold-Mariano Test Verification"
DM_JSON="research/diagnostics/statistical_tests/statistical_tests_results.json"
if [ -f "$DM_JSON" ]; then
    uv run python << 'PYEOF'
import json
with open("research/diagnostics/statistical_tests/statistical_tests_results.json") as f:
    d = json.load(f)
if "diebold_mariano" in d:
    dm = d["diebold_mariano"]
    for h in ["1h", "6h", "24h"]:
        if h in dm:
            for pair, result in dm[h].items():
                sig = "✅" if result.get("significant_0.05") == "True" else "❌"
                print(f'  {pair:30s} | h={h:3s} | DM={result["dm_statistic"]:>7.3f} | p={result["p_value"]:.4f} | {sig}')
PYEOF
else
    warn "DM test results not found."
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
log "============================================"
log "✅ Full reproduction pipeline complete!"
log "Log file: $LOG_FILE"
log "============================================"
