"""Audit Dashboard — Quality Gate for hardcoded metrics detection.

Scans app.py, pages.py, and info_cards.py for suspicious hardcoded float
patterns that look like MASE/MAE values. These should come from
ReportingEngine, not be typed inline.

Usage:
    python scripts/audit_dashboard.py

Returns exit code 0 if clean, 1 if hardcoded metrics detected.
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Files to scan for hardcode violations
SCAN_FILES = [
    PROJECT_ROOT / "app.py",
    PROJECT_ROOT / "pages.py",
    PROJECT_ROOT / "src" / "info_cards.py",
]

# ── Patterns that indicate hardcoded MASE/MAE values ──
# These are float patterns commonly found as MASE (0.6-1.5) or MAE (2-7)
# We look for them inside strings, f-strings, and dicts
HARDCODE_PATTERNS = [
    # MASE-like values in strings: "0.987", "0.745", "0.676"
    (r'["\'](?:MASE[=:]\s*)?(\d\.\d{2,3})["\']', "MASE-like string literal"),
    # Dict values: {"TFT": [0.987, ...]}
    (r':\s*\[[\d.,\s]+\]', "Hardcoded metric array"),
    # Specific known-bad patterns from previous versions
    (r'"Ensemble_Stack".*0\.745', "Hardcoded Ensemble_Stack MASE"),
    (r'"TFT".*0\.987', "Hardcoded TFT MASE"),
    (r'"LSTM".*0\.676', "Hardcoded LSTM MASE"),
]

# ── Lines to SKIP (legitimate uses) ──
# These are false positives — code that's OK to have float values
SKIP_PATTERNS = [
    r"#\s*",  # Comments
    r"st\.caption",  # Captions (non-metric text)
    r"seasonal_diff.*0\.903",  # Experiment description (not dashboard metric)
    r"STL.*0\.(507|736)",  # Experiment description
    r"Forecastability.*0\.434",  # EDA statistic
    r"seasonal strength.*0\.343",  # Domain constant
    r"apply_plotly_style",  # Theme imports
    r"def ",  # Function definitions
    r"import ",  # Imports
    r"from ",  # Imports
    r"PALETTE_",  # Color definitions
    r"rgba\(",  # CSS rgba values
    r"line_width",  # Chart styling
    r"font.*size",  # Font sizing
    r"padding|margin|border-radius",  # CSS properties
    r"height=\d",  # Chart height
    r"\.0{1,2}\b",  # Round numbers like 1.0, 2.0
    r"MASE.*1\.0(?:\b|[^0-9])",  # MASE = 1.0 baseline reference (OK)
    r"p.*=.*1\.4e",  # p-value scientific notation
    r"σ=5\.18",  # STL residual sigma (domain constant)
    r"inflation",  # Leakage experiment description
    r"v7-exp",  # Experiment label
    r"autocorrelation.*0\.99",  # Domain constant
    r"0\.85.*6h",  # Autocorrelation description
    r"0\.45.*24h",  # Autocorrelation description
    # ── Literature comparison (static reference data) ──
    r'"Năm"',  # Year column in literature table
    r'"MAE \(µg/m³\)"',  # Literature MAE column
    r'"RMSE"',  # Literature RMSE column
    r'"R²"',  # Literature R² column
    r'"p-value"',  # DM test p-values
    r'"DM Statistic"',  # DM test statistics
    r'"Δ vs Persistence"',  # DM test delta
    r'"Kết luận"',  # DM test conclusion
    r'"Hạng"',  # Ranking column
    r'"So sánh"',  # DM comparison label
    # ── Hyperparameters (static config, not metrics) ──
    r'"h=\d+"',  # Hyperparameter horizon keys
    r'"Rolling window"',  # Hyperparameter labels
    r"hidden_dim|lr|batch|dropout|epochs",  # Hyperparameter names
]


def audit_file(filepath: Path) -> list[dict]:
    """Scan a file for hardcoded metric patterns.

    Returns list of violations: [{line_num, line, reason}]
    """
    if not filepath.exists():
        return []

    violations = []
    lines = filepath.read_text(encoding="utf-8").splitlines()

    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            continue

        # Skip legitimate patterns
        if any(re.search(skip, stripped, re.IGNORECASE) for skip in SKIP_PATTERNS):
            continue

        # Check for hardcode violations
        for pattern, reason in HARDCODE_PATTERNS:
            if re.search(pattern, stripped):
                violations.append({
                    "file": filepath.name,
                    "line_num": i,
                    "line": stripped[:120],
                    "reason": reason,
                })
                break  # One violation per line

    return violations


def main():
    print("=" * 60)
    print("Dashboard Audit — Hardcode Detection")
    print("=" * 60)

    all_violations = []
    for fpath in SCAN_FILES:
        violations = audit_file(fpath)
        all_violations.extend(violations)
        status = "✅ CLEAN" if not violations else f"⚠️  {len(violations)} violations"
        print(f"  {fpath.name:30s} {status}")

    if all_violations:
        print(f"\n{'─' * 60}")
        print(f"❌ Found {len(all_violations)} hardcoded metric(s):\n")
        for v in all_violations:
            print(f"  {v['file']}:{v['line_num']} [{v['reason']}]")
            print(f"    → {v['line']}")
            print()
        print("Fix: Replace hardcoded values with ReportingEngine calls.")
        sys.exit(1)
    else:
        print(f"\n✅ All {len(SCAN_FILES)} files are clean — no hardcoded metrics detected!")
        sys.exit(0)


if __name__ == "__main__":
    main()
