import json
from pathlib import Path

file_path = Path("/Users/trihx/Desktop/time-series-forecasting/research/experiments/dashboard_content.json")
with open(file_path, "r", encoding="utf-8") as f:
    content = json.load(f)

content["global"]["info_cards"]["shap_guide"] = "SHAP giải thích **tại sao** model đưa ra dự đoán cụ thể:\n\n- **Bar chart**: Top features theo SHAP mean absolute value\n- **Beeswarm**: Impact distribution (đỏ = giá trị cao, xanh = thấp)\n- **GRU Permutation**: Feature importance cho Deep Learning\n- **Dependence**: Quan hệ phi tuyến feature → SHAP value\n\n**Phương pháp:**\n- LightGBM: TreeSHAP (exact, O(TLD))\n- GRU: Permutation Importance (100 shuffles)\n- Cả hai phương pháp đều trên test set only"

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(content, f, indent=2, ensure_ascii=False)
print("Updated dashboard_content.json with correct shap_guide")
