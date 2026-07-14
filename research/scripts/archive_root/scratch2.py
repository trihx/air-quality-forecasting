import json

with open('research/experiments/dashboard_runs/v7_cqr_20260428.json', encoding="utf-8") as f:
    data = json.load(f)

for h in ['1h', '6h', '24h']:
    if h in data.get('results', {}):
        metrics = data['results'][h]
        print(f"\nHorizon {h}:")
        for model, m_data in metrics.items():
            if model in ['Persistence', 'TFT']:
                mase = m_data.get('mase_unified', m_data.get('mase_original', m_data.get('mase')))
                mase_orig = m_data.get('mase_original')
                print(f"  {model}: MAE={m_data['mae']}, MASE_UNIFIED={mase}, MASE_ORIG={mase_orig}")
