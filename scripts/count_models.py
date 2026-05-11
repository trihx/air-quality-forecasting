import json

try:
    with open('/Users/trihx/Desktop/time-series-forecasting/research/experiments/dashboard_runs/v9_multi_resolution.json', encoding="utf-8") as f:
        data = json.load(f)

    models = set()
    for h in ['1h', '6h', '24h']:
        for m in data.get('metrics', {}).get(h, {}):
            models.add(m)
            
    with open('/Users/trihx/Desktop/time-series-forecasting/research/experiments/dashboard_runs/models_count.txt', 'w', encoding="utf-8") as f:
        f.write(f'Count: {len(models)}\n')
        f.write('Models:\n' + '\n'.join(sorted(models)))
except Exception as e:
    with open('/Users/trihx/Desktop/time-series-forecasting/research/experiments/dashboard_runs/models_count.txt', 'w', encoding="utf-8") as f:
        f.write(f'Error: {str(e)}')
