import json

path = 'research/experiments/dashboard_runs/v9_multi_resolution.json'
d = json.load(open(path, encoding='utf-8'))
for h in ['1h', '6h', '24h']:
    for m_name, m in d['metrics'][h].items():
        rmse = m.get('rmse')
        if rmse is not None and not isinstance(rmse, (int, float)):
            print(f'Dashboard JSON {h} {m_name} rmse type: {type(rmse)} value: {repr(rmse)}')
        r2 = m.get('r2')
        if r2 is not None and not isinstance(r2, (int, float)):
            print(f'Dashboard JSON {h} {m_name} r2 type: {type(r2)} value: {repr(r2)}')
        da = m.get('da')
        if da is not None and not isinstance(da, (int, float)):
            print(f'Dashboard JSON {h} {m_name} da type: {type(da)} value: {repr(da)}')
print('Done checking dashboard json')
