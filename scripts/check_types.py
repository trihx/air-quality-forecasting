from src.snapshot_adapter import load_all_normalized
data = load_all_normalized().get('v9_multi_resolution', {})
for h in ['1h', '6h', '24h']:
    for model, m in data['results'][h].items():
        rmse = m.get('rmse')
        if rmse is not None and not isinstance(rmse, (int, float)):
            print(f'{h} {model} rmse type: {type(rmse)} value: {rmse}')
        r2 = m.get('r2')
        if r2 is not None and not isinstance(r2, (int, float)):
            print(f'{h} {model} r2 type: {type(r2)} value: {r2}')
        da = m.get('da')
        if da is not None and not isinstance(da, (int, float)):
            print(f'{h} {model} da type: {type(da)} value: {da}')
print('Done checking types')
