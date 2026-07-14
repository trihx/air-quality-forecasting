import json
import glob
import os

for f in glob.glob('d:/01-Repos/time-series-forecasting/research/predictions/*30m*.json'):
    try:
        with open(f, encoding='utf-8') as file:
            d = json.load(file)
            print(f'{os.path.basename(f)}:')
            for k, v in d.items():
                if isinstance(v, list):
                    print(f'  {k}: {len(v)}')
    except Exception as e:
        print(f'{f}: {e}')
