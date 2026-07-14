import sys
import re
from pathlib import Path

sys.path.append('d:/01-Repos/time-series-forecasting')
from src.frontend.citations import IEEE_REFS

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find all occurrences of cite('key') or cite("key")
citations = re.findall(r'cite\([\'"]([^\'"]+)[\'"]\)', content)
missing = set()

for key in citations:
    if key not in IEEE_REFS:
        missing.add(key)

if missing:
    print('Missing citations:', missing)
else:
    print('All citations are valid!')
