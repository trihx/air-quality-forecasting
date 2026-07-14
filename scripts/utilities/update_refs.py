import re
import codecs

groups = [
    # Phase 1: Domain & Preprocessing (1-15)
    'who2021', 'barkjohn2021', 'zhang2017', 'zannetti1990', 'blanchard2003',
    'shumway2017', 'boxcox1964', 'rosner1983', 'cleveland1990', 'cleveland1993', 
    'dickey1979', 'kwiatkowski1992', 'ljung1978', 'moritz2015', 'troyanskaya2001',
    # Phase 2: Evaluation (16-23)
    'peixeiro2022', 'hyndman2021', 'tashman2000', 'makridakis2020',
    'hyndman2006', 'willmott2005', 'gneiting2007', 'diebold1995',
    # Phase 3: Modeling (24-37)
    'box2015', 'cho2014', 'hochreiter1997', 'lim2021', 'ke2017', 'breiman2001',
    'dietterich2000', 'wolpert1992', 'christ2018', 'akiba2019', 'joseph2022', 
    'huang2022', 'vishwas2020', 'kang2017',
    # Phase 4: Uncertainty & XAI (38-45)
    'lundberg2017', 'fisher2019', 'romano2019', 'gibbs2021', 'gal2016', 
    'lakshminarayanan2017', 'gu2021', 'houdou2024',
    # Phase 5: Benchmark (46-53)
    'shetty2024', 'tian2024', 'inam2024', 'kim2023', 'patel2025', 'kaveh2025', 
    'nguyen2024', 'rakholia2022'
]

mapping = {k: i+1 for i, k in enumerate(groups)}

with codecs.open('src/frontend/citations.py', 'r', 'utf-8') as f:
    content = f.read()

for k, new_id in mapping.items():
    pattern = r'("' + k + r'"\s*:\s*\{[^}]*?"id"\s*:\s*)(\d+)'
    content = re.sub(pattern, r'\g<1>' + str(new_id), content, count=1, flags=re.DOTALL)

with codecs.open('src/frontend/citations.py', 'w', 'utf-8') as f:
    f.write(content)

print("Updated citations.py successfully")
