import urllib.request, json
doist = [
    '10.1016/j.chemosphere.2022.136180',
    '10.4209/aaqr.220355',
    '10.1016/j.atmosenv.2023.119852',
    '10.1007/978-981-99-6547-2',
    '10.3390/s24051523',
    '10.1016/j.scitotenv.2024.170245',
    '10.1016/j.envres.2024.120363',
    '10.3390/app14125062',
    '10.1016/j.envpol.2024.125630',
    '10.1007/s40808-025-02214-5',
    '10.3846/jeelm.2024.22361',
    '10.52939/ijg.v19i12.2975',
    '10.4209/aaqr.230155',
    '10.3390/atmos13111822'
]

titles = []
for doi in doist:
    try:
        url = 'https://api.crossref.org/works/' + doi
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        res = urllib.request.urlopen(req).read()
        data = json.loads(res)
        title = data['message']['title'][0]
        titles.append(title)
    except Exception as e:
        titles.append(f"Error for {doi}: {e}")

for d, t in zip(doist, titles):
    print(f"{d}: {t}")
