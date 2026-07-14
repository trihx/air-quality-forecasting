import urllib.request
import json

dois = ['10.1016/j.chemosphere.2022.136180', '10.1016/j.ecoinf.2023.102067', '10.1016/j.jclepro.2023.139278', '10.17977/um018v5i12022p53-66', '10.52939/ijg.v19i12.2975', '10.1016/j.atmosenv.2023.120161']
for doi in dois:
    try:
        req = urllib.request.urlopen(f'https://api.unpaywall.org/v2/{doi}?email=test@example.com')
        data = json.loads(req.read())
        best_oa = data.get('best_oa_location')
        if best_oa and best_oa.get('url_for_pdf'):
            url = best_oa['url_for_pdf']
            print(f'{doi} -> {url}')
        else:
            print(f'{doi} -> NOT OA or NO PDF')
    except Exception as e:
        print(f'{doi} -> Error: {e}')
