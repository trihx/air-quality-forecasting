import os
import requests
from bs4 import BeautifulSoup
import re

dois = {
    '39_Zhang2022': '10.1016/j.chemosphere.2022.136180',
    '42_Gokul2023': '10.1016/j.ecoinf.2023.102067',
    '44_Shakya2023': '10.1016/j.jclepro.2023.139278',
    '45_Pranolo2022': '10.17977/um018v5i12022p53-66',
    '50_Hai2023': '10.52939/ijg.v19i12.2975',
    '52_Tran2023': '10.1016/j.atmosenv.2023.120161'
}

base_url = 'https://sci-hub.st/'
out_dir = 'papers_downloaded'
os.makedirs(out_dir, exist_ok=True)

for name, doi in dois.items():
    try:
        print(f"Trying to download {name} ({doi})")
        resp = requests.get(base_url + doi, headers={'User-Agent': 'Mozilla/5.0'})
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, 'html.parser')
            iframe = soup.find('iframe', id='pdf')
            if iframe and iframe.get('src'):
                pdf_url = iframe['src']
                if pdf_url.startswith('//'):
                    pdf_url = 'https:' + pdf_url
                elif pdf_url.startswith('/'):
                    pdf_url = base_url.rstrip('/') + pdf_url
                
                print(f"  -> Found PDF URL: {pdf_url}")
                pdf_resp = requests.get(pdf_url, headers={'User-Agent': 'Mozilla/5.0'})
                if pdf_resp.status_code == 200:
                    with open(os.path.join(out_dir, f'{name}.pdf'), 'wb') as f:
                        f.write(pdf_resp.content)
                    print(f"  -> Saved {name}.pdf")
                else:
                    print(f"  -> Failed to download PDF (Status: {pdf_resp.status_code})")
            else:
                print("  -> Could not find PDF iframe on Sci-Hub page.")
        else:
            print(f"  -> Failed to access Sci-Hub (Status: {resp.status_code})")
    except Exception as e:
        print(f"  -> Error: {e}")
