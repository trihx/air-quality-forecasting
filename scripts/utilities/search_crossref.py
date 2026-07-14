import urllib.request, json, urllib.parse

def search(query):
    url = 'https://api.crossref.org/works?query=' + urllib.parse.quote(query) + '&select=DOI,title,abstract,author,volume,issue,page,URL,published-print,published-online&rows=2'
    try:
        req = urllib.request.urlopen(url)
        data = json.loads(req.read())['message']['items']
        for item in data:
            print(f'Title: {item.get("title", [""])[0]}')
            print(f'DOI: {item.get("DOI", "")}')
            print(f'Abstract: {item.get("abstract", "No abstract")[:300]}...\n')
    except Exception as e:
        print(f'Error: {e}')

print('--- Mutual Information ---')
search('Estimating mutual information Kraskov')

print('--- Temperature Inversion PM2.5 ---')
search('temperature inversion PM2.5')

print('--- PM2.5 Weekend Effect ---')
search('PM2.5 weekend effect urban')
