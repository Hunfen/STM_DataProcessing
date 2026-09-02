import json, time, urllib.request, urllib.error

final = json.load(open('intercalation_doi_work/final_doi.json', encoding='utf-8'))
# unique DOIs to fetch (skip no-DOI and duplicates)
todos = []
seen = set()
for e in final:
    if not e['doi']: 
        continue
    if e['doi'].lower() in seen:
        continue
    seen.add(e['doi'].lower())
    todos.append(e)

meta = {}
def get_doi(doi):
    url = 'https://api.crossref.org/works/' + urllib.parse.quote(doi) + '?mailto=research@example.org'
    req = urllib.request.Request(url, headers={'User-Agent':'ref-doi-finder/1.0 (mailto:research@example.org)'})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)['message']
        except urllib.error.HTTPError as e:
            if e.code in (404,): return None
            if e.code == 429:
                time.sleep(3*(attempt+1)); continue
            return None
        except Exception:
            return None
    return None

for i, e in enumerate(todos):
    m = get_doi(e['doi'])
    meta[e['doi']] = m
    time.sleep(0.3)
    if (i+1) % 25 == 0:
        print('progress', i+1, '/', len(todos))

json.dump(meta, open('intercalation_doi_work/crossref_meta.json','w',encoding='utf-8'), ensure_ascii=False)
print('fetched', len(meta), 'metadata records')
missing = [e['doi'] for e in todos if not meta.get(e['doi'])]
print('missing/empty:', len(missing))
for x in missing: print('  ', x)
