import json, re, time, difflib, urllib.parse, urllib.request

data = {d['ref']: d for d in json.load(open('intercalation_doi_work/parsed.json', encoding='utf-8'))}
raw = json.load(open('intercalation_doi_work/crossref_raw.json', encoding='utf-8'))

def norm(s):
    s = s.lower(); s = re.sub(r'[^a-z0-9\u00c0-\u024f]+',' ',s); return re.sub(r'\s+',' ',s).strip()

def tsim(a,b):
    a,b=norm(a),norm(b)
    return difflib.SequenceMatcher(None,a,b).ratio() if a and b else 0.0

def get(q):
    url='https://api.crossref.org/works?rows=6&query.bibliographic='+urllib.parse.quote(q)+'&mailto=research@example.org'
    req=urllib.request.Request(url, headers={'User-Agent':'ref-doi-finder/1.0 (mailto:research@example.org)'})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r).get('message',{}).get('items',[])
        except urllib.error.HTTPError as e:
            if e.code==429:
                time.sleep(3*(attempt+1)); continue
            return []
        except Exception:
            return []
    return []

def score(it, d):
    ttl=(it.get('title') or [''])[0]
    year=None
    for k in ('published-print','published-online','issued','published'):
        if it.get(k) and it[k].get('date-parts') and it[k]['date-parts'][0]:
            year=it[k]['date-parts'][0][0]; break
    cj=(it.get('container-title') or [''])[0]
    ts=tsim(ttl,d['title'])
    ym=1.0 if (d['year'] and year and str(year)==str(d['year'])) else 0.0
    sc=0.7*ts+0.3*ym
    return sc, {'doi':it.get('DOI'),'title':ttl,'container':cj,'year':year,'vol':it.get('volume'),'page':it.get('page') or it.get('article-number'),'score':round(sc,3),'ts':round(ts,3),'ym':ym}

low_refs = [r['ref'] for r in raw if not r.get('candidates') or r['candidates'][0]['score']<0.7]
print('low refs:', low_refs)
results={}
for ref in low_refs:
    d=data[ref]
    fa = d['authors'].split(',')[0].strip()
    fa = re.sub(r'^([A-Z]\.?\s*)+','',fa).strip()
    cands=[]
    for q in [d['title'], d['title']+' '+fa]:
        for it in get(q):
            sc,c=score(it,d)
            cands.append(c)
        time.sleep(0.7)
    cands=sorted(cands,key=lambda x:-x['score'])
    results[ref]=cands[:4]
    print('---',ref,'|',d['title'][:55])
    for c in cands[:4]:
        print('    ',c['doi'],c['score'],'|',(c['title'] or '')[:55],'|',c['container'],c['year'])

json.dump(results, open('intercalation_doi_work/requery.json','w',encoding='utf-8'), ensure_ascii=False, indent=1)
print('done')
