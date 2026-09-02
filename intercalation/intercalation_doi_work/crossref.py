import json, re, time, difflib, urllib.parse, urllib.request

data = json.load(open('intercalation_doi_work/parsed.json', encoding='utf-8'))

def norm(s):
    s = s.lower()
    s = re.sub(r'[^a-z0-9\u00c0-\u024f]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def title_sim(a, b):
    a, b = norm(a), norm(b)
    if not a or not b: return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()

def journal_match(cross_journal, our_journal):
    if not cross_journal or not our_journal: return 0.0
    cj = norm(cross_journal); oj = norm(our_journal)
    # check if our abbrev appears in cross container or cross short container
    if oj in cj or cj in oj: return 1.0
    # token overlap
    ct = set(cj.split()); ot = set(oj.split())
    if not ot: return 0.0
    return len(ct & ot) / len(ot)

def lookup(d):
    q = ' '.join(x for x in [d['title'], d['journal'], d['year']] if x)
    url = 'https://api.crossref.org/works?rows=4&query.bibliographic=' + urllib.parse.quote(q) + '&mailto=research@example.org'
    req = urllib.request.Request(url, headers={'User-Agent':'ref-doi-finder/1.0 (mailto:research@example.org)'})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            j = json.load(r)
    except Exception as e:
        return {'error': str(e)}
    items = j.get('message', {}).get('items', [])
    cands = []
    for it in items:
        ttl = (it.get('title') or [''])[0]
        year = None
        for k in ('published-print','published-online','issued','published'):
            if it.get(k) and it[k].get('date-parts') and it[k]['date-parts'][0]:
                year = it[k]['date-parts'][0][0]; break
        cj = (it.get('container-title') or [''])[0]
        cjs = (it.get('short-container-title') or [''])[0] if it.get('short-container-title') else ''
        vol = it.get('volume',''); page = it.get('page','') or it.get('article-number','')
        doi = it.get('DOI','')
        ts = title_sim(ttl, d['title'])
        ym = 1.0 if (d['year'] and year and str(year)==str(d['year'])) else (0.5 if d['year'] and year and abs(int(year)-int(d['year']))<=1 else 0.0)
        jm = max(journal_match(cj, d['journal']), journal_match(cjs, d['journal']))
        score = 0.6*ts + 0.25*ym + 0.15*jm
        cands.append({'doi':doi,'title':ttl,'container':cj,'short':cjs,'year':year,'vol':vol,'page':page,'ts':round(ts,3),'ym':ym,'jm':round(jm,2),'score':round(score,3)})
    cands.sort(key=lambda x:-x['score'])
    return {'query':q,'candidates':cands[:3]}

results = []
for i, d in enumerate(data):
    r = lookup(d)
    r['ref'] = d['ref']; r['our_title'] = d['title']; r['our_journal'] = d['journal']; r['our_year'] = d['year']
    results.append(r)
    time.sleep(0.35)
    if (i+1) % 25 == 0:
        print('progress', i+1, '/', len(data))

json.dump(results, open('intercalation_doi_work/crossref_raw.json','w',encoding='utf-8'), ensure_ascii=False, indent=1)

# summary
def best(r):
    c = r.get('candidates', [])
    return c[0] if c else None
hi = [r for r in results if best(r) and best(r)['score']>=0.7]
lo = [r for r in results if not best(r) or best(r)['score']<0.7]
print('high confidence (>=0.7):', len(hi))
print('low confidence / none:', len(lo))
for r in lo:
    b = best(r)
    print('---', r['ref'], '|', r['our_title'][:60])
    if b: print('     best:', b['doi'], b['score'], '|', b['title'][:60])
    else: print('     no candidates')
