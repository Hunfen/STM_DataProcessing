import json, re

out = json.load(open('intercalation_doi_work/parsed.json', encoding='utf-8'))
for d in out:
    # clean title trailing comma
    d['title'] = d['title'].strip().rstrip(',').strip()
    # strip " (YYYY)" from page
    d['page'] = re.sub(r'\s*\(\d{4}\)\s*$', '', d['page']).strip()
    # strip trailing period already done in raw; keep raw
json.dump(out, open('intercalation_doi_work/parsed.json','w',encoding='utf-8'), ensure_ascii=False, indent=1)

# flag special cases: no year, no volume, no journal
special = [d for d in out if not d['year'] or not d['volume'] or not d['journal']]
print('=== special/flagged entries (no year/volume/journal) ===', len(special))
for d in special:
    print(d['ref'], '|', d['title'][:70], '| raw=', d['raw'][:90])
