import json, re, sys

lines = open('intercalation_reference.md', encoding='utf-8').read().splitlines()
# section three: lines 2159..2292 (1-based). Collect entries starting with "- ["
entries = []
for i, ln in enumerate(lines, 1):
    if 2159 <= i <= 2292 and ln.startswith('- ['):
        entries.append(ln)

def parse(ln):
    # strip "- [NNN] "
    m = re.match(r'^- \[(\d+)\]\s+(.*)$', ln)
    refnum = m.group(1)
    body = m.group(2)
    # find title in double quotes
    q = re.search(r'"([^"]*)"', body)
    title = q.group(1) if q else ''
    if q:
        authors = body[:q.start()].strip().rstrip(',')
        rest = body[q.end():].strip().lstrip(', ')
    else:
        authors = body.strip()
        rest = ''
    return refnum, authors, title, rest

def parse_rest(rest):
    # returns dict journal, volume, page, year
    d = {'journal':'','volume':'','page':'','year':'','raw':rest}
    if not rest:
        return d
    # year at end: (YYYY) possibly with trailing periods
    ym = re.search(r'\((\d{4})\)', rest)
    if ym:
        d['year'] = ym.group(1)
    # journal + volume, page pattern:  "... JOURNAL VOL, PAGE (YEAR)."
    # split rest at last comma before (year) to get page
    # remove trailing period
    r = rest.rstrip('.')
    # try to find "JOURNAL VOL, PAGE" where VOL and PAGE numeric-ish
    # Generic: match  'JOURNAL  VOL,  PAGE'
    m = re.match(r'^(.*?)\s+([A-Za-z0-9]?[\d\-–]{1,8})\s*,\s*(.+?)\s*(?:\((\d{4})\))?$', r)
    # simpler: split on ',' -> last element is page+year
    parts = r.split(',')
    if len(parts) >= 2:
        pageyear = parts[-1].strip()
        d['page'] = pageyear
        # journal+volume = everything before last comma
        jv = ','.join(parts[:-1]).strip()
        # volume = trailing token of jv
        jv2 = jv.rsplit(' ', 1)
        if len(jv2) == 2 and re.search(r'\d', jv2[1]):
            d['journal'] = jv2[0].strip()
            d['volume'] = jv2[1].strip()
        else:
            d['journal'] = jv
    return d

out = []
for ln in entries:
    refnum, authors, title, rest = parse(ln)
    d = parse_rest(rest)
    d.update({'ref': refnum, 'authors': authors, 'title': title})
    out.append(d)

json.dump(out, open('intercalation_doi_work/parsed.json','w',encoding='utf-8'), ensure_ascii=False, indent=1)
print('parsed', len(out), 'entries')
for d in out[:6]:
    print(d)
