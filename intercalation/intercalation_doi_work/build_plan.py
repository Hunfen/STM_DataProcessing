import json

final = json.load(open('intercalation_doi_work/final_doi.json', encoding='utf-8'))
meta = json.load(open('intercalation_doi_work/crossref_meta.json', encoding='utf-8'))

existing_coll = {
 'Ag':'J8XK3NBF','Au':'K2C35XUC','Bi':'GCLKI4JV','Ca':'A77J76WZ','Ce':'PEFLAJ3E','Co':'5XLC549M',
 'Cu':'QZIQ4LED','Dy':'PFZAQ3GR','Eu':'G6KSSEBR','Fe':'CAVWIWSY','F':'HWRXB7NS','GaN':'BVYDVP4G',
 'Gd':'M67S6HKS','Ge':'CYUJX6PT','H':'RV4XYK6H','In':'XRVP2C3Z','Li':'BCJ4F6X5','Mn':'7GGP3HLU',
 'N':'P2PTPXCI','Na':'9L9KRFW5','NbSe2':'59TAHAG4','O':'F8SJ2EX8','Pb':'TXT3UESN','Pd':'WA8MAAGB',
 'Pt':'ZMNVSY5D','Si':'PID8AW7M','Sm':'QUSFG2PC','Sn':'ZTBKP9RR','Tb':'5L5MHW8B','Yb':'3SH9SYXF',
}
new_names = ['K','Rb','Cs','Mg','Al','Sr/Ba','Ga','Er','Mo']
GRAP = 'CYS4DBUL'

def zotero_item(e):
    m = meta.get(e['doi'], {})
    t = m.get('type')
    if t in ('book','edited-book','monograph','reference-book'):
        it = {'itemType':'book'}
    else:
        it = {'itemType':'journalArticle'}
    it['title'] = (m.get('title') or [e['title']])[0]
    if it['itemType']=='journalArticle':
        ct = (m.get('container-title') or [''])
        it['publicationTitle'] = ct[0] if ct else e.get('cr_container','')
        sc = (m.get('short-container-title') or [''])
        if sc and sc[0]: it['journalAbbreviation'] = sc[0]
    it['volume'] = m.get('volume','')
    it['issue'] = m.get('issue','')
    it['pages'] = m.get('page','') or m.get('article-number','')
    # date
    issued = m.get('issued',{})
    if issued and issued.get('date-parts') and issued['date-parts'][0]:
        y = issued['date-parts'][0][0]
        mo = issued['date-parts'][0][1] if len(issued['date-parts'][0])>1 else None
        it['date'] = f"{y}-{mo:02d}" if mo else str(y)
    elif e.get('cr_year'):
        it['date'] = str(e['cr_year'])
    it['DOI'] = e['doi']
    if it['itemType']=='book':
        it['publisher'] = m.get('publisher','')
    # creators
    creators = []
    for a in (m.get('author') or []):
        g = (a.get('given') or '').strip()
        f = (a.get('family') or '').strip()
        if not f and not g: continue
        creators.append({'creatorType':'author','firstName':g,'lastName':f})
    if creators:
        it['creators'] = creators
    return it

# new collection key placeholders (will be assigned after creation)
new_keys = {n: f'NEWKEY:{n}' for n in new_names}

items_create = []   # dicts: ref, doi, target(coll key or name), item
items_patch = []    # existing items to add to collections
dup_refs = set()
seen_doi = set()
for e in final:
    ref = e['ref']
    if not e['doi']:
        continue  # no-DOI, handled separately
    k = e['doi'].lower()
    if k in seen_doi:
        dup_refs.add(ref)
        continue
    seen_doi.add(k)

# existing items map (the 3 already in library)
existing_items = {
    '287': ('CYCSHICS', 'general'),
    '1427': ('SKF9XHNY', 'Yb'),
    '1800': ('44TSQ2IA', 'Yb'),
}

for e in final:
    ref = e['ref']
    if not e['doi']: continue
    if ref in dup_refs: continue
    if ref in existing_items:
        key, el = existing_items[ref]
        tgt = GRAP if el=='general' else existing_coll[el]
        items_patch.append({'ref':ref,'key':key,'addCollection':tgt,'element':el})
        continue
    el = e['element']
    if el == 'general':
        tgt = GRAP
    elif el in existing_coll:
        tgt = existing_coll[el]
    else:
        tgt = new_keys[el]
    items_create.append({'ref':ref,'doi':e['doi'],'target':tgt,'element':el,'item':zotero_item(e)})

plan = {
    'new_collections': [{'name':n,'parentCollection':GRAP} for n in new_names],
    'items_create': items_create,
    'items_patch': items_patch,
}
json.dump(plan, open('intercalation_doi_work/write_plan.json','w',encoding='utf-8'), ensure_ascii=False, indent=1)

# summary
from collections import Counter
tc = Counter(it['target'] for it in items_create)
print('=== WRITE PLAN SUMMARY ===')
print('new collections to create:', len(plan['new_collections']), new_names)
print('items to CREATE:', len(items_create))
print('items to PATCH (already exist, add to collection):', len(items_patch))
for p in items_patch:
    print('   ref', p['ref'], '->', p['key'], 'add to', p['addCollection'], '(',p['element'],')')
print()
print('create by target:')
for tgt, n in sorted(tc.items(), key=lambda x:-x[1]):
    print(f'   {tgt:<14} {n}')
# no-DOI refs
nodoi = [e['ref'] for e in final if not e['doi']]
print()
print('no-DOI (skipped):', nodoi)
print('duplicate refs (skipped):', sorted(dup_refs))
