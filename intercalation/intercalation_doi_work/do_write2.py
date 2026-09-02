import json, urllib.request, urllib.error

KEY = "LSFT10kyamjMIqRr0xLBSVugAaA2nRmT"
SID = "UVpsFxMAUyau"
BASE = "http://127.0.0.1:23119"
HDRS = {"Authorization": f"Bearer {KEY}", "Zotero-Server-ID": SID, "Content-Type": "application/json"}

def req(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(BASE+path, data=data, headers=HDRS, method=method)
    try:
        with urllib.request.urlopen(r, timeout=90) as resp:
            txt = resp.read().decode()
            return resp.status, (json.loads(txt) if txt.strip() else None)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()

def get_collections():
    st, cols = req('GET', '/api/users/0/collections?format=json&limit=100')
    return cols

# map new collection names -> keys (children of CYS4DBUL)
cols = get_collections()
name2key = {}
for c in cols:
    d = c.get('data', {})
    if d.get('parentCollection') == 'CYS4DBUL':
        name2key[d['name']] = d['key']
print('existing children of graphene:', sorted(name2key.keys()))

plan = json.load(open('intercalation_doi_work/write_plan.json', encoding='utf-8'))
new_names = [c['name'] for c in plan['new_collections']]
missing = [n for n in new_names if n not in name2key]
print('new collections present:', {n: name2key.get(n) for n in new_names})
if missing:
    print('MISSING (need create):', missing)

GRAP = 'CYS4DBUL'
existing_coll = {
 'Ag':'J8XK3NBF','Au':'K2C35XUC','Bi':'GCLKI4JV','Ca':'A77J76WZ','Ce':'PEFLAJ3E','Co':'5XLC549M',
 'Cu':'QZIQ4LED','Dy':'PFZAQ3GR','Eu':'G6KSSEBR','Fe':'CAVWIWSY','F':'HWRXB7NS','GaN':'BVYDVP4G',
 'Gd':'M67S6HKS','Ge':'CYUJX6PT','H':'RV4XYK6H','In':'XRVP2C3Z','Li':'BCJ4F6X5','Mn':'7GGP3HLU',
 'N':'P2PTPXCI','Na':'9L9KRFW5','NbSe2':'59TAHAG4','O':'F8SJ2EX8','Pb':'TXT3UESN','Pd':'WA8MAAGB',
 'Pt':'ZMNVSY5D','Si':'PID8AW7M','Sm':'QUSFG2PC','Sn':'ZTBKP9RR','Tb':'5L5MHW8B','Yb':'3SH9SYXF',
}

# resolve target keys
def resolve(tgt):
    if tgt.startswith('NEWKEY:'):
        return name2key[tgt[7:]]
    return tgt

# build item payloads
payloads = []
for it in plan['items_create']:
    tgt = resolve(it['target'])
    item = dict(it['item'])
    item['collections'] = [tgt]
    payloads.append((it['ref'], it['doi'], tgt, item))

print('items to create:', len(payloads))

# batch create (50 max)
def post_items(batch):
    st, resp = req('POST', '/api/users/0/items', [p[3] for p in batch])
    return st, resp

results_ok = []
results_fail = []
for i in range(0, len(payloads), 50):
    batch = payloads[i:i+50]
    st, resp = post_items(batch)
    if isinstance(resp, dict) and 'successful' in resp:
        for idx, v in resp['successful'].items():
            results_ok.append((batch[int(idx)][0], v.get('key'), v.get('data',{}).get('title','')))
        for idx, v in resp.get('failed', {}).items():
            results_fail.append((batch[int(idx)][0], v))
        print(f'batch {i//50}: ok={len(resp["successful"])} fail={len(resp.get("failed",{}))}')
    else:
        print('batch error status', st, str(resp)[:500])
        results_fail.append(('batch', str(resp)[:300]))

print('created OK:', len(results_ok), '| failed:', len(results_fail))
for f in results_fail:
    print('  FAIL', f)

# patch existing items
patch_plan = {
 '287': ('CYCSHICS', ['2ACVK684', 'CYS4DBUL']),
 '1427': ('SKF9XHNY', ['2ACVK684','MKVCXMVC','3SH9SYXF']),
 '1800': ('44TSQ2IA', ['2ACVK684','3SH9SYXF']),
}
print('=== patching existing items ===')
for ref, (key, cols) in patch_plan.items():
    st, resp = req('PATCH', f'/api/users/0/items/{key}', {'collections': cols})
    print(f'  ref {ref} {key}: status {st}', str(resp)[:120] if resp else '')

json.dump({'created': results_ok, 'failed': results_fail}, open('intercalation_doi_work/write_result.json','w'), ensure_ascii=False, indent=1)
print('DONE')
