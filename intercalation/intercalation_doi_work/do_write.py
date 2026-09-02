import json, urllib.request, urllib.error, time, sys

KEY = "LSFT10kyamjMIqRr0xLBSVugAaA2nRmT"
SID = "UVpsFxMAUyau"
BASE = "http://127.0.0.1:23119"
HDRS = {"Authorization": f"Bearer {KEY}", "Zotero-Server-ID": SID, "Content-Type": "application/json"}

def req(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(BASE+path, data=data, headers=HDRS, method=method)
    try:
        with urllib.request.urlopen(r, timeout=60) as resp:
            txt = resp.read().decode()
            return resp.status, (json.loads(txt) if txt.strip() else None)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()

plan = json.load(open('intercalation_doi_work/write_plan.json', encoding='utf-8'))

# 1. create collections
print('=== creating collections ===')
status, resp = req('POST', '/api/users/0/collections', plan['new_collections'])
print('status', status, json.dumps(resp)[:800] if resp else '')
name2key = {}
if isinstance(resp, dict) and 'successful' in resp:
    for entry in resp['successful']:
        name2key[entry['name']] = entry['key']
        print('  created', entry['name'], '->', entry['key'])
if 'failed' in resp:
    print('  FAILED:', resp['failed'])
if isinstance(resp, dict) and 'success' in resp:  # alternate shape
    for k,v in resp['success'].items():
        pass
json.dump(name2key, open('intercalation_doi_work/new_collection_keys.json','w'), ensure_ascii=False, indent=1)
print('name2key:', name2key)
