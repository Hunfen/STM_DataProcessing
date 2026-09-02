import json

# element mapping: ref -> element symbol ('general' = no single element/review/book/multi)
elem = {
 '11':'Li','13':'general','14':'general','15':'general','76':'Li','89':'Au','90':'Au','91':'general',
 '97':'Li','98':'Na','118':'O','122':'Li','133':'O','136':'general','137':'general','139':'H',
 '177':'N','180':'general','183':'K','234':'Pb','244':'Dy','265':'general','287':'general',
 '290':'Si','292':'Fe','305':'general','312':'Fe','316':'Cu','345':'general','357':'general',
 '391':'Ca','392':'Li','394':'general','412':'Li','419':'general','429':'general','485':'Ca',
 '491':'Bi','539':'O','551':'H','554':'Mg','555':'K','560':'Pb','564':'Li','578':'Au',
 '590':'general','605':'Pb','662':'Co','673':'general','683':'general','695':'K','725':'Pt',
 '729':'Li','736':'general','763':'general','764':'general','792':'Ca','793':'Eu','795':'Al',
 '806':'general','808':'H','838':'In','840':'Sr/Ba','845':'Ca','852':'Ge','863':'Rb','864':'Cs',
 '865':'Rb','868':'Pb','871':'K','883':'general','884':'K','886':'Gd','899':'general','901':'O',
 '981':'Li','985':'general','1023':'Gd','1067':'general','1069':'general','1088':'general',
 '1132':'Pb','1139':'Ca','1154':'Fe','1164':'Mo','1203':'general','1234':'Sn','1278':'Cu',
 '1297':'K','1300':'K','1341':'Cs','1357':'general','1359':'Au','1369':'general','1411':'general',
 '1416':'general','1423':'H','1425':'O','1427':'Yb','1475':'Pb','1482':'Bi','1490':'Eu',
 '1492':'Eu','1533':'general','1553':'Si','1554':'Si','1555':'Si','1576':'H','1590':'Ag',
 '1593':'general','1603':'Eu','1604':'Li','1617':'general','1661':'Li','1672':'general',
 '1701':'O','1722':'Ge','1731':'Li','1742':'Si','1744':'Co','1745':'Co','1771':'Ge','1800':'Yb',
 '1824':'Fe','1859':'general','1861':'O','1864':'Ga','1912':'Er','1938':'general','1941':'general',
 '1947':'Ca','1970':'Fe','1977':'general','1987':'Li',
}

# existing subcollection keys under graphene (CYS4DBUL)
existing = {
 'Ag':'J8XK3NBF','Au':'K2C35XUC','Bi':'GCLKI4JV','Ca':'A77J76WZ','Ce':'PEFLAJ3E','Co':'5XLC549M',
 'Cu':'QZIQ4LED','Dy':'PFZAQ3GR','Eu':'G6KSSEBR','Fe':'CAVWIWSY','F':'HWRXB7NS','GaN':'BVYDVP4G',
 'Gd':'M67S6HKS','Ge':'CYUJX6PT','H':'RV4XYK6H','In':'XRVP2C3Z','Li':'BCJ4F6X5','Mn':'7GGP3HLU',
 'N':'P2PTPXCI','Na':'9L9KRFW5','NbSe2':'59TAHAG4','O':'F8SJ2EX8','Pb':'TXT3UESN','Pd':'WA8MAAGB',
 'Pt':'ZMNVSY5D','Si':'PID8AW7M','Sm':'QUSFG2PC','Sn':'ZTBKP9RR','Tb':'5L5MHW8B','Yb':'3SH9SYXF',
}
# new subcollections to create (name -> key placeholder)
new = ['K','Rb','Cs','Mg','Al','Sr/Ba','Ga','Er','Mo']

final = json.load(open('intercalation_doi_work/final_doi.json', encoding='utf-8'))
# mark duplicate [1555] = same DOI as [1553]
dup = set()
seen_doi = {}
for e in final:
    if e['doi']:
        k = e['doi'].lower()
        if k in seen_doi:
            dup.add(e['ref'])
        else:
            seen_doi[k] = e['ref']

from collections import Counter, defaultdict
targets = defaultdict(list)
for e in final:
    ref = e['ref']
    if ref in dup:
        e['target'] = 'DUP:'+seen_doi[e['doi'].lower()]
        e['element'] = elem[ref]
        continue
    el = elem[ref]
    e['element'] = el
    if el == 'general':
        e['target'] = 'graphene'
        targets['graphene'].append(ref)
    elif el in existing:
        e['target'] = existing[el]
        targets[existing[el]].append(ref)
    else:
        e['target'] = 'NEW:'+el
        targets['NEW:'+el].append(ref)

json.dump(final, open('intercalation_doi_work/final_doi.json','w',encoding='utf-8'), ensure_ascii=False, indent=1)

print('=== target distribution ===')
c = Counter()
for e in final:
    c[e['element']] += 1
for el in sorted(c, key=lambda x:-c[x]):
    tgt = 'graphene' if el=='general' else (existing[el] if el in existing else 'NEW:'+el)
    print(f'{el:>8} -> {tgt:<14} count={c[el]}')
print()
print('duplicates flagged:', sorted(dup))
print('total entries:', len(final), '| unique-to-write:', len(final)-len(dup))
