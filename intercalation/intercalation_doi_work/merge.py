import json

data = {d['ref']: d for d in json.load(open('intercalation_doi_work/parsed.json', encoding='utf-8'))}
raw = json.load(open('intercalation_doi_work/crossref_raw.json', encoding='utf-8'))

# resolved low-confidence (ref -> (doi, crossref_container, crossref_year))
resolved = {
 '345': ('10.1080/00018730110113644','Advances in Physics','2002'),
 '394': ('10.1093/oso/9780195128277.001.0001','Oxford University Press','2003'),
 '725': ('10.1016/j.apsusc.2021.151345','Applied Surface Science','2022'),
 '763': (None,'',''),   # no DOI (Hikari journal)
 '901': ('10.1063/1.4914161','Journal of Applied Physics','2015'),
 '1411': ('10.1088/0022-3727/43/37/374009','Journal of Physics D: Applied Physics','2010'),
 '1416': (None,'',''),   # no DOI (thesis)
 '1554': ('10.1002/sia.5574','Surface and Interface Analysis','2014'),
 '1661': ('10.1088/1361-648x/aa88fb','Journal of Physics: Condensed Matter','2017'),
 '1672': ('10.1070/pu1993v036n11abeh002180','Physics-Uspekhi','1993'),
 '1722': ('10.1038/srep17700','Scientific Reports','2015'),
 '1731': ('10.1088/1367-2630/12/12/125015','New Journal of Physics','2010'),
 '1742': ('10.1103/physrevb.94.245421','Physical Review B','2016'),
 '1744': ('10.1063/1.4868119','Applied Physics Letters','2014'),
 '1745': ('10.1103/physrevmaterials.1.053406','Physical Review Materials','2017'),
 '1800': ('10.4236/graphene.2013.22010','Graphene','2013'),
 '1859': ('10.1016/j.progsurf.2021.100637','Progress in Surface Science','2021'),
 '1861': ('10.1016/j.carbon.2015.11.002','Carbon','2016'),
 '1938': ('10.1007/978-3-642-75270-4','Springer Series in Materials Science','1990'),
 '1977': ('10.1038/s41586-020-2241-9','Nature','2020'),
}

final = []
for r in raw:
    ref = r['ref']
    d = data[ref]
    if ref in resolved:
        doi, cj, cy = resolved[ref]
        entry = dict(ref=ref, title=d['title'], journal=d['journal'], year=d['year'], doi=doi, cr_container=cj, cr_year=cy, source='manual')
    else:
        b = r['candidates'][0]
        entry = dict(ref=ref, title=d['title'], journal=d['journal'], year=d['year'], doi=b['doi'], cr_container=b['container'], cr_year=b['year'], source='crossref')
    final.append(entry)

final.sort(key=lambda e: int(e['ref']))
json.dump(final, open('intercalation_doi_work/final_doi.json','w',encoding='utf-8'), ensure_ascii=False, indent=1)
nodoi = [e for e in final if not e['doi']]
print('total', len(final), '| with DOI', len(final)-len(nodoi), '| no DOI', len(nodoi))
print('NO DOI entries:')
for e in nodoi:
    print('  ', e['ref'], e['title'][:70])
