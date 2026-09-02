#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sync_intercalation_page.py — Zotero「Intercalation/graphene」标签 → 插层元素周期表网页同步脚本

用法
----
    python3 scripts/sync_intercalation_page.py            # 等价于 --check
    python3 scripts/sync_intercalation_page.py --check    # 只读：Zotero 标签统计 vs 网页数据区，输出逐元素对照报告
    python3 scripts/sync_intercalation_page.py --apply    # 自动更新：备份 → 重写 5 个锚点区 → 写后断言 → 失败回滚

前置条件
--------
* 本地 Zotero 桌面端运行中（HTTP API 127.0.0.1:23119），仅只读 GET，无需 API key。
* 网页 intercalation_site/index.html 已含 5 组 SYNC 锚点（E/SHADE/SUBTITLE/LEGEND/FOOTER，本脚本会校验成对性）。

退出码
------
* 0   一致（--check）或更新成功且写后断言通过（--apply）
* 1   --check 发现差异或异常（元素计数/总数不一致、疑似化合物、锚点异常、API 失败）
* 2   --apply 拒绝写入（疑似化合物标签非空、API 请求失败、锚点异常）→ 转人工
* 3   --apply 写后断言失败，已用时间戳备份回滚

数据来源与统计口径
------------------
* 集合按名称发现：顶层 name=="Intercalation"，其下 name=="graphene"（parentCollection 匹配），不硬编码 key。
* 仅统计顶层文献（itemType in journalArticle/book），附件/笔记过滤，按 key 去重。
* 白名单标签计数（绝不黑名单）：
    - tag ∈ ELEMENTS（118 元素符号）        → 该元素直接计数 +1（同篇去重）
    - tag ∈ COMPOUNDS                      → 其组成元素各 +1（同篇论文对同一元素只 +1）
    - tag == "review"                      → 计入综述篇数，不归任何元素
    - 其余 tag                             → 忽略计数，列入「其他标签」报告清单
    - 形如 ^[A-Z][a-z]?([A-Z][a-z]?){1,}$ 且不在白名单 → 「疑似化合物」清单（--apply 拒绝写入）
* 元素论文数 = 直接标签 + 化合物来源（共享论文会在多个元素重复出现）。

扩展 COMPOUNDS / PAIRS
----------------------
* 新增化合物：在 COMPOUNDS 中加一项，如 {"MgO": {"elements": ["Mg","O"], "display": "MgO",
  "note_zh": "来自 MgO 插层体系", "note_en": "from MgO intercalation system"}}。display 仅用于展示
  （如 NbSe₂），note_zh/note_en 用于自动生成双语备注。
* 新增配对元素（同篇论文带两个元素标签）：在 PAIRS 中加对称两项，如 {"Li": "Mg", "Mg": "Li"}，
  备注自动生成为「来自 Li/Mg（碱土金属）」（按原子序数规范顺序）。
"""

import argparse
import json
import os
import re
import shutil
import sys
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime

# ---------------------------------------------------------------- 常量

BASE = "http://127.0.0.1:23119/api/users/0"
ITEM_TYPES = ("journalArticle", "book")
PAGE_SIZE = 100

# 全部 118 个元素符号（下标 = 原子序数 - 1，用于配对元素规范排序）
ELEMENTS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]
ELEMENT_SET = frozenset(ELEMENTS)

# 化合物标签白名单：elements=组成元素（各 +1），display=页面展示名，note_zh/note_en=自动双语备注用
COMPOUNDS = {
    "GaN": {"elements": ["Ga", "N"], "display": "GaN",
            "note_zh": "另见 GaN 体系", "note_en": "also in GaN system"},
    "NbSe": {"elements": ["Nb", "Se"], "display": "NbSe₂",
             "note_zh": "来自 NbSe₂ 二维材料体系", "note_en": "from NbSe₂ 2D material system"},
}

# 配对元素：同一论文同时带两标签（如 Sr+Ba 碱土金属共享文献）时适用，需对称定义；
# 备注自动生成为 zh「来自 Sr/Ba（碱土金属）」/ en「from Sr/Ba (alkaline earths)」（按原子序数规范顺序）
PAIRS = {"Sr": "Ba", "Ba": "Sr"}

# 疑似化合物标签正则（多字母大写开头、形如 GaN/NbSe 但不在白名单）
SUSPECT_RE = re.compile(r"^[A-Z][a-z]?([A-Z][a-z]?){1,}$")

# 5 组锚点：HTML 区用 <!-- -->，JS 区用 // 行注释
HTML_ANCHORS = ("SUBTITLE", "LEGEND", "FOOTER")
JS_ANCHORS = ("E", "SHADE")


class SyncError(Exception):
    """锚点/解析类异常。"""


def http_get(url):
    """GET 并返回 JSON（仅标准库）。"""
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def fetch_paged(url_fmt):
    """按 limit/start 分页拉取全部记录。"""
    out = []
    start = 0
    while True:
        page = http_get(url_fmt.format(start=start, limit=PAGE_SIZE))
        out.extend(page)
        if len(page) < PAGE_SIZE:
            break
        start += len(page)
        if start > 100000:  # 防御：异常循环兜底
            break
    return out


# ---------------------------------------------------------------- Zotero 读取

def find_graphene_collection_key():
    """按名称发现 Intercalation → graphene 集合（不硬编码 key）。"""
    cols = fetch_paged(BASE + "/collections?format=json&limit={limit}&start={start}")
    inter_key = None
    for c in cols:
        d = c.get("data", {})
        if d.get("name") == "Intercalation" and d.get("parentCollection") is False:
            inter_key = c.get("key")
            break
    if not inter_key:
        raise SyncError("未找到顶层分类「Intercalation」（按名称匹配）")
    for c in cols:
        d = c.get("data", {})
        if d.get("name") == "graphene" and d.get("parentCollection") == inter_key:
            return c.get("key")
    raise SyncError("未在「Intercalation」下找到子分类「graphene」")


def load_zotero():
    """读取 graphene 分类下全部顶层文献（journalArticle/book，按 key 去重）。"""
    coll_key = find_graphene_collection_key()
    url = BASE + "/collections/%s/items?format=json&limit={limit}&start={start}" % coll_key
    raw = fetch_paged(url)
    items, seen = [], set()
    for it in raw:
        d = it.get("data", {})
        if d.get("itemType") not in ITEM_TYPES:
            continue
        if it.get("key") in seen:
            continue
        seen.add(it.get("key"))
        items.append(it)
    return items


# ---------------------------------------------------------------- 标签统计

def count_tags(items):
    """
    白名单标签统计。
    返回 dict：counts(元素→论文数), direct_n, comp_src(元素→{化合物:篇数}),
    pair_flag, review_n, total, n_journal, n_book, max, unclassified,
    other_tags(Counter), suspect(list, 去重保序)。
    """
    counts = Counter()
    direct_n = Counter()
    comp_src = defaultdict(Counter)
    pair_flag = defaultdict(bool)
    other_tags = Counter()
    suspect = []
    review_n = 0
    n_journal = n_book = unclassified = 0

    for it in items:
        d = it.get("data", {})
        if d.get("itemType") == "journalArticle":
            n_journal += 1
        elif d.get("itemType") == "book":
            n_book += 1
        tags = [t.get("tag") for t in d.get("tags", []) if isinstance(t, dict) and t.get("tag")]

        direct_els = {t for t in tags if t in ELEMENT_SET}
        comp_els = {}  # 元素 -> 本论文命中的化合物集合（同篇对同一元素只 +1）
        for t in tags:
            if t in COMPOUNDS:
                for el in COMPOUNDS[t]["elements"]:
                    comp_els.setdefault(el, set()).add(t)

        if "review" in tags:
            review_n += 1

        for t in tags:
            if t in ELEMENT_SET or t in COMPOUNDS or t == "review":
                continue
            if SUSPECT_RE.fullmatch(t):
                if t not in suspect:
                    suspect.append(t)
            else:
                other_tags[t] += 1

        for el in direct_els:
            direct_n[el] += 1
            counts[el] += 1
        for el, cs in comp_els.items():
            for c in cs:
                comp_src[el][c] += 1
            counts[el] += 1

        for el in direct_els:
            partner = PAIRS.get(el)
            if partner and partner in direct_els:
                pair_flag[el] = True

        if not direct_els and not comp_els:
            unclassified += 1

    total = len(items)
    return {
        "counts": counts,
        "direct_n": direct_n,
        "comp_src": comp_src,
        "pair_flag": pair_flag,
        "review_n": review_n,
        "total": total,
        "n_journal": n_journal,
        "n_book": n_book,
        "max": max(counts.values()) if counts else 0,
        "n_highlight": sum(1 for v in counts.values() if v > 0),
        "unclassified": unclassified,
        "other_tags": other_tags,
        "suspect": suspect,
    }


# ---------------------------------------------------------------- 网页解析

def anchor_markers(name):
    """返回 (BEGIN, END) 锚点行文本。"""
    if name in JS_ANCHORS:
        return "// SYNC:%s:BEGIN" % name, "// SYNC:%s:END" % name
    return "<!-- SYNC:%s:BEGIN -->" % name, "<!-- SYNC:%s:END -->" % name


def parse_web(path):
    """读取 index.html，抽取各锚点区内容并解析网页当前数据（中英双语文案数字）。锚点缺失/不配对报错。"""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    sections = {}
    for name in HTML_ANCHORS + JS_ANCHORS:
        begin, end = anchor_markers(name)
        i = text.find(begin)
        j = text.find(end)
        if i == -1 or j == -1 or j <= i:
            raise SyncError("锚点 %s 缺失或不成对（BEGIN/END 未配对）" % name)
        i += len(begin)
        sections[name] = text[i:j]

    # E 数组（7 字段：原子序数/符号/中文名/英文名/论文数/中文备注/英文备注）
    elements = {}
    pat = re.compile(
        r'\[\s*(\d+)\s*,\s*"([A-Za-z]+)"\s*,\s*"([^"]*)"\s*,\s*"([^"]*)"\s*,\s*(null|-?\d+)'
        r'\s*,\s*((?:"[^"]*")|null)\s*,\s*((?:"[^"]*")|null)\s*\]'
    )
    for m in pat.finditer(sections["E"]):
        z, sym, zh, en, papers, note_zh, note_en = m.groups()
        elements[int(z)] = {
            "sym": sym,
            "zh": zh,
            "en": en,
            "papers": int(papers) if papers != "null" else None,
            "note_zh": None if note_zh == "null" else note_zh[1:-1],
            "note_en": None if note_en == "null" else note_en[1:-1],
        }
    if len(elements) != 118 or set(elements) != set(range(1, 119)):
        raise SyncError("E 数组解析异常：应有 118 个元素，实际 %d 个" % len(elements))

    # shade 上限
    m = re.search(r"var t = \(p - 1\) / \((\d+) - 1\);", sections["SHADE"])
    if not m:
        raise SyncError("SHADE 锚点区未解析出渐变上限")
    shade_max = int(m.group(1))

    # 副标题（zh/en 两行）
    m = re.search(r"<strong>(\d+) 种</strong>，累计 <strong>(\d+) 篇</strong> 文献（(\d+) 篇期刊论文 \+ (\d+) 本专著）", sections["SUBTITLE"])
    if not m:
        raise SyncError("SUBTITLE 锚点区未解析出中文副标题数字")
    sub_zh = tuple(int(x) for x in m.groups())
    m = re.search(r"<strong>(\d+)</strong>, totaling <strong>(\d+)</strong> references \((\d+) journal articles \+ (\d+) books\)", sections["SUBTITLE"])
    if not m:
        raise SyncError("SUBTITLE 锚点区未解析出英文副标题数字")
    sub_en = tuple(int(x) for x in m.groups())

    # 表内图例（LEGEND）：高亮元素数（zh/en）+ 渐变两端标签（zh/en）
    m = re.search(r"已用于插层工作的元素（(\d+) 种）", sections["LEGEND"])
    if not m:
        raise SyncError("LEGEND 锚点区未解析出中文高亮元素数")
    legend_N = int(m.group(1))
    m = re.search(r"Elements reported for intercalation \((\d+)\)", sections["LEGEND"])
    if not m:
        raise SyncError("LEGEND 锚点区未解析出英文高亮元素数")
    legend_N_en = int(m.group(1))
    caps = re.findall(r'grad-label" data-lang="zh">(\d+) 篇</span>', sections["LEGEND"])
    caps_en = re.findall(r'grad-label" data-lang="en">(\d+) paper', sections["LEGEND"])
    if len(caps) != 2 or len(caps_en) != 2:
        raise SyncError("LEGEND 锚点区未解析出渐变标签")
    grad_max = max(int(x) for x in caps)
    grad_max_en = max(int(x) for x in caps_en)

    # 页脚（FOOTER）：注释段 review 篇数 + 数据来源行（zh/en）
    m = re.search(r"另有 (\d+) 篇综述", sections["FOOTER"])
    note_review = int(m.group(1)) if m else None
    m = re.search(r"Another (\d+) review", sections["FOOTER"])
    note_review_en = int(m.group(1)) if m else None
    m = re.search(r"去附件后 (\d+) 篇文献：(\d+) 篇期刊论文 \+ (\d+) 本专著", sections["FOOTER"])
    if not m:
        raise SyncError("FOOTER 锚点区未解析出中文文献数")
    foot_zh = tuple(int(x) for x in m.groups())
    m = re.search(r"excluding attachments: (\d+) references — (\d+) journal articles \+ (\d+) books", sections["FOOTER"])
    if not m:
        raise SyncError("FOOTER 锚点区未解析出英文文献数")
    foot_en = tuple(int(x) for x in m.groups())

    return {
        "sections": sections,
        "elements": elements,
        "shade_max": shade_max,
        "sub": sub_zh,
        "sub_en": sub_en,
        "legend_N": legend_N,
        "legend_N_en": legend_N_en,
        "grad_max": grad_max,
        "grad_max_en": grad_max_en,
        "note_review": note_review,
        "note_review_en": note_review_en,
        "foot": foot_zh,
        "foot_en": foot_en,
    }


# ---------------------------------------------------------------- 生成与写入

def pair_label(sym):
    """配对元素规范展示顺序（按原子序数升序），如 Sr/Ba。"""
    other = PAIRS[sym]
    a, b = (sym, other) if ELEMENT_SET and ELEMENTS.index(sym) < ELEMENTS.index(other) else (other, sym)
    return "%s/%s" % (a, b)


def gen_note(sym, direct, comp_sources, pair_flag):
    """按规则生成双语元素备注（返回 (zh, en)，None = 无备注）。"""
    if direct > 0 and comp_sources:
        extras_zh = "、".join(COMPOUNDS[c]["note_zh"] for c in comp_sources)
        extras_en = "; ".join(COMPOUNDS[c]["note_en"] for c in comp_sources)
        return ("%s 插层 %d 篇；%s" % (sym, direct, extras_zh),
                "%s: %d papers; %s" % (sym, direct, extras_en))
    if direct > 0 and pair_flag:
        label = pair_label(sym)
        return "来自 %s（碱土金属）" % label, "from %s (alkaline earths)" % label
    if direct == 0 and comp_sources:
        return ("、".join(COMPOUNDS[c]["note_zh"] for c in comp_sources),
                "; ".join(COMPOUNDS[c]["note_en"] for c in comp_sources))
    return None, None


def build_sections(web, stats):
    """按 Zotero 统计重建 5 个锚点区内容（不含锚点行本身），文案中英双语。"""
    counts, direct_n, comp_src, pair_flag = (
        stats["counts"], stats["direct_n"], stats["comp_src"], stats["pair_flag"],
    )
    n_high, total, x, y, k, mx = (
        stats["n_highlight"], stats["total"], stats["n_journal"], stats["n_book"],
        stats["review_n"], stats["max"],
    )

    lines = ["  var E = ["]
    for z in range(1, 119):
        e = web["elements"][z]
        sym = e["sym"]
        if z == 6:
            papers, note_zh, note_en = None, "host", "host"
        else:
            n = counts.get(sym, 0)
            if n > 0:
                cs = [c for c in COMPOUNDS if comp_src[sym].get(c, 0) > 0]
                note_zh, note_en = gen_note(sym, direct_n.get(sym, 0), cs, pair_flag.get(sym, False))
                papers = n
            else:
                papers, note_zh, note_en = None, None, None
        papers_s = "null" if papers is None else str(papers)
        note_zh_s = "null" if note_zh is None else json.dumps(note_zh, ensure_ascii=False)
        note_en_s = "null" if note_en is None else json.dumps(note_en, ensure_ascii=False)
        lines.append('    [%d,"%s","%s","%s",%s,%s,%s],' % (z, sym, e["zh"], e["en"], papers_s, note_zh_s, note_en_s))
    lines[-1] = lines[-1][:-1]  # 去掉最后一行尾逗号
    lines.append("  ];")

    sections = {
        "E": "\n".join(lines),
        "SHADE": "    var t = (p - 1) / (%d - 1);          // 1 篇 -> 0, %d 篇 -> 1" % (mx, mx),
        "SUBTITLE": (
            '    <p class="subtitle" data-lang="zh">已报道插层元素/体系 <strong>%d 种</strong>，累计 <strong>%d 篇</strong> 文献（%d 篇期刊论文 + %d 本专著）。</p>\n'
            '    <p class="subtitle" data-lang="en">Elements/systems reported for intercalation: <strong>%d</strong>, totaling <strong>%d</strong> references (%d journal articles + %d books).</p>'
            % (n_high, total, x, y, n_high, total, x, y)
        ),
        "LEGEND": (
            '    <div class="legend-block" style="grid-column:3/13; grid-row:2/4;">\n'
            '      <div class="legend-grad">\n'
            '        <span class="grad-label" data-lang="zh">1 篇</span><span class="grad-label" data-lang="en">1 paper</span>\n'
            '        <span class="grad"></span>\n'
            '        <span class="grad-label" data-lang="zh">%d 篇</span><span class="grad-label" data-lang="en">%d papers</span>\n'
            '      </div>\n'
            '      <div class="legend-items">\n'
            '        <div class="litem"><span class="swatch"></span><span data-lang="zh">已用于插层工作的元素（%d 种）</span><span data-lang="en">Elements reported for intercalation (%d)</span></div>\n'
            '        <div class="litem"><span class="swatch gray"></span><span data-lang="zh">未报道用于插层</span><span data-lang="en">Not reported for intercalation</span></div>\n'
            '        <div class="litem"><span class="swatch host"></span><span data-lang="zh">C 不计入插层元素</span><span data-lang="en">C not counted as intercalant</span></div>\n'
            '      </div>\n'
            '    </div>'
            % (mx, mx, n_high, n_high)
        ),
        "FOOTER": (
            '    <p class="fnote" data-lang="zh">GaN、NbSe₂ 为石墨烯封装/限域异质外延形成的二维材料体系（其组成元素 Ga、N、Nb、Se 一并高亮）；Sr/Ba 为碱土金属共享文献。各元素右上角 ×N 为该元素作为插层物种的论文数（按标签计，共享论文会在多个元素重复出现）。另有 %d 篇综述/综合类文献（review 标签）未归入单一元素。</p>\n'
            '    <p class="fnote" data-lang="en">GaN and NbSe₂ are 2D material systems formed by graphene-encapsulated/confined heteroepitaxy (their constituent elements Ga, N, Nb, Se are highlighted together); Sr/Ba share the same references (alkaline earths). The ×N badge shows the number of papers for each element as an intercalant (counted by tag; shared papers count toward multiple elements). Another %d review/comprehensive references (review tag) are not assigned to a single element.</p>\n'
            '    <p class="fsrc" data-lang="zh">数据来源：Zotero「Intercalation / graphene」分类（按元素标签统计，去附件后 %d 篇文献：%d 篇期刊论文 + %d 本专著，其中 %d 篇带 review 标签）。本页为自包含静态网页，离线可直接打开。</p>\n'
            '    <p class="fsrc" data-lang="en">Data source: Zotero “Intercalation / graphene” collection (counted by element tags, excluding attachments: %d references — %d journal articles + %d books, %d tagged review). This is a self-contained static page that works offline.</p>'
            % (k, k, total, x, y, k, total, x, y, k)
        ),
    }
    return sections


def splice_sections(text, sections):
    """用新内容替换锚点间文本（锚点行本身保留），返回新全文。"""
    for name, content in sections.items():
        begin, end = anchor_markers(name)
        i = text.find(begin)
        j = text.find(end)
        if i == -1 or j == -1 or j <= i:
            raise SyncError("锚点 %s 缺失或不成对，无法写入" % name)
        i += len(begin)
        if text[i:i + 1] == "\n":  # 跳过 BEGIN 行尾换行，保证锚点独占一行
            i += 1
        # 回溯到 END 锚点所在行的行首（先退缩进空白，再退换行），保证锚点独占一行
        k = j
        while k > 0 and text[k - 1] in " \t":
            k -= 1
        if k > 0 and text[k - 1] == "\n":
            k -= 1
        j = k
        text = text[:i] + content + text[j:]
    return text


def verify_written(web_path, stats):
    """写后断言：高亮元素数、各元素 papers、总数（+渐变上限）与 Zotero 完全一致，锚点结构合法。"""
    web2 = parse_web(web_path)
    with open(web_path, encoding="utf-8") as f:
        text = f.read()
    counts = stats["counts"]
    diffs = []
    # 锚点结构：BEGIN/END 必须各自独占一行（行内仅缩进空白 + 标记；否则 JS 区内容会被 // 注释吞掉）
    for name in HTML_ANCHORS + JS_ANCHORS:
        for marker in anchor_markers(name):
            i = text.find(marker)
            if i == -1:
                diffs.append("锚点 %s 缺失" % marker)
                continue
            line_start = text.rfind("\n", 0, i) + 1
            prefix = text[line_start:i]
            if prefix.strip(" \t"):
                diffs.append("锚点 %s 行首有非空白内容 %r" % (marker, prefix))
            after = text[i + len(marker):i + len(marker) + 1]
            if after != "\n":
                diffs.append("锚点 %s 行尾有内容（后随 %r）" % (marker, after))
    for z in range(1, 119):
        e = web2["elements"][z]
        want = counts.get(e["sym"], 0) if z != 6 else 0
        got = e["papers"] or 0
        if got != want:
            diffs.append("%s: 网页=%s Zotero=%s" % (e["sym"], got, want))
    n_high = web2["sub"][0]
    if n_high != stats["n_highlight"]:
        diffs.append("高亮元素数: 网页=%d Zotero=%d" % (n_high, stats["n_highlight"]))
    if web2["sub"][1] != stats["total"]:
        diffs.append("总数: 网页=%d Zotero=%d" % (web2["sub"][1], stats["total"]))
    if web2["shade_max"] != stats["max"] or web2["grad_max"] != stats["max"]:
        diffs.append("渐变上限: 网页=%s Zotero=%d" % ((web2["shade_max"], web2["grad_max"]), stats["max"]))
    # 英文文案数字与 Zotero 期望值一致（en 与 zh 同步）
    exp_sub = (stats["n_highlight"], stats["total"], stats["n_journal"], stats["n_book"])
    if web2["sub_en"] != exp_sub:
        diffs.append("英文副标题: 网页=%s Zotero=%s" % (web2["sub_en"], exp_sub))
    if web2["legend_N_en"] != stats["n_highlight"]:
        diffs.append("英文图例元素数: 网页=%s Zotero=%d" % (web2["legend_N_en"], stats["n_highlight"]))
    if web2["grad_max_en"] != stats["max"]:
        diffs.append("英文渐变上限: 网页=%s Zotero=%d" % (web2["grad_max_en"], stats["max"]))
    if web2["note_review_en"] != stats["review_n"]:
        diffs.append("英文注释 review: 网页=%s Zotero=%d" % (web2["note_review_en"], stats["review_n"]))
    exp_foot = (stats["total"], stats["n_journal"], stats["n_book"])
    if web2["foot_en"] != exp_foot:
        diffs.append("英文数据来源: 网页=%s Zotero=%s" % (web2["foot_en"], exp_foot))
    return diffs


# ---------------------------------------------------------------- 报告

def fmt_stats(stats):
    counts = stats["counts"]
    lines = []
    lines.append("== Zotero 统计（按名称发现 Intercalation → graphene 分类） ==")
    lines.append("总文献: %d（期刊论文 %d，专著 %d）" % (stats["total"], stats["n_journal"], stats["n_book"]))
    lines.append("review 标签: %d 篇" % stats["review_n"])
    lines.append("高亮元素: %d 种，max = %d（%s）" % (
        stats["n_highlight"], stats["max"],
        "、".join(s for s, v in counts.items() if v == stats["max"]) or "-"))
    lines.append("元素计数: " + ("、".join("%s %d" % (s, v) for s, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))) or "（空）"))
    lines.append("未分类论文（无任何元素/化合物标签）: %d" % stats["unclassified"])
    if stats["other_tags"]:
        lines.append("其他标签: %s（%d 个标签名，不影响元素统计）" % (
            "、".join("%s×%d" % (t, c) for t, c in stats["other_tags"].most_common()),
            len(stats["other_tags"])))
    else:
        lines.append("其他标签: （空）")
    lines.append("疑似化合物标签: %s" % ("、".join(stats["suspect"]) if stats["suspect"] else "（空）"))
    return "\n".join(lines)


def fmt_check(web, stats):
    """--check 对照报告，返回 (文本, 差异数)。"""
    counts = stats["counts"]
    out = []
    out.append("== 网页 vs Zotero 逐元素对照 ==")
    diffs = []
    rows = []
    for z in range(1, 119):
        e = web["elements"][z]
        sym = e["sym"]
        web_n = e["papers"] or 0
        zot_n = counts.get(sym, 0)
        if web_n == 0 and zot_n == 0:
            continue
        ok = "✓" if web_n == zot_n else "✗"
        rows.append("  %-3s 网页=%-3s Zotero=%-3s %s" % (sym, web_n if web_n else "-", zot_n if zot_n else "-", ok))
        if web_n != zot_n:
            diffs.append(sym)
    out.append("（仅列出网页或 Zotero 非零的元素；零差异元素已省略）" if len(rows) > 20 else "")
    out.extend(rows)

    web_only = [web["elements"][z]["sym"] for z in range(1, 119)
                if (web["elements"][z]["papers"] or 0) > 0 and counts.get(web["elements"][z]["sym"], 0) == 0]
    zot_only = [s for s, v in counts.items() if v > 0 and (web["elements"][ELEMENTS.index(s) + 1]["papers"] or 0) == 0]

    out.append("")
    out.append("== 汇总对比 ==")
    sub_N, sub_M, sub_X, sub_Y = web["sub"]
    pairs = [
        ("高亮元素数", sub_N, stats["n_highlight"]),
        ("文献总数", sub_M, stats["total"]),
        ("期刊论文", sub_X, stats["n_journal"]),
        ("专著", sub_Y, stats["n_book"]),
        ("review 篇数", web["note_review"] if web["note_review"] is not None else sub_M - sub_X - sub_Y, stats["review_n"]),
        ("渐变上限 max", web["shade_max"], stats["max"]),
    ]
    total_diff = 0
    for label, w, z in pairs:
        ok = "✓" if w == z else "✗"
        out.append("  %-12s 网页=%-6s Zotero=%-6s %s" % (label, w, z, ok))
        if w != z:
            total_diff += 1
    if web["grad_max_en"] != web["grad_max"]:
        total_diff += 1
        out.append("  英文渐变上限: en=%s zh=%s ✗" % (web["grad_max_en"], web["grad_max"]))
    # 英文文案数字须与中文一致（额外断言）
    en_checks = [
        ("英文副标题", web["sub_en"], web["sub"]),
        ("英文图例元素数", web["legend_N_en"], web["legend_N"]),
        ("英文注释 review", web["note_review_en"], web["note_review"]),
        ("英文数据来源", web["foot_en"], web["foot"]),
    ]
    for label, en_val, zh_val in en_checks:
        if en_val != zh_val:
            total_diff += 1
            out.append("  %s: en=%s zh=%s ✗" % (label, en_val, zh_val))

    n_diff = len(diffs) + total_diff
    out.append("")
    out.append("差异元素: %s" % ("、".join(diffs) if diffs else "（空）"))
    out.append("仅网页高亮: %s" % ("、".join(web_only) if web_only else "（空）"))
    out.append("仅 Zotero 高亮: %s" % ("、".join(zot_only) if zot_only else "（空）"))
    out.append("")
    if stats["suspect"]:
        out.append("⚠ 疑似化合物标签非空（%s）→ 需人工确认" % "、".join(stats["suspect"]))
        n_diff += 1
    if n_diff == 0:
        out.append("结果: 完全一致 ✅（退出码 0）")
    else:
        out.append("结果: 存在 %d 处差异/异常 ❌（退出码 1）" % n_diff)
    return "\n".join(out), n_diff


# ---------------------------------------------------------------- 主流程

def main():
    parser = argparse.ArgumentParser(
        description="同步 Zotero「Intercalation/graphene」标签统计到插层元素周期表网页")
    parser.add_argument("mode", nargs="?", choices=["check", "apply"],
                        help="运行模式（默认 check；也可用 --check/--apply）")
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true", help="只读：输出对照报告（默认）")
    g.add_argument("--apply", action="store_true", help="自动更新：备份→重写→写后断言→失败回滚")
    args = parser.parse_args()
    if args.apply:
        mode = "apply"
    elif args.check:
        mode = "check"
    else:
        mode = args.mode or "check"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    index_html = os.path.join(os.path.dirname(script_dir), "intercalation_site", "index.html")

    # 1) 读 Zotero
    try:
        items = load_zotero()
        stats = count_tags(items)
    except SyncError as e:
        print("Zotero 读取失败：%s" % e)
        sys.exit(2 if mode == "apply" else 1)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        print("Zotero API 请求失败：%s（请确认本地 Zotero 桌面端已运行）" % e)
        sys.exit(2 if mode == "apply" else 1)

    print(fmt_stats(stats))
    print()

    # 2) 读网页
    try:
        web = parse_web(index_html)
    except SyncError as e:
        print("网页解析失败：%s" % e)
        sys.exit(2 if mode == "apply" else 1)
    except OSError as e:
        print("网页读取失败：%s" % e)
        sys.exit(2 if mode == "apply" else 1)

    report, n_diff = fmt_check(web, stats)
    print(report)

    if mode == "check":
        sys.exit(1 if n_diff else 0)

    # ---- apply ----
    if stats["suspect"]:
        print("\n⚠ --apply 拒绝写入：疑似化合物标签非空（%s）→ 转人工确认" % "、".join(stats["suspect"]))
        sys.exit(2)

    backup = "%s.bak.%s" % (index_html, datetime.now().strftime("%Y%m%d%H%M%S"))
    try:
        sections = build_sections(web, stats)
        with open(index_html, encoding="utf-8") as f:
            text = f.read()
        new_text = splice_sections(text, sections)
    except SyncError as e:
        print("\n--apply 中止：%s" % e)
        sys.exit(2)

    shutil.copy2(index_html, backup)
    try:
        with open(index_html, "w", encoding="utf-8") as f:
            f.write(new_text)
        diffs = verify_written(index_html, stats)
        if diffs:
            raise AssertionError("；".join(diffs))
    except AssertionError as e:
        shutil.copy2(backup, index_html)
        print("\n✗ 写后断言失败，已用备份回滚：%s" % backup)
        print("  断言差异：%s" % e)
        sys.exit(3)
    except OSError as e:
        shutil.copy2(backup, index_html)
        print("\n✗ 写入失败，已用备份回滚：%s（%s）" % (backup, e))
        sys.exit(3)

    print("\n✔ --apply 完成")
    print("  备份: %s" % backup)
    print("  已重写锚点区: E（118 元素，7 字段双语备注）、SHADE、SUBTITLE、LEGEND（表内图例）、FOOTER（注释+数据来源，双语）")
    print("  写后断言: 高亮元素数=%d、逐元素 papers、总数=%d、渐变上限=%d 全部与 Zotero 一致"
          % (stats["n_highlight"], stats["total"], stats["max"]))
    sys.exit(0)


if __name__ == "__main__":
    main()
