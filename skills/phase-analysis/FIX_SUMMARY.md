# FIX_SUMMARY — stm-topo-phase-analysis v2.1（三个缺陷的修复与验证）

任务：`t1`（stm-skill-fix / engineer）。全部改动在 **staging 副本** `data_processing/_skill_fix/`
（= `/Users/hunfen/.agents/skills/stm-topo-phase-analysis` 的整目录拷贝）上进行。

**没有改动**：`data/`、`data_processing/topo4_50nm_analysis_mask10px/`、`data_processing/phase_reanalysis/`、
`/Users/hunfen/.agents/skills/`、`/Users/hunfen/Documents/GitHub/STM_DataProcessing/` 下的任何文件。
**没有重新处理 topo4 数据**：三次真跑全部以 `-o`/`--no-figures` 落在 `_work/` 内；对已有矫正产物只做**只读**输入。
**没有改动相位口径**：估计量定义、门、r0 规范固定、图集 25/族契约均不变；`phase_stats.json` 只**新增**字段。

解释器与运行方式（任务指定）：

```bash
cd /Users/hunfen/Documents/GitHub/STM_DataProcessing
MPLCONFIGDIR=/Users/hunfen/Documents/论文/c6lic6/data_processing/_skill_fix/_work/mpl \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  /Users/hunfen/Documents/论文/c6lic6/data_processing/_skill_fix/scripts/<script>.py ...
```

---

## 1 改动点（逐项）

### F1 `scripts/stm_phase_analysis.py` — `--size-nm-from-log` 取错视场行

* 现象：`re.search(r"field of view ([\d.]+) nm", text)` 取 log 里**第一处**匹配，而矫正 log 的第一处是
  **输入画布**（topo4: `50 nm`）；脚本分析的却是**矫正后** CSV（矫正后画布 `51.7090 nm`）。
  nm/px、环半径、`nm^-1` 全部按 50/51.709 的比例错。
* 改法：
  * 新增 `FovMatch`（`namedtuple`）+ `parse_fov_log(text)`：按**行**扫描
    `field of view\s+([\d.]+)\s+nm(?P<tail>[^\n]*)`，逐条记下值、是否带 `corrected canvas` 标签、来源行；
  * `field_of_view(args)` 现在返回 `(value, provenance)`，优先级：
    **① 带 `corrected canvas` 标签的行 → ② 最后一处匹配（单行 log 即该行）→ ③ 报错退出**；
  * log 头部新增一行来源说明（紧跟在 `# input:` 之后、`# canvas:` 之前）：
    `# field of view from log: 51.7090 nm (corrected canvas line; <log>)`；
  * `phase_stats.json` 新增 `field_of_view_source`（主路径与「未检出 r3」早退路径都写）；
  * `-L/--size-nm` 仍以命令行值为准，且同样打印来源行（`(command line -L/--size-nm)`）；
  * `--size-nm-from-log` 的 `--help` 改写为描述该优先级（原文只写 "the line 'field of view …'"）；
  * `--size-nm-from-log` 指向不存在/不可读文件时给出明确 `SystemExit` 信息而不是 traceback。

### F2 `scripts/stm_topo_correct.py` — 锚定自检新增全局拉伸 tell-tale

* 现象：自检只看**矫正后两个最强环的半径比**。当**原始数据本身**就有精确 1 : √3 环对时，错锚定把整个
  拟合按 1/√3 全局拉伸，矫正后比值**仍是** √3 ⇒ 错锚被报成 `consistent`。
* 改法：
  * 新增常量 `STRETCH_SCALE_TOL = 0.05`（含理由注释）；
  * `anchor_self_check` 新增字段（**既有字段 `ratio`/`deviation`/`consistent`/`verdict`/`outer_radius_px`/
    `inner_radius_px`/`tolerance`/`n_rings` 原样保留**）：
    `stretch_scale_sqrt_det`、`stretch_scale_tolerance`、`stretch_scale_deviation`、
    `stretch_scale_consistent`、`ratio_consistent`、`verdict_reason`；
  * `|det M|^(1/2)` 偏离 1 超过容差 → 打 WARNING（说明"锚定环标错或视场 L 标错都会整体缩放拟合"）；
  * log 改为**两条 tell-tale 各打一行判词**，再打一行合并后的 `# anchor verdict = …`；
  * 合并规则：任一条失败 → `inconsistent`；比值无法构成（<2 环）但拉伸尺度合格 → `unverifiable`；
    两条都过 → `consistent`。
* **容差 5 % 的理由**：包自身口径是环半径聚类 3 % / 标签匹配 2 %；正确锚定时 `|det M|^(1/2)` 只含
  "要被 undo 的那点各向异性"（实测 1.00187，偏 0.19 %），错锚定给 1/√3 = 0.5774 或 √3 = 1.7321
  （实测 1x1 错锚偏 **42.16 %**、反演晶格常数偏 **+73.93 %**）——5 % 两侧都有 ~10× 以上的余量。

### F3 `scripts/atlas.py` — `qspace_mask` 图标注峰编号

* 现象：图里只有六个绿色 mask 圆，没有峰号；看图无法把它与 `p0…p5`（per-peak 图与
  `phase_stats.json` 的峰号）对应。
* 改法：圆圈循环改为 `enumerate(centres)`，每个圆在
  **圆心 + (mask_radius + 8) · 单位径向矢量**（径向 = 背离 FFT 中心 `(N/2, N/2)`）处画
  `p{i}`：`color="#39ff14"`（同圆框）、`fontsize=10`、`fontweight="bold"`、`ha="center"`、`va="center"`。
  编号 = `centres` 顺序 = `phase_stats.json` 的 peak index 0…5。
* 该图 `panels` 改为 `["FFT2 log amplitude with the ring masks (p0-p5 labelled at the circle rims)"]`，
  `SKILL.md` 图集表第 4 行同步；manifest / PNG `tEXt` / 注解由同一份代码生成，`atlas --check` 自洽。

### 文档

* `SKILL.md`：§2.1 补「两条 tell-tale + 5 % 容差理由 + topo4 实测（错锚 0.57843 仍比值 √3 → 现判
  inconsistent；正确 1.00187 → consistent）+ 新增 JSON 字段清单」；§3 新增 `--size-nm-from-log`
  取值优先级与来源打印段；§4 图集表第 4 行 panels 更新；§5 计数改 **72/72** 并新增第 11 项检查说明；
  §8 修改记录追加 v2.1。
* `CHANGES.md`：**顶部**新增「v2.1 修复」段（F1/F2/F3 三行缺陷→改动 + 5 行验证结果表 + 保留的已知行为）。
* `README.md`：文件表计数改 **72 项**；关键点 1 补拉伸 tell-tale，关键点 4 补 `p0–p5` 标注。
* `scripts/selftest.py`：**新增** `test_field_of_view_from_log()`（3 项检查，真跑三次子进程），
  计数 69 → 72。**没有放宽或删除任何既有断言**（v2 的 69 项一条未动）。

### 未改的东西（明确声明）

* `SKILL_VERSION` 保持 `"2.0"`：三处修复都没有改变任何既有报告数字或字段语义，图集契约也不变；
  版本叙事写在 `CHANGES.md`/`SKILL.md` 的 v2.1 段里。（改版本串会牵动 `log`/JSON/图的既有断言，
  收益不抵风险。如需真正 bump 版本号，请在同一批次里统一改 `scripts/*.py` + 文档 + 断言。）
* `correction_report.json` **顶层** `stretch_scale_sqrt_det`（v2 既有字段）保留；它与新增的
  `anchor_self_check.stretch_scale_sqrt_det` 同值双份是**有意**的：顶层那份不动既有契约，
  自检里那份让 `verdict` 与数字就地可读。

---

## 2 验证（全部真跑，产物保留在 `_work/`）

**汇总证据文件：[`_work/VERIFICATION.log`](_work/VERIFICATION.log)**（每条命令 + 关键输出行的一次性摘录）。

### a) 全量 selftest — 通过

```bash
cd /Users/hunfen/Documents/GitHub/STM_DataProcessing
MPLCONFIGDIR=.../_skill_fix/_work/mpl PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python .../_skill_fix/scripts/selftest.py --workdir .../_skill_fix/_work/selftest
```

* 实际：**退出码 0**、`72/72 checks passed`、`ALL CHECKS PASSED`（日志：`_work/selftest_final.log`，
  工作目录：`_work/selftest/`）。
* 其中覆盖：图集契约（每族 25 张 / 共 50 / 命名模式 / PNG 内嵌注解 = manifest / 注解数字 = JSON）、
  **`atlas --check` 真以命令行方式调用**（`exit code 0, stdout tail: ATLAS CHECK PASSED`）、
  两次运行逐像素相同 + JSON 相同、log 数字 = JSON 数字、锚定自检两条断言、
  新增的 `--size-nm-from-log` 三项（实测 51.5000 nm 取矫正后画布 / 单行 log 回退 50.0 nm / 无匹配非零退出）。
* 对照基线：v2.1 改动前的原版 skill 同一命令为 **69/69**（真跑过，未放宽任何断言）。

### b) FOV 回归 — log 头部显示 51.7090 nm

```bash
.../stm_phase_analysis.py \
  data_processing/topo4_50nm_analysis_mask10px/correction/20251117_topo4_30nm_corrected.csv \
  -o .../_skill_fix/_work/fov_check \
  --size-nm-from-log data_processing/topo4_50nm_analysis_mask10px/correction/correction.log \
  --no-figures --pct 0.95 --gate p50 --anchor auto
```

* 实际：**退出码 0**；`_work/fov_check/phase_stats.log` 头部

  ```
  # field of view from log: 51.7090 nm (corrected canvas line; .../correction.log)
  # canvas: 1059 x 1059 px, field of view 51.709 nm (0.048828 nm/px), valid pixels 93.65 %
  ```

  ⇒ 取到的是**矫正后画布**的 51.7090 nm，而不是输入画布的 50 nm；`phase_stats.json` 的
  `field_of_view_nm = 51.709`、`field_of_view_source` 同文（输入为只读，产物全部在 `_work/fov_check/`）。

### c) 锚定回归 — 错锚 inconsistent / 正确锚 consistent

```bash
.../stm_topo_correct.py data/20251117_topo4_30nm.csv -L 50 --a 0.246 \
  --anchor-ring {1x1|r3} -o .../_work/anchor_{wrong|right} --list-rings
```

| | `--anchor-ring 1x1`（错锚，`_work/anchor_wrong/`） | `--anchor-ring r3`（正确锚，`_work/anchor_right/`） |
| --- | --- | --- |
| 退出码 | 0 | 0 |
| `\|det M\|^(1/2)` | **0.57843**（偏 42.16 %；WARNING） | **1.00187**（偏 0.1868 %） |
| 矫正后两强环比值 | 1.731570（偏 0.0278 %）→ **ring-pair consistent** | 1.731497（偏 0.0320 %）→ ring-pair consistent |
| stretch 判词 | `stretch verdict = inconsistent` | `stretch verdict = consistent` |
| `anchor verdict` | **`inconsistent`**（`verdict_reason` 指全局拉伸：锚定环标错或 L 标错） | **`consistent`** |
| 反演 1x1 晶格常数 | 0.4279 nm（**+73.933 %**） | 0.2470 nm（+0.420 %） |
| 矫正后画布 / 视场 | 621² / 30.3223 nm | **1059² / 51.7090 nm**（与已有产物逐位一致） |

* 关键回归对照：`_work/anchor_right/correction_report.json` 与**已有交付产物**
  `data_processing/topo4_50nm_analysis_mask10px/correction/correction_report.json` 相比，
  34 个键里 **31 个逐位相同**，只有 `anchor_self_check`（新增字段）与两个自身路径字段
  （`corrected_csv` / `corrected_fft2_npy`，因为输出目录不同）不同 ⇒ **修复没有移动任何既有数字**。
* 错锚 log 第 34 行是新增的 WARNING、第 37 行是合并 verdict；`_work/anchor_wrong/correction.log`
  第 33 行仍显示 ring-pair consistent —— 正是"比值单独看会误报"的现场证据。

### d) `qspace_mask` 峰号标注 — 12/12 标签（两族 ×6）

两条独立证据：

1. **文本调用探针**（`atlas.qspace_mask` 的 `Axes.text` 被拦截记录）：6 次调用恰为
   `p0…p5`，位置与契约 `圆心 + (mask_radius + 8)·径向外单位矢量` **逐位相同（d = 0.00e+00）**，
   颜色 `#39ff14`、字号 10、`ha/va = center`，且到 FFT 中心距离 46.0 px > mask 半径 19 px（在圆外）。
2. **已渲染 PNG 的像素分析**（`_work/selftest/pipeline/out_{a,b}/RING_qspace_mask.png`）：
   每个图的 6 个圆框（连通域）与 6 个标签都检出；12/12 个标签落在对应圆的**径向外侧带**上，
   方向 = 该峰的径向，且不在圆内（`#39ff14` 像素中 6 个大连通域=圆框、其余小连通域=字形）。
   `panels`（manifest + PNG `tEXt`）逐字为
   `["FFT2 log amplitude with the ring masks (p0-p5 labelled at the circle rims)"]`，
   与 `SKILL.md` 图集表第 4 行一致，且 `atlas --check` 全部通过。

---

## 3 产物清单（staging 内）

| 路径 | 内容 |
| --- | --- |
| `scripts/stm_phase_analysis.py` | F1（`parse_fov_log` / `field_of_view` / 来源行 / `field_of_view_source`） |
| `scripts/stm_topo_correct.py` | F2（`STRETCH_SCALE_TOL` + 两条 tell-tale + 新 JSON 字段） |
| `scripts/atlas.py` | F3（`p{i}` 标注 + panels） |
| `scripts/selftest.py` | 新增 `test_field_of_view_from_log`（+3 项，共 72） |
| `SKILL.md` / `CHANGES.md` / `README.md` | 三处修改的文档同步 |
| `FIX_SUMMARY.md` | 本文件 |
| `_work/VERIFICATION.log` | 三条验证命令与关键输出行的一次性摘录（供快速复核） |
| `_work/selftest_final.log`、`_work/selftest/` | 全量自检日志与工作目录（50 张/族图集、manifest、JSON） |
| `_work/fov_check/` | FOV 回归产物（`phase_stats.log/.json/.csv`、`ring_candidates.csv` 等） |
| `_work/anchor_wrong/`、`_work/anchor_right/` | 两个锚定回归的 `correction.log` / `correction_report.json` / 修正图 / FFT2 |
| `_work/pristine/skill/` | **未改动**的原版 skill 拷贝（供对照复核） |
| `_work/mpl/` | `MPLCONFIGDIR` 可写目录 |

## 4 已知局限 / 备注

1. `unverifiable` 分支仍有旧语义：矫正后退化为不足两族环时无法确认锚定；此时若全局拉伸尺度也失败，
   判 `inconsistent`（更保守的方向）。
2. FOV 解析按**行内**子串 `corrected canvas` 判定，"矫正后画布"行是 `stm_topo_correct.py` 自己写的格式；
   第三方 log 若用别的措辞，会走"最后一处匹配"回退（对单行/末尾行 log 都正确）。
3. 标签字号 10 固定在 6.5 in 画布上；`--dpi` 改变只影响 PNG 分辨率，不影响几何。
