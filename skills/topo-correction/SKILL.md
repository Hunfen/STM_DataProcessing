---
name: topo-correction
description: STM 拓扑图 CSV 的几何矫正：正方形数值矩阵（可含 NaN）+ 视场 L（nm）→ 数据锚定峰检测、显式六方参考晶格与可指定锚定环（1x1 或 r3）、对称正定拉伸矫正（纯拉伸、零旋转）与锚定自检（1:√3 环对 + 全局拉伸尺度双 tell-tale），输出矫正 CSV/复数 FFT2/预览图/报告，可接任意后处理。当用户给出拓扑图 CSV 与图像边长（nm）并要求仿射矫正时使用。
---

# topo-correction（STM 拓扑图几何矫正，v2.1）

**范围**：本 skill 只做**几何矫正数学**——数据锚定峰检测、亚像素定位、对称正定拉伸矫正、
锚定自检。畸变在**倒空间**（FFT2）检测、在**实空间**用拟合拉伸的逆重采样。
**物理解释由使用者进行**：脚本、图、log 与文档都不给物理结论，也不用任何 k 空间/高对称点称呼。

**下游**：矫正产物是通用中间结果，可接**任意后处理**（例如双环相位数学分析见
`phase-analysis` skill，见 §3 接口契约）。

## 1 输入与执行环境

| 项 | 说明 |
| --- | --- |
| 输入 | 正方形拓扑数值矩阵 CSV/txt（可含 NaN）。**第一行 = 扫描起始行 = 图像顶部** |
| 视场 L (nm) | 正方形扫描边长，**必须显式给出**（文件名里的标注不可信） |
| 参考晶格 | 六方，1x1 晶格常数 `a`（默认 0.246 nm）；**取向取数据自身的 (1,0) 方向**，不做点群推断 |
| 锚定环 | 必须显式指定：`--anchor-ring 1x1`（默认）/ `r3`（见 §2.1） |
| 分隔符 | 自动识别；显式用 `--delimiter ','` / `--delimiter '\t'` |
| 解释器 | 仓库 venv：`cd /path/to/STM_DataProcessing && MPLCONFIGDIR=<可写目录> PYTHONDONTWRITEBYTECODE=1 .venv/bin/python <脚本>`。**不要用 `uv run`**（会写仓库 `.venv`） |
| 绘图 | `text.usetex=False`（mpl 3.10 + TeX Live 2026 有编码 bug），mathtext cm + Palatino |

## 2 矫正流程（`scripts/stm_topo_correct.py`）

```bash
cd /path/to/STM_DataProcessing
MPLCONFIGDIR=<可写目录> PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  skills/topo-correction/scripts/stm_topo_correct.py \
  INPUT.csv -L 50 -o OUT --a 0.246 --anchor-ring r3 --list-rings
```

流程：读入 → `flipud(subtractMeanPlane(...))` → 包内数据锚定峰检测（`bragg_peak.detect_bragg_peaks`）
→ 取数据自身 (1,0) 方向构造显式 `LatticeSpec` → `correct_bragg_peaks` 对称正定拉伸
（纯拉伸、零旋转，加权最小二乘，失败时闭式回退并记 `meta`）→ `scipy.ndimage.affine_transform`
重采样（`order=3`，`cval=NaN`），画布 `n_out`、**矫正后视场 = `L · n_out / n`**。

### 2.1 锚定环必须显式指定（v2 核心）

包内检测把**它认定的第一个环**标成 (1,0)。当 r3 环落在 1x1 环**内侧**（半径比 1/√3）时，
这个环可能正是 r3 环；若此时按 1x1 给参考晶格，拟合会把 r3 环拉到 1x1 的理想半径上，
**几何被破坏**。`--anchor-ring` 就是把这个事实说出来：

| 参数 | 含义 | 参考晶格 |
| --- | --- | --- |
| `--anchor-ring 1x1`（默认） | 检测锚定的环是 1x1 环 | `a_ref = a` |
| `--anchor-ring r3` | 检测锚定的环是 r3 环 | `a_ref = √3 · a` |

**报告**（log + `correction_report.json`）：`method` / `fallback` / `n_labelled` / rms、
检测到的环表（半径 px 与 nm⁻¹、成员数、总幅度）、`|b1|` 矫正前后与理想值的比、
`|det M|^(1/2)`、画布 `n_out`、矫正后视场与 nm/px、NaN 比例、锚定环反演的晶格常数
（`a_ref` 与 1x1 的 `a`）。

### 2.2 锚定自检（非循环，两条 tell-tale）

矫正只把**锚定环**放到理想半径上，所以"矫正后锚定环 = 理想值"不能证明锚定正确；判据是
**另一族**——正确的锚定会让矫正后两个最强环构成 1 : √3 对。
**但环对比值单独用是不够的**：若**原始数据本身**就有精确的 1 : √3 环对，错的锚定会把整个拟合按一个
全局因子拉伸（1/√3 或 √3），而矫正后两个最强环仍然相差 √3 ⇒ 只看比值会误报 consistent。
所以自检同时给出**全局拉伸尺度 `|det M|^(1/2)`**：锚定环或视场 `L` 标错都会让拟合成比例地整体缩放，
正确锚定的 `|det M|^(1/2)` 只含"要被 undo 的那点各向异性"（≈1），错锚定则给出 0.5774 / 1.7321。
判定：两条 tell-tale 都通过才 `consistent`；任一条失败即 `inconsistent`（**比值通过、拉伸尺度失败也是
inconsistent**）；矫正后退化为不足两环时，若拉伸尺度仍在容差内则 `unverifiable`。容差**5 %**
（`STRETCH_SCALE_TOL`：超出 `[1/1.05, 1.05]` 即失败；包自身口径是环聚类 3 % / 标签匹配 2 %，
5 % 两侧都有余量——正确实测 ≤2 %，错锚实测偏 42 %/73 %）。log 打印每条 tell-tale 的判词与
合并后的 `anchor verdict`，`correction_report.json` 的 `anchor_self_check` 里除原有字段外还有
`stretch_scale_sqrt_det` / `stretch_scale_deviation` / `stretch_scale_consistent` /
`ratio_consistent` / `verdict_reason`。
实测（50 nm 视场真实数据，raw 三环本身是 1 : 1/√3 : 1/2）：正确锚定 `--anchor-ring r3`
→ `|det M|^(1/2) = 1.00187`（偏 0.1868 %）、比值 1.731497（偏 0.0320 %）→ consistent；
错锚定 `--anchor-ring 1x1` → `|det M|^(1/2) = 0.57843`（≈1/√3，偏 42 %）、反演 1x1 晶格常数
偏离 +73.9 %，**而环对比值仍是 1 : √3** ⇒ 现在判 inconsistent 并打 WARNING（旧版此处误报 consistent）。
另一组几何下矫正后退化为不足两环 → unverifiable。

**已知限制**：矫正需要先有**带标签**的晶格；环半径聚类容差 3 %、`match_labels` 容差 2 %
决定了可用范围 ≈ 各向异性 ≤ 3 %。超出时包返回 `method="identity_fallback"`、`fallback=True`，
**什么都没矫正**（数组只是重铺到更大的 NaN 画布上）——脚本会照实打印并打 WARNING，
**信任结果前先看 `method`/`fallback`**。

## 3 输出与下游接口契约

输出（`-o OUT`）：`<stem>_corrected.csv`、`<stem>_corrected_fft2.npy`（复数 FFT2，含 Hanning 窗与
NaN 平面填充）、`<stem>_corrected.png`（gwyddion）、`<stem>_corrected_fft.png`（inferno 对数）、
`correction.log`、`correction_report.json`。

**下游接口契约（跨 skill，勿改）**：

| 项 | 约定 |
| --- | --- |
| 矫正后画布视场 | `correction.log` 必须含一行 `# corrected canvas: <n_out> x <n_out> px, field of view <X> nm (...)`。`phase-analysis` 的 `--size-nm-from-log` 优先解析这一行；其他后处理也应读它而非输入视场 |
| 输入画布视场 | log 的 `# canvas ... field of view <L> nm` 行（原始 L，供对照） |
| `affine_q` | `correction_report.json` 里拟合出的对称正定拉伸 M：把**测得**倒格矢映射到**理想**倒格矢；实空间矫正重采样用的是 M⁻¹（数组坐标 `A = P·M⁻¹·P`，`P` 为轴交换） |
| `anchor_self_check` | `verdict` ∈ `consistent` / `inconsistent` / `unverifiable` + 两条 tell-tale 的字段（见 §2.2） |

**把变换应用到其他图像**：当前**没有 CLI 开关**（拟合与应用于同一张图一步完成）；`affine_q`、
`n_out`、`correction_report.json` 已完整保存变换，手工对另一张图做同样重采样即可——该功能是
拆分后可选的后继任务，不在本 skill 范围内。

## 4 一键 self-test（5 项）

```bash
cd /path/to/STM_DataProcessing
MPLCONFIGDIR=<可写目录> PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python skills/topo-correction/scripts/selftest.py
# 可选：--workdir DIR、--keep
```

覆盖与验收阈值（实测 **5/5** 通过，约 1 min）：

| # | 检查 | 验收阈值 |
| --- | --- | --- |
| 1 | 矫正 stage 跑通（`--anchor-ring r3` 与 `1x1` 两次都产出 `correction_report.json`） | 报告写出 |
| 2 | 注入 2 % 各向异性（`diag(1.010, 0.990)`）被回收 | `affine_q` 与注入矩阵逐项差 < 3e-3（实测 `[[1.01000, −0.00000], [−0.00000, 0.98998]]`） |
| 3 | 锚定环反演的 1x1 晶格常数回来 | 与标称 `a = 0.246 nm` 偏差 < 1 % |
| 4 | 正确锚定（r3）通过自检 | `anchor_self_check.verdict == consistent`（实测比值 1.734029，偏 0.1142 %） |
| 5 | 错误锚定（1x1）被同一自检抓住 | `verdict != consistent`（实测比值 1.871416，偏 8.05 %） |

包不可导入时检查 1 以 informational 跳过（同原 self-test 的分支）。

## 5 修改记录

2026-09（拆分，本版）：从 `stm-topo-phase-analysis` 拆出独立 skill（目录
`skills/topo-correction/`）；`stm_topo_correct.py` 的兄弟依赖（`group_rings`、样式/色图 helper）
内聚为 `correction_lib.py`；自带 5 项 self-test。输出与 log 格式逐字节不变（跨 skill 契约）。
2026-09（v2.1）：锚定自检新增**全局拉伸尺度 tell-tale**（错锚不再误报 consistent），见 `CHANGES.md`。
2026-09（v2）：锚定环显式化（`--anchor-ring`），见 `CHANGES.md`。
