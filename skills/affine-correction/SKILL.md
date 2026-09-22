---
name: affine-correction
description: STM 拓扑图 CSV 的几何矫正：正方形数值矩阵（可含 NaN）+ 视场 L（nm）→ 数据锚定峰检测、显式六方参考晶格与可指定锚定环（1x1 或 r3）、对称正定拉伸矫正（纯拉伸、零旋转）与锚定自检（1:√3 环对 + 全局拉伸尺度双 tell-tale），输出矫正 CSV/HDF5 数组（corrected + fft2）/复数 FFT2 npy/预览图/报告，可接任意后处理；并支持把拟合出的变换导出为独立 JSON（--save-transform）后应用到另一张拓扑图（stm_apply_transform.py）。当用户给出拓扑图 CSV 与图像边长（nm）并要求仿射矫正、或要求把某张图的矫正变换套用到另一张图时使用。
---

# affine-correction（STM 拓扑图几何矫正，v2.2）

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
export MPLCONFIGDIR=<可写目录> PYTHONDONTWRITEBYTECODE=1

# 第一段：在参考图上拟合（并导出变换 JSON）
.venv/bin/python skills/affine-correction/scripts/stm_topo_correct.py \
  INPUT.csv -L 50 -o OUT --a 0.246 --anchor-ring r3 --list-rings \
  --save-transform OUT/transform.json

# 第二段（可选）：把同一变换应用到另一张拓扑图，-L 给的是那张图自己的视场
.venv/bin/python skills/affine-correction/scripts/stm_apply_transform.py \
  TARGET.csv --transform OUT/transform.json -L 50 -o OUT_TARGET
```

流程：读入 → `flipud(subtractMeanPlane(...))` → 包内数据锚定峰检测（`bragg_peak.detect_bragg_peaks`）
→ 取数据自身 (1,0) 方向构造显式 `LatticeSpec` → `correct_bragg_peaks` 对称正定拉伸
（纯拉伸、零旋转，加权最小二乘，失败时闭式回退并记 `meta`）→ `scipy.ndimage.affine_transform`
重采样（`order=3`，`cval=NaN`），画布 `n_out`、**矫正后视场 = `L · n_out / n`**。
加 `--save-transform FILE` 时，拟合结束后额外把该变换写成独立 JSON（契约见 §3）；
**不加这个开关时所有既有输出（log / 报告 / csv / npy / png）逐字节不变**。

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

输出（`-o OUT`）：`<stem>_corrected.csv`、`<stem>_corrected.h5`、`<stem>_corrected_fft2.npy`
（复数 FFT2，含 Hanning 窗与 NaN 平面填充）、`<stem>_corrected.png`（gwyddion）、
`<stem>_corrected_fft.png`（inferno 对数）、`correction.log`、`correction_report.json`。

**HDF5 数组产物**（`<stem>_corrected.h5`，仓库统一 h5 约定，见 `docs/hdf5_convention.md`）：

| 数据集 | 内容 | `units` |
| --- | --- | --- |
| `corrected` | `float64`，**与写进 CSV 的是同一个数组** | 无（输入拓扑是无量纲/任意单位，仓库约定无量纲量不写 `units`） |
| `fft2` | `complex128`，**与 `<stem>_corrected_fft2.npy` 是同一个数组** | 无（同上） |

文件根部属性：`schema_version = 1`、`generator = "stm_topo_correct.py"`（apply 段为
`"stm_apply_transform.py"`）、`creation_date`（ISO-8601 带时区偏移）。每个数据集用显式 chunk
（从末维起填满 ≤ 1 MiB 未压缩字节）且 `compression = "gzip"`、`compression_opts = 4`、
`track_times = False`。写入脚本是本 skill 自包含的 `scripts/h5io.py`（skill **不** import
`stm_data_processing`）。**`.npy` FFT2 保留**：`skills/phase-analysis/` 通过它的 `--fft2` 选项
读这个文件，phase-analysis 不在本次产物迁移范围内。

**下游接口契约（跨 skill，勿改）**：

| 项 | 约定 |
| --- | --- |
| 矫正后画布视场 | `correction.log` 必须含一行 `# corrected canvas: <n_out> x <n_out> px, field of view <X> nm (...)`。`phase-analysis` 的 `--size-nm-from-log` 优先解析这一行；其他后处理也应读它而非输入视场 |
| 输入画布视场 | log 的 `# canvas ... field of view <L> nm` 行（原始 L，供对照） |
| `affine_q` | `correction_report.json` 里拟合出的对称正定拉伸 M：把**测得**倒格矢映射到**理想**倒格矢；实空间矫正重采样用的是 M⁻¹（数组坐标 `A = P·M⁻¹·P`，`P` 为轴交换） |
| `anchor_self_check` | `verdict` ∈ `consistent` / `inconsistent` / `unverifiable` + 两条 tell-tale 的字段（见 §2.2） |

**把变换应用到其他图像（两段式，v2.2）**：`--save-transform FILE` 把拟合出的变换写成独立 JSON
（schema `topo-correction-transform`，`schema_version = 1`；**该 schema id 是冻结的产物标识，本 skill 改名后保持不变**），
`scripts/stm_apply_transform.py` 再把它应用到**指定的另一张**拓扑图：

```bash
.venv/bin/python skills/affine-correction/scripts/stm_apply_transform.py \
    TARGET.csv --transform OUT/transform.json -L <TARGET 自己的视场 nm> \
    [-o OUT_TARGET] [--delimiter ','] [--stm-lib DIR]
```

apply 段只做**重采样**：与拟合段同样的读入/分隔符自动识别与预处理
（`flipud(subtractMeanPlane(...))`）→ 用**目标图自己的像素数 n** 重算数组矩阵
`A = P·M⁻¹·P`、画布 `n_out = 2·(⌈max|A⁻¹·角点|⌉ + pad) + 1` 与 offset（`order` / `pad` 取自 JSON）
→ `scipy.ndimage.affine_transform`（`cval=NaN`）→ 输出与拟合段同名的产物
（`<stem>_corrected.csv`、`_corrected_fft2.npy`、`_corrected.png`、`_corrected_fft.png`）、
`correction.log`（含上面的契约行）与 `apply_report.json`。
**不重新检测峰、不重跑锚定自检**；矫正后视场 `= L · n_out / n`。
`method == "identity_fallback"` 时（拟合段本来就没能矫正）目标图**原样复制**：
`n_out = n`、视场不变、不加 NaN，log 里打 WARNING。

| transform JSON 字段 | 含义 |
| --- | --- |
| `schema` / `schema_version` | `topo-correction-transform` / `1`；apply 段校验，不匹配或 `affine_q` 非有限 2×2 / `det ≤ 0` 时打印原因并以非零码退出 |
| `affine_q` | 拟合出的对称正定拉伸 M（apply 段只用它重算几何） |
| `affine_image_reference` / `n_px_reference` / `n_out_reference` / `field_of_view_nm_reference` | 参考图的数组矩阵、像素数、画布与视场（记录用） |
| `pad` / `order` | 重采样边距与样条阶数（apply 段直接采用） |
| `method` / `fallback` / `n_labelled` / `anchor_ring` / `a_nm` / `a_ref_nm` / `stretch_scale_sqrt_det` / `anchor_verdict` | 拟合来源与可信度，原样透传进 apply 的 log 与 `apply_report.json` |
| `source_report` / `source_input` | 拟合段 `correction_report.json` 与参考输入图的绝对路径 |
| `usage` | 一段命令行使用提示字符串 |

`apply_report.json` 的字段：`skill`、`stage`、`input`、`transform_file`、`source_report`、`source_input`、
`method`、`fallback`、`anchor_verdict`、`field_of_view_nm`、`n_px`、`affine_q`、`matrix_used`、
`offset`、`n_out`、`corrected_field_of_view_nm`、`corrected_nm_per_px`、`nan_fraction` 与
`written`（六个产物路径）。注意 `anchor_verdict` 是**参考图**的结论：apply 段不做自检，
是否可信取决于参考图与目标图是否可比。

## 4 一键 self-test（8 项）

```bash
cd /path/to/STM_DataProcessing
MPLCONFIGDIR=<可写目录> PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python skills/affine-correction/scripts/selftest.py
# 可选：--workdir DIR、--keep
```

覆盖与验收阈值（实测 **8/8** 通过，约 1.5 min）：

| # | 检查 | 验收阈值 |
| --- | --- | --- |
| 1 | 矫正 stage 跑通（`--anchor-ring r3` 与 `1x1` 两次都产出 `correction_report.json`） | 报告写出 |
| 2 | 注入 2 % 各向异性（`diag(1.010, 0.990)`）被回收 | `affine_q` 与注入矩阵逐项差 < 3e-3（实测 `[[1.01000, −0.00000], [−0.00000, 0.98998]]`） |
| 3 | 锚定环反演的 1x1 晶格常数回来 | 与标称 `a = 0.246 nm` 偏差 < 1 % |
| 4 | 正确锚定（r3）通过自检 | `anchor_self_check.verdict == consistent`（实测比值 1.734029，偏 0.1142 %） |
| 5 | 错误锚定（1x1）被同一自检抓住 | `verdict != consistent`（实测比值 1.871416，偏 8.05 %） |
| 6 | 两段式变换（单条检查，内部 (a)–(e) 子步骤）：(a) `--save-transform` 导出的 JSON 20 个字段齐全、`affine_q` 与 `correction_report.json` 逐位一致；(b) 同一参考图 apply → `_corrected.csv` 与拟合段一致（`np.allclose rtol=atol=1e-10`，NaN 掩码相同且实测逐字节相同）；(c) 换尺寸目标（生成器 160 → 画布 183 px）的 `n_out` 按**目标 n** 重算（期望 205，与参考图 435 不同）、log 契约行视场 = `L·n_out/n`、`apply_report.json` 写出；(d) 本地 `_image_transform` 镜像与包内 `bragg_peak.correct._image_transform` 在矩阵 / `n_out` / offset 上逐位一致；(e) `identity_fallback` 变换 → 目标原样复制且 log 含 WARNING | (a)–(e) 全部通过 |
| 7 | 矫正 stage 的 `<stem>_corrected.h5`：root 三属性、`generator = "stm_topo_correct.py"`、`gzip`/4、显式 chunk = 约定 chunk、`track_times=False` | 约定完全符合；`fft2` 与 `.npy` `np.array_equal`（`complex128`）、`corrected` 用 `%.10e` 重渲染后与 CSV 文本逐字符相同（`float64`） |
| 8 | 两段式两个 stage 的 h5 产物：拟合段同上；同网格 apply 段的 `corrected`/`fft2` 与拟合段**逐位相同**（dtype 相同）；换尺寸与 `identity_fallback` 两段也写出合规 h5（`generator = "stm_apply_transform.py"`） | 两个 stage 的 h5 都合规 + 同网格逐位相同 |

包不可导入时检查 1、6、7、8 各自以 informational 跳过（同原 self-test 的分支）。

## 5 修改记录

2026-09（HDF5 数组产物，本版）：矫正数组改为 **HDF5 优先**——`stm_topo_correct.py` 与
`stm_apply_transform.py` 额外写出 `<stem>_corrected.h5`（数据集 `corrected` = 写进 CSV 的数组、
`fft2` = 写进 `.npy` 的数组），由本 skill **自包含**的新模块 `scripts/h5io.py` 按仓库 h5 约定写出
（`schema_version = 1`、`generator`、`creation_date`、显式 chunk ≤ 1 MiB + `gzip`/4、
`track_times=False`；两个数组无物理单位故不写 `units`）。**CSV/PNG/JSON/log 的名称与字节全部不变**
（跨 skill 契约与 `affine_q` 等字段不受影响），**`.npy` FFT2 保留**供
`skills/phase-analysis/` 的 `--fft2` 使用（phase-analysis 不在本次迁移范围）。self-test 6 → 8 项
（新增两条 h5 检查：约定合规 + 数组逐位一致）。见 `CHANGES.md`。
2026-09（改名，本版）：skill 由 `topo-correction` 改名为 `affine-correction`——目录、frontmatter
`name:`、文档与用法示例脚本路径同步改名；**运行时行为零变化**：产物 schema id
`topo-correction-transform`、报告 `skill` 字段值 `"topo-correction"`、log/stdout 行与 CLI help
文本逐字节不变（改名只动名字，不动产物）。本条目之前的记录保留当时的旧名 `topo-correction`。
见 `CHANGES.md`。
2026-09（v2.2）：新增两段式变换——`stm_topo_correct.py --save-transform` 导出独立变换 JSON +
新脚本 `stm_apply_transform.py` 应用到另一张图；self-test 5 → 6 项。不加开关时既有输出逐字节不变。
见 `CHANGES.md`。
2026-09（拆分，本版）：从 `stm-topo-phase-analysis` 拆出独立 skill（目录
`skills/topo-correction/`）；`stm_topo_correct.py` 的兄弟依赖（`group_rings`、样式/色图 helper）
内聚为 `correction_lib.py`；自带 5 项 self-test。输出与 log 格式逐字节不变（跨 skill 契约）。
2026-09（v2.1）：锚定自检新增**全局拉伸尺度 tell-tale**（错锚不再误报 consistent），见 `CHANGES.md`。
2026-09（v2）：锚定环显式化（`--anchor-ring`），见 `CHANGES.md`。
