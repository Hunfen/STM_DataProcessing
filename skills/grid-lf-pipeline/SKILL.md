---
name: grid-lf-pipeline
description: 把 Nanonis grid spectroscopy（`.3ds`，或经 `stm_data_processing.io.grid2h5` 转出的 `.h5`）一站跑完「抽 z map → 三步预处理 → Lawler–Fujita 位移场拟合 → 用该位移场矫正每个 bias 帧的 current map 与 dI/dV map」：`scripts/grid_topo.py` 从 `/params` 的 `Z (m)` 列抽 topo、逐行去背景、减二维最小二乘平面、平移 min 到 0，写成 lawler-fujita-correction 的输入 CSV；`scripts/grid_lf_all_maps.py` 一条命令接着调用 `stm_lf_correct.py ... --save-transform` 与 `scripts/apply_lf_transform.py`，把 h5 里每个 bias 帧的 current/dI/dV 全部矫正到 `<out>/current_corrected/` 与 `<out>/didv_corrected/`。当用户给出 grid 的 `.3ds`/`.h5` 并要求矫正它的 dI/dV 或 current map 时使用。
---

# grid-lf-pipeline（grid → Lawler–Fujita 全流程，v1.0）

**范围**：本 skill 只负责「从 grid 文件取数据 → 预处理 topo → 调 Lawler–Fujita 拟合 →
把位移场套用到所有 map」这条流水线；矫正数学、重采样与拟合产物契约都在
`lawler-fujita-correction` skill 里（本 skill 只调用它的脚本，不重写任何重采样）。
**脚本、log 与本文档都不给物理结论。**

**分工**：`lawler-fujita-correction` = 拟合位移场 + 重采样实现；
本 skill = 面向 grid 数据的前后处理与批量套用。

## 1 输入与执行环境

| 项 | 说明 |
| --- | --- |
| 输入 | `<grid>.h5`，由 `python -m stm_data_processing.io.grid2h5 <grid>.3ds -o <grid>.h5` 得到；也可以直接把 `.3ds` 交给脚本，脚本先做这次转换并**打印该转换命令** |
| h5 结构 | `/data (ny, nx, nch, npts)`、`/channels`、`/channel_units`、`/bias`（V）、`/params` + `/param_columns`、`/header` |
| 视场 L (nm) | 从 h5 header 的 `Scan>Scanfield` 解析（没有时退回 header 的 `Grid settings`）：两者都是 `x0;y0;width;height;angle`（**单位 m**，`;` 分隔）的同一记录，取第 3 段宽 ×1e9=nm；`-L` 可显式覆盖 |
| 解释器 | 仓库 venv：`cd /path/to/STM_DataProcessing && MPLCONFIGDIR=<可写目录> PYTHONDONTWRITEBYTECODE=1 .venv/bin/python <脚本>`（不要用 `uv run`） |
| 依赖 | `h5py`、`numpy`、matplotlib（经 `stm_data_processing.utils.plot_funcs`），以及本仓库的 `skills/lawler-fujita-correction/scripts` 与 `scripts/apply_lf_transform.py` |

## 2 四步流程

第 2.1–2.4 步可以逐条手跑，也可以用 §2.5 的一条命令整条跑完（推荐，产出布局见 §5）。

### 2.1 归位：`.3ds` → `.h5`

```bash
cd /path/to/STM_DataProcessing
.venv/bin/python -m stm_data_processing.io.grid2h5 GRID.3ds -o GRID.h5
```

### 2.2 抽 z map + 三步预处理（`scripts/grid_topo.py`）

```bash
.venv/bin/python skills/grid-lf-pipeline/scripts/grid_topo.py GRID.h5 -o TOPO.csv [-L 30]
```

处理链（`grid_topo.preprocess`，顺序固定，逐字如下）：

```
z = np.flipud(params['Z (m)'].reshape(ny, nx))   # ⓪ 转到 top-down（见 §4 第 4 条）
z -= np.median(z, axis=1, keepdims=True)     # ① 逐行去背景（axis=1 = 快轴，逐慢扫行）
z  = subtractMeanPlane(z.T).T                # ② 减二维最小二乘平面（仓库 helper，NaN 安全）
z -= z.min()                                 # ③ 平移最小值到 0
np.savetxt(TOPO.csv, z, delimiter=",", fmt="%.10e")
```

② 用仓库的 `stm_data_processing.utils.plot_funcs.subtractMeanPlane`；`.T` 不改变拟合出的平面
（基底 `{1, i, j}` 两种写法张成同一张平面），只改变平面求值的末位舍入——转置调用是与
已确认的参考 topo CSV 在**撤销 ⓪ 的约定翻转之后**逐字节一致的那一种（§6）。

### 2.3 Lawler–Fujita 拟合 + 位移场包

```bash
.venv/bin/python skills/lawler-fujita-correction/scripts/stm_lf_correct.py \
    TOPO.csv -L 30 -o OUT/lf --save-transform OUT/lf/<stem>_transform.json
```

`<stem>` 是 topo CSV 的文件名词干；`--save-transform` 会把可迁移包写成
`<stem>_transform.json` + 同名的 `<stem>_transform.h5`（u 场）。本流程固定用
`--lambda-nm 3`：LF 自己的默认 30 nm 对 30 nm 视场太宽，`correction.log` 会报
`band margin ... OUTSIDE the low-pass radius` 的 WARNING；用 3 nm 时同一行报
`band margin lambda x offset = 0.418`（判据要求 < 1），落在低通半径之内。
**这条 WARNING 是必看的**：出现就说明提取出的相位/位移场不是晶格畸变，应调
`--lambda-nm` 重跑，不要拿这份位移场去矫正数据。

### 2.4 把位移场套用到所有 current / dI/dV map

对**每个 bias 帧**的 current map 与 dI/dV map 各跑一次
`scripts/apply_lf_transform.py`（重采样只由它做）：

```bash
.venv/bin/python scripts/apply_lf_transform.py MAP.csv \
    --transform OUT/lf/<stem>_transform.json -L 30 -o OUT/current_corrected
```

输出 `<stem>_corrected.csv`。文件名约定：`<i>_<bias>meV`（`i` = `/bias` 帧序号，`bias` 为
mV 值，例如 `0_30meV`、`3_-5meV`）。

**手工跑这一条时要自己把朝向翻回来**：applier 内部把输入翻进 u 场的索引帧、产物也留在那一帧，
所以按本 skill 的 top-down 约定，落盘前要对产物再做一次 `np.flipud`（`grid_lf_all_maps.py`
就是这么做的，见 §4 第 4 条）；喂给它的 MAP.csv 也必须先是 top-down 的（与 §2.2 的 TOPO.csv 同约定）。

### 2.5 一条命令跑完 §2.2–§2.4（`scripts/grid_lf_all_maps.py`）

```bash
cd /path/to/STM_DataProcessing
.venv/bin/python skills/grid-lf-pipeline/scripts/grid_lf_all_maps.py GRID.h5 -o OUT [-L 30] [--lambda-nm 3] \
    [--current-ch 'Current (A)'] [--didv-ch 'DSP 7280 Y (%)']
```

（`--lambda-nm` 默认 3 nm，见 §2.3。）它把每条实际执行的命令都打印出来（含 `.3ds→h5` 的转换命令），产出见 §5。

## 3 通道归属规则

通道一律**按 h5 `/channels` 的名字**解析（代码里没有字面量通道索引，索引只在
`list.index(name)` 查到名字之后得出）；一个 forward 通道总是连同它的后扫镜像一起取——
Nanonis 在名字的单位前插 `' [bwd]'`：`Current (A)` ↔ `Current [bwd] (A)`。

| 量 | 默认（自动） | 显式覆盖 |
| --- | --- | --- |
| current | 名字含 `Current` 的 forward 通道 + 其 `[bwd]` 镜像，两方向取平均（本例 `Current (A)` + `Current [bwd] (A)`） | `--current-ch 'Current (A)'`（写一个 forward 名字即自动带上它的 `[bwd]` 镜像；也可写全两个名字，或写 0-based 索引） |
| dI/dV | 由 header 判定：`Lock-in>Lock-in status = OFF` → **外部锁相**的 Y 通道 + 其 `[bwd]` 镜像（本例 `DSP 7280 Y (%)` + `DSP 7280 Y [bwd] (%)`）；`= ON` → 内建 `LI Demod 1 Y (A)` + `[bwd]` 镜像 | `--didv-ch 'DSP 7280 Y (%)'` |

**什么情况下必须显式指定**：`Lock-in status = OFF` 时外部锁相通道的名字是装置相关的
（本例是 `DSP 7280 Y (%)`），自动规则只按「非内建锁相的 Y 输出」找候选。因此：

* 自动规则**找不到候选**、或**找到多于一条候选**时，脚本**不会静默挑一条**——退出码 1，
  在 **stderr** 打印 `/channels` 全表与可复制的 `--didv-ch '<名字>'` 提示，让你点名；
* `--current-ch` / `--didv-ch` 是**第一类路径**：名字按 `/channels` 精确匹配（顺带接受
  0-based 索引），写一个 forward 名字就会带上它的 `[bwd]` 镜像。

## 4 四条已知坑

1. **h5 `/channels` 里没有 Z**。针尖高度不在通道里，而在 `/params` 的 `Z (m)` 列
   （本例 `param_columns` = `Sweep Start, Sweep End, X (m), Y (m), Z (m), Z offset (m),
   Z-Ctrl hold, Final Z (m)`），`reshape(ny, nx)` 才是 topo。`/channels` 里那 14 条
   是电流 / 偏压 / 外部锁相 / 内建锁相的通道。
2. **z 要在源头翻一次，之后全链都用 top-down**。`/params['Z (m)'].reshape(ny, nx)` 取出来的
   是 3ds 的原生顺序（自下而上），`grid_topo.load_z_map` 对它做**一次** `np.flipud` 再进入
   预处理，所以落盘的 TOPO.csv 是 top-down（见第 4 条）。`scripts/apply_lf_transform.py`
   自己还会对**待套用的 map** 做一次 `flipud`（top-down → u 场的数组序）——那一次是 applier
   内部的事、不要「统一」掉，而它的输出处在 u 场帧、要由调用方翻回（见第 4 条）。
3. **`scripts/apply_lf_transform.py` 只做 `flipud` + 重采样，不做任何值域预处理**：
   不减拟合平面、不去均值、不归一化。所以 current/dI/dV 自己的 DC 电平与倾斜被原样保留
   （这正是不用 `stm_lf_apply.py` 的原因：后者会对 target 减一次拟合平面，那是为 topo
   自对拍准备的）。若下游需要去平面，请在矫正**之后**自己做。
4. **朝向约定：所有 grid 产品统一 top-down（跟 sxm 一致）**。事实依据读仓库代码即可确认：
   `stm_data_processing/io/nanonis_loader.py` 的 **3ds 路径**用**同一个扁平像素索引 `n`**
   把数据写进 `params.loc[n]` 与 `grid[n]`，两边**都不翻转** ⇒ h5 的 `/data` 与 `/params`
   **同帧**（都是 3ds 的原生"自下而上"顺序，`grid2h5` 的 `reshape(ny, nx, nch, npts)` 也只
   按 `n` 切行）；而 **sxm 路径** `_reform_sxm_data` 才是做归一的那一侧——它对后扫行
   （奇数行）做 `fliplr`，并在 `SCAN_DIR = up` 时对每行做 `flipud`，即 sxm 出来是
   **top-down**。用户要求 grid 的结果跟 sxm 对齐（其 notebook 的 `grid_slice` 里本来就有
   `cp.flip(slice_data, axis=2)` 作人工补偿），所以本 skill 在**源头翻一次**、整条链统一
   top-down：topo CSV（`grid_topo.py`）→ raw map（驱动写盘前 `np.flipud`）→ 矫正图
   （applier 的输出处在 u 场帧，落盘前 `np.flipud` 回 top-down）。
   **注意这不是修帧错**：`/data` 与 `/params` 本来就同帧，原来的流程（未翻转的 map 直接喂
   applier）在帧上是自洽的；两次 flip 只是把整条链的产品换成 top-down 约定。

## 5 产物布局

`grid_lf_all_maps.py -o OUT` 产出：

```
OUT/<stem>_topo.csv                      预处理后的 topo（LF 的输入）
OUT/lf/<stem>_topo_transform.json        可迁移包（位移场）
OUT/lf/<stem>_topo_transform.h5          u 场本体
OUT/lf/<stem>_topo_corrected*.{csv,h5,npy,png} + <stem>_topo_lf*.{h5,png} + correction.log + correction_report.json
OUT/current/<i>_<bias>meV.csv            current 原始（通道平均）矩阵
OUT/didv/<i>_<bias>meV.csv               dI/dV 原始（通道平均）矩阵
OUT/current_corrected/<i>_<bias>meV_corrected.csv
OUT/didv_corrected/<i>_<bias>meV_corrected.csv
```

`OUT/<stem>_topo.csv`、`OUT/current/*`、`OUT/didv/*` 与 `OUT/*_corrected/*` 全部处在
**top-down 约定**（与 sxm 一致，见 §4 第 4 条；`OUT/lf/<stem>_topo_corrected.csv` 是 LF
拟合段产物、处在 u 场帧，不在这条约定里），因此该约定下的原始图与矫正图可以逐点比较。

## 6 自检口径

```bash
# topo 与已确认的参考输入只差一次约定翻转（⓪ 的 np.flipud）：两者必须互为 flipud
.venv/bin/python skills/grid-lf-pipeline/scripts/grid_topo.py GRID.h5 -o OUT/grid1_20251110_topo.csv
.venv/bin/python - <<'PY'
import numpy as np
new = np.loadtxt("OUT/grid1_20251110_topo.csv", delimiter=",")
old = np.loadtxt("data_processing/20251110/20251110_grid1_30mV500pA_30nm/grid1_20251110_topo.csv",
                 delimiter=",")
print("max|flipud(new) - old| =", np.abs(np.flipud(new) - old).max())
PY

# 一条命令整链跑通，产物计数 11（5 current + 5 dI/dV + 1 transform）
.venv/bin/python skills/grid-lf-pipeline/scripts/grid_lf_all_maps.py GRID.h5 -L 30 -o OUT/run
ls OUT/run/current_corrected/*_corrected.csv OUT/run/didv_corrected/*_corrected.csv OUT/run/lf/*_transform.json | wc -l
```

参考 topo CSV（`.../grid1_20251110_topo.csv`）是 **top-down 约定落地之前**的产物，
sha256 不再相同是预期的；判据是上式的 `flipud` 关系（实测见 PATCH2.md §4 的 R3(c)）。
