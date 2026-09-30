---
name: grid-lf-pipeline
description: 把 Nanonis grid spectroscopy（`.3ds`，或经 `stm_data_processing.io.grid2h5` 转出的 `.h5`）一站跑完「3ds→h5 → 抽 grid topo（三步预处理）→ Lawler–Fujita 位移场拟合并导出可迁移位移场包」：`scripts/grid_topo.py` 从 `/params` 的 `Z (m)` 列抽 topo（flipud → 逐慢扫行减 median → 减二维最小二乘平面 → min 移到 0），写成 lawler-fujita-correction 的输入 CSV；`scripts/grid_lf_fit.py` 接着调用 `stm_lf_correct.py ... --save-transform`，把位移场包落到 `<out>/lf/`。**本 skill 不把位移场套用到任何其它 map**（那是配套 skill B 的职责）。当用户给出 grid 的 `.3ds`/`.h5` 并要求抽出 topo、拟合 Lawler–Fujita 位移场、导出可迁移位移场包时使用。
---

# grid-lf-pipeline（grid → topo → Lawler–Fujita 位移场拟合，v2.0）

**范围**：本 skill 只负责以下**恰好三步**：① `.3ds`（或已是 `.h5`）→ `.h5`；② `.h5` → LF 输入 topo CSV
（固定三步预处理）；③ Lawler–Fujita（LF）位移场拟合并导出可迁移位移场包。
矫正数学、重采样与拟合产物契约都在 `lawler-fujita-correction` skill 里（本 skill 只按相对路径嵌套调用
它的 `stm_lf_correct.py`，不重写任何拟合/重采样）。**本 skill 不含任何「把位移场套用到其它 map」的逻辑**
（那是 Skill B 的职责），也不修改 `skills/lawler-fujita-correction` 下的任何文件。脚本、log 与本文档都不给物理结论。

## 1 输入与执行环境

| 项 | 说明 |
| --- | --- |
| 输入 | `<grid>.h5`，由 `python -m stm_data_processing.io.grid2h5 <grid>.3ds -o <grid>.h5` 得到（步骤 ①）；也可以直接把 `.3ds` 交给 `grid_topo.py`，脚本先做这次转换并**打印该转换命令** |
| h5 结构 | `/data (ny, nx, nch, npts)`、`/channels`、`/channel_units`、`/bias`（V）、`/params` + `/param_columns`、`/header` |
| 视场 L (nm) | 从 h5 header 的 `Scan>Scanfield` 解析（`x0;y0;width;height;angle`，`;` 分隔、单位 m，取第 3 段宽 ×1e9=nm）；没有则退回 header 的 `Grid settings`（同一条记录）；`-L` 可显式覆盖 |
| 解释器 | 仓库 venv：`cd /path/to/STM_DataProcessing && MPLCONFIGDIR=<可写目录> PYTHONDONTWRITEBYTECODE=1 .venv/bin/python <脚本>`（不要用 `uv run`） |
| 依赖 | `h5py`、`numpy`、matplotlib（经 `stm_data_processing.utils.plot_funcs`），以及本仓库 `skills/lawler-fujita-correction/scripts/stm_lf_correct.py`（**相对路径嵌套调用，不修改它**） |

## 2 三步流程（确切命令）

### 2.1 步 ①：`.3ds` → `.h5`

```bash
cd /path/to/STM_DataProcessing
.venv/bin/python -m stm_data_processing.io.grid2h5 GRID.3ds -o GRID.h5
```

产物 `<out>/GRID.h5`（与 `grid2h5.py` 的 h5 约定一致）。

### 2.2 步 ②：`.h5` → LF 输入 topo CSV（`scripts/grid_topo.py`，行为不变）

```bash
.venv/bin/python skills/grid-lf-pipeline/scripts/grid_topo.py GRID.h5 -o <out>/<stem>_topo.csv [-L 30]
```

处理链（`grid_topo.preprocess`，顺序固定，逐字如下）：

```
z = np.flipud(params['Z (m)'].reshape(ny, nx))   # ⓪ 转到 top-down（见 §5 第 1 条）
z -= np.median(z, axis=1, keepdims=True)     # ① 逐行去背景（axis=1 = 快轴，逐慢扫行）
z  = subtractMeanPlane(z.T).T                # ② 减二维最小二乘平面（仓库 helper，NaN 安全）
z -= z.min()                                 # ③ 平移最小值到 0
np.savetxt(<out>/<stem>_topo.csv, z, delimiter=",", fmt="%.10e")
```

② 用仓库的 `stm_data_processing.utils.plot_funcs.subtractMeanPlane`；`.T` 不改变拟合出的平面，只改变平面求值的
末位舍入——转置调用是与已确认的参考 topo CSV 在撤销 ⓪ 的约定翻转之后逐字节一致的那一种（§6）。
`<stem>` 是 h5 的文件名词干。此步 fail-closed：输入 `.h5` 不存在、或 `/param_columns` 缺 `Z (m)` 列时，脚本
`SystemExit`（非零退出）。

### 2.3 步 ③：LF 位移场拟合 + 导出可迁移位移场包（`scripts/grid_lf_fit.py`）

```bash
.venv/bin/python skills/grid-lf-pipeline/scripts/grid_lf_fit.py <out>/<stem>_topo.csv -L 30 -o <out>/lf
```

内部按相对路径调用 `skills/lawler-fujita-correction/scripts/stm_lf_correct.py`（默认 `--lambda-nm 3`，`--stm-lib`
用仓库默认），并带 `--save-transform`，把可迁移位移场包 `<stem>_topo_transform.json` + `<stem>_topo_transform.h5`
落进 `-o` 指定的目录（即流水线布局里的 `<out>/lf/`）。

- 本流程固定用 `--lambda-nm 3`：LF 自己的默认 30 nm 对 30 nm 视场太宽，`correction.log` 会报
  `band margin ... OUTSIDE the low-pass radius` 的 WARNING；用 3 nm 时同一行报 `band margin lambda x offset =
  0.418`（判据要求 < 1），落在低通半径之内。**这条 WARNING 是必看的**：出现就说明提取出的相位/位移场不是晶格畸变，
  应调 `--lambda-nm` 重跑，不要拿这份位移场去矫正数据。
- `-o` 的值就是 `<out>/lf/`：该目录下直接是 LF 拟合的全部产物（见 §3），并含 `--save-transform` 带来的位移场包。
- 此步 fail-closed：缺 topo CSV、LF 脚本不存在、**LF 拟合失败**（`stm_lf_correct.py` 非零退出、或它的
  `correction_report.json` 记录 `fallback: true` 即没有可用的位移场）→ 脚本**只删除本次运行新增的产物**、打印
  原因、非零退出；`-o` 目录里原有的文件一律保留（脚本进入时先记下目录内容，见 `scripts/grid_lf_fit.py:115` 的
  `pre_existing`），退出信息形如 `fail-closed: fit failed, removed N file(s) written by this run from <dir>
  (files that were already there are kept)`（`scripts/grid_lf_fit.py:180-181`）。所以重跑前把已有产物留在 `-o`
  里是安全的：这次失败不会清空它们，非空目录也不阻碍重跑。

## 3 产物布局

`grid_lf_pipeline -o <out>` 的完整产物：

```
<out>/<stem>_topo.csv                    步 ② 的 topo（LF 的输入，top-down；grid_topo.py 产出）
<out>/lf/<stem>_topo_corrected.csv       LF 拟合段产物（u 场校正后的 topo）
<out>/lf/<stem>_topo_corrected.h5        校正后的 topo + FFT2（HDF5，仓库 h5 约定）
<out>/lf/<stem>_topo_corrected_fft2.npy  复数 FFT2（complex128，fftshifted）
<out>/lf/<stem>_topo_corrected.png       topo 图
<out>/lf/<stem>_topo_corrected_fft.png   FFT 图
<out>/lf/<stem>_topo_lf.h5               局域相位·幅度·位移·掩码（LF 单一 h5）
<out>/lf/<stem>_topo_lf_*.png            各 LF 数组的预览图
<out>/lf/<stem>_topo_transform.json      可迁移位移场包（JSON 元数据）
<out>/lf/<stem>_topo_transform.h5        u 场本体（h5）
<out>/lf/correction_report.json          LF 机器可读报告
<out>/lf/correction.log                  LF 控制台日志
```

## 4 与 `lawler-fujita-correction` 的分工

| 项 | `lawler-fujita-correction` | 本 skill（grid-lf-pipeline） |
| --- | --- | --- |
| 职责 | 拟合位移场 + 重采样实现（`stm_lf_correct.py` / `stm_lf_apply.py`） | 面向 grid 的 ① 3ds→h5、② 抽 topo 三步预处理、③ 调 `stm_lf_correct.py` 并导出位移场包 |
| 依赖方向 | 独立 | **按相对路径** `skills/lawler-fujita-correction/scripts/stm_lf_correct.py` 嵌套调用 |
| 是否修改对方 | — | **否**。只运行那个脚本，不读不写 `skills/lawler-fujita-correction` 下任何文件 |

把拟合出的位移场套用到其它 map（current / dI/dV）**不在此 skill** ——那是 Skill B 的职责。本 skill 在步 ③
写完位移场包即止。

## 5 已知坑

1. **h5 `/channels` 里没有 Z**。针尖高度在 `/params` 的 `Z (m)` 列，`reshape(ny, nx)` 才是 topo；3ds 路径
   `grid2h5` 对 `/params` 与 `/data` 用同一扁平像素索引、都不翻转（同帧、自下而上），因此本 skill 在源头做
   **一次** `np.flipud` 把 topo 定为 top-down（跟 sxm 一致）。dI/dV / current 的抽取与套用不在此 skill。
2. **LF 的 `--lambda-nm` 默认要显式设为 3**：30 nm 视场用 LF 自带默认 30 nm 会触发
   `OUTSIDE the low-pass radius` 警告，位移场不可用；换 3 nm 重跑。
3. **`-o` 即 `<out>/lf/`**：`grid_lf_fit.py -o <out>/lf` 把拟合产物与位移场包直接写进该目录，不要再叠一层 `lf/`。

## 6 自检口径

```bash
# 步 ② topo 必须与已确认的参考 topo CSV 逐字节相同（都出自同一 flipud 三步预处理）
.venv/bin/python skills/grid-lf-pipeline/scripts/grid_topo.py GRID.h5 -o /tmp/gridskill-a/topo.csv
shasum -a 256 /tmp/gridskill-a/topo.csv \
    <既有>/20251110_grid1_30mV500pA_30nm_topo.csv   # 必须逐字节相同

# 步 ③ LF 拟合必须可复现既有 run（同一 byte 输入、同一 -L / --lambda-nm 3）：
# 产出 <stem>_corrected.csv 与既有 lf/<stem>_topo_corrected.csv 逐元素相对差 ≤1e-9（实测逐字节=0）
.venv/bin/python skills/grid-lf-pipeline/scripts/grid_lf_fit.py /tmp/gridskill-a/topo.csv -L 30 -o /tmp/gridskill-a/lf
```

**本 task 未修改 `skills/lawler-fujita-correction` 下任何文件。**