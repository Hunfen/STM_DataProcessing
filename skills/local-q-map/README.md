# local-q-map —— 文件与产物说明

对**矫正后**拓扑 CSV，用矫正产物给出的 1×1 六方倒格基矢 `{b1, b2}` 表达任意 `q(h,k)`，
做 Gaussian 窗局域傅里叶滤波，输出每个 q 的复数局域场及其幅度/相位/掩码。
只做几何与信号处理，不做物理解释。完整方法、约定与契约见 `SKILL.md`。

## 1 文件列表

| 路径 | 说明 |
| --- | --- |
| `SKILL.md` | 方法定义、约定、CLI、输出契约、警告、限制、self-test 表 |
| `README.md` | 本文件：文件与产物说明 |
| `CHANGES.md` | 修改记录 |
| `scripts/stm_local_q_map.py` | 命令行入口（参数解析、基矢/视场解析、逐 q 出产物、写 log 与报告） |
| `scripts/localqmap.py` | 引擎库：`psi_q` 解调、单位换算、基矢来源解析、掩码、圆统计、`save_map`/`save_products` |
| `scripts/h5io.py` | 逐 q HDF5 产物的自包含约定（镜像仓库 `h5_convention.py`：chunk/压缩/属性），该模块只用标准库 + `h5py` + `numpy` |
| `scripts/selftest.py` | 一键 self-test（12 项 / 53 条断言，退出码 0 = 全过） |

> 落盘（两处字节级一致）时**排除 `scripts/__pycache__/`**：它是 `py_compile` 验证步骤的副产物，不是 skill 内容。

## 2 运行

```bash
cd /path/to/STM_DataProcessing
export MPLCONFIGDIR=<可写目录> PYTHONDONTWRITEBYTECODE=1

# 一键 self-test
.venv/bin/python <this skill>/scripts/selftest.py

# 用矫正报告里的基矢分析矫正画布（取向默认由矫正画布自身测量；失配默认 --strict 直接非零退出）
.venv/bin/python <this skill>/scripts/stm_local_q_map.py \
  CORRECTED.csv -o OUT \
  --basis-from OUT_CORRECTION/correction_report.json \
  --q 1,0 --q 0,1 --q 1,-1 --q 1/3,1/3 --q 2/3,-1/3 --q 1/3,-2/3 \
  --size-nm-from-log OUT_CORRECTION/correction.log        # 失配即退出 3；--no-strict 可降级为只告警

# 显式基矢（画布上有符号 fftshift 偏移，px）
.venv/bin/python <this skill>/scripts/stm_local_q_map.py \
  CORRECTED.csv -o OUT --basis-px '242.48,0;121.24,209.99' --q 1,0 -L 51.709
```

> 注意：`--lambda-nm` 默认 **3.0 nm**（局域傅里叶窗宽），**不是** `lawler-fujita-correction` 的
> 默认 30 nm（lock-in 畸变尺度）。`--basis-from` 指向 LF 报告时，log 会把报告里的 `lockin.lambda_nm`
> 仅作参考打印出来。

> 注意：报告只可靠给出基矢**半径**；`affine-correction` 的 `orientation_deg` 与 LF 的 `q_a`/`q_b`
> 都是矫正**输入画布**的方向。默认（`--basis-orientation-from canvas`）由矫正画布自身测取向，
> 并在写出任何产物前做**无条件基矢-画布校验**；失配 ⇒ `WARNING` + `warnings.basis_ring_mismatch`，
> 失配 ⇒ 默认（`--strict` 默认开）不写逐 q 产物、退出码 3；`--no-strict` 降级为只 WARNING + 报告标志。详见 `SKILL.md` §3。

## 3 产物

`-o OUT`（`stem` = 输入 CSV 文件名主体，`j = 0,1,...` 对应命令行 `--q` 的顺序）：

| 产物 | 内容 | PNG |
| --- | --- | --- |
| `<stem>_q{j}.h5` | **主产物**：该 q 的四个数据集（`field` complex128、`amplitude` float64、`theta` float64、`mask` float64），见 §3.1 | — |
| `<stem>_q{j}_amplitude.png` | `|psi_q|` ≈ `A_q/2` | inferno |
| `<stem>_q{j}_theta.png` | `arg psi_q`（弧度，`(-pi, pi]`，无 `q.r` 斜坡） | twilight，`(-180, 180]` 度，`vmin=-180, vmax=180` |
| `<stem>_q{j}_mask.png` | 有效掩码（1 = 有效） | gray |
| `local_q_map.log` | 控制台报告（基矢来源与解析值、窗宽、警告、逐 q 统计、产物清单） | — |
| `local_q_map_report.json` | 机器可读报告（顶层设置 + 基矢来源 + 警告 + 逐 q 全字段；`artifacts` 指向 h5 与三张 PNG） | — |

`--no-figures` 只写 h5（不写上面 3 张 PNG）。

**掩码语义**：无效 = 输入 NaN 区域 ∪ `|psi_q| < amplitude_fraction × median(|psi_q|)`；
`mask_coverage_fraction` 是**有效像素占比**。PNG 里无效像素画成灰（bad colour），h5 的四个数据集一律保持有限。

### 3.1 h5 约定（每 q 一个文件）

| 层 | 内容 |
| --- | --- |
| 文件名 | `<stem>_q{j}.h5`（一个 q 一个文件，`j` = 命令行 `--q` 的顺序） |
| 根属性 | `schema_version = 1`、`generator = 'stm_local_q_map.py'`、`creation_date`（ISO-8601 带时区偏移） |
| 数据集 | `field`（complex128，主数据集）、`amplitude`（float64）、`theta`（float64，`units = 'rad'`）、`mask`（float64，1 = 有效，无量纲不写 units） |
| 存储 | `track_times=False`、`compression='gzip'` + `compression_opts=4`、显式 `chunks`（从最后一维起填满 ~1 MiB 未压缩字节；小于预算时 `chunks == shape`） |

该约定由 `scripts/h5io.py` 实现，是仓库 `src/stm_data_processing/io/h5_convention.py` 的**自包含镜像**
（该模块不 import `stm_data_processing`，只用标准库 + `h5py` + `numpy`）；详细口径见 `SKILL.md` §4.6 与
`docs/hdf5_convention.md`。

## 4 self-test

```bash
cd /path/to/STM_DataProcessing
MPLCONFIGDIR=<可写目录> PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python <this skill>/scripts/selftest.py [--workdir DIR] [--size N] [--keep]
```

12 项检查（53 条断言；单项阈值与实测值见 `SKILL.md` §5）：单平面波幅相与符号约定、12 个 q 的幅相与串扰、
`(h,k)` 与 `--basis-px` 等价、基矢解析（lawler-fujita / affine 两级退化 / affine `anchor_ring=r3` /
**旋转 affine 报告**：默认被画布重新取向、`--basis-orientation-from report` 复现失配（`--no-strict` 仍出产物并带标志、默认 `--strict` 退出 3）/
无匹配退出）、规范律（φ0 与平移）、Friedel 恒等式、均值恒等式、
NaN 规则与掩码、确定性（6 张 PNG 逐字节 + 两个 h5 的 8 个数据集逐位）、视场来源（`--size-nm-from-log` 三支 + 报告 `corrected_nm_per_px` × 画布）、
端到端产物契约（含基矢-画布校验通过、nm/px 不一致告警与幅度加权圆中位数）、
逐 q h5 产物契约（§3.1 的 schema、报告的 `artifacts`/`conventions` 措辞、`--no-figures` 文件集、
以及「h5 文件或某个数据集缺失时读产物的断言变红」与「`conventions` 退回 npy 措辞时断言变红」两组负向对照）。
