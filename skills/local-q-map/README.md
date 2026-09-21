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
| `scripts/localqmap.py` | 引擎库：`psi_q` 解调、单位换算、基矢来源解析、掩码、圆统计、`save_map`/`save_npy` |
| `scripts/selftest.py` | 一键 self-test（11 项 / 46 条断言，退出码 0 = 全过） |

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
| `<stem>_q{j}_field.npy` | 主产物：`complex128` 的 `psi_q(r)` | — |
| `<stem>_q{j}_amplitude.npy` / `.png` | `|psi_q|` ≈ `A_q/2` | inferno |
| `<stem>_q{j}_theta.npy` / `.png` | `arg psi_q`（弧度，`(-pi, pi]`，无 `q.r` 斜坡） | twilight，`(-180, 180]` 度，`vmin=-180, vmax=180` |
| `<stem>_q{j}_mask.npy` / `.png` | 有效掩码（float 0/1，**1 = 有效**） | gray |
| `local_q_map.log` | 控制台报告（基矢来源与解析值、窗宽、警告、逐 q 统计、产物清单） | — |
| `local_q_map_report.json` | 机器可读报告（顶层设置 + 基矢来源 + 警告 + 逐 q 全字段） | — |

`--no-figures` 只写 npy（不写上面 3 张 PNG）。

**掩码语义**：无效 = 输入 NaN 区域 ∪ `|psi_q| < amplitude_fraction × median(|psi_q|)`；
`mask_coverage_fraction` 是**有效像素占比**。PNG 里无效像素画成灰（bad colour），npy 一律保持有限。

## 4 self-test

```bash
cd /path/to/STM_DataProcessing
MPLCONFIGDIR=<可写目录> PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python <this skill>/scripts/selftest.py [--workdir DIR] [--size N] [--keep]
```

11 项检查（46 条断言；单项阈值与实测值见 `SKILL.md` §5）：单平面波幅相与符号约定、12 个 q 的幅相与串扰、
`(h,k)` 与 `--basis-px` 等价、基矢解析（lawler-fujita / affine 两级退化 / affine `anchor_ring=r3` /
**旋转 affine 报告**：默认被画布重新取向、`--basis-orientation-from report` 复现失配（`--no-strict` 仍出产物并带标志、默认 `--strict` 退出 3）/
无匹配退出）、规范律（φ0 与平移）、Friedel 恒等式、均值恒等式、
NaN 规则与掩码、确定性（逐字节）、视场来源（`--size-nm-from-log` 三支 + 报告 `corrected_nm_per_px` × 画布）、
端到端产物契约（含基矢-画布校验通过、nm/px 不一致告警与幅度加权圆中位数）。
