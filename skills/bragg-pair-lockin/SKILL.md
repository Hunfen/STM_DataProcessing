---
name: bragg-pair-lockin
description: 任意指定 Bragg 点（对）的局域场、实空间重构与逐点相位相干：把 (h,k) 或「n× 基 + 系数」给出的点集逐个解调成局域复场 ψ_q、逐点判定解析式 B=2Re[ψe^{iq·r}] 的适用性（不适用就改用双侧严格重建）、并对每个点自己的场做相位相干分析（C(Δr)，配对数 ≥0.5 才算有效、衰减长度 ξ、残余相位斜坡与其割线按 min(fit_range, L/2, 2×掩码 inset) 截断以免循环回绕污染、应变常数 A = 2β/(√3a₀) 由 (β, a₀) 现算（= 1.408171388 Å⁻¹）），输出每点 h5（field/amplitude/theta/mask）、每对 B(r) h5 与预览、五张泛化步骤图、report.json（含 coherence 块）与 log；`--replot` 可只出图不重解调。只做「点→场→实空间→该点场的相位相干」，跨点整合/成对与三重积统计/图集归 phase-analysis。
---

# bragg-pair-lockin

## 1. 这个 skill 做什么

把原来只针对「三对 r3」的一次性 lock-in 泛化成任意指定 Bragg 点（对）的局域场、实空间重构与**该点场的相位相干**。层有四层：

```
一组 Bragg 点 q  ->  q 处的局域复场 psi_q  ->  该反射对的实空间 B(r)
                 ->  该点自己的场的相位相干 C(Δr)（ξ / 残余相位斜坡 = 等效应变）
```

**不做**：跨点整合、成对/三重积相位统计、相位直方图与图集——那些属于姊妹 skill `phase-analysis`。第 5 步是**逐点分开**做的（不把所有 q 合到一起），且**默认总开**（不加 `--coherence` 之类开关；`--no-figures` 负责关 PNG）。

引擎是姊妹 skill `local-q-map`：同进程 `import localqmap`，用它的解调函数算子（不复制、不子进程、仓库内不存在第二份实现）。HDF5 写入走 `local-q-map/scripts/h5io.py` 的约定（gzip/4、显式 1 MiB 分块、`schema_version`/`generator`/`creation_date`）。

## 2. 点集的两种写法（内部唯一真值是 (h,k)）

| 写法 | 命令行 | 语义 |
|---|---|---|
| 分数/小数 | `--q 1/3,1/3` | q = h·b1 + k·b2 |
| n× 基 + 系数 | `--g 3:1,1` | q = (1·b1 + 1·b2)/3 |

两者在内部都化成精确的 `Fraction`，(h,k) 是唯一真值：`--q 1/3,1/3` 与 `--g 3:1,1` 得到**逐位相同**的 h、k、|q|、q_px、route。report.json、log、图注同时打印两种写法（例如 `(h, k) = (1/3, 1/3) = 3:1,1 x basis   [--q 1/3,1/3 | --g 3:1,1]`）。

`--g` 的系数是整数、可以是负的（`--g 3:2,-1` = (2/3, −1/3)）；两种写法也可以混用、可重复、按给出的顺序编号 `q0, q1, …`（`--label NAME` 可按顺序改名）。

## 3. 用法

```bash
cd /path/to/STM_DataProcessing
MPLCONFIGDIR=<可写目录> .venv/bin/python <skill>/scripts/stm_bragg_pair_lockin.py \
    CORRECTED.csv -o OUT_DIR \
    --basis-from correction_report.json \
    --lambda-nm 13 \
    --q 1/3,1/3 --q 2/3,-1/3 --q 1/3,-2/3
```

主要选项：

| 选项 | 含义 |
|---|---|
| `--lambda-nm` | **必填、无默认**；本 skill 的高斯窗宽（不是 lawler-fujita 的 λ）。缺失即非零退出（exit 2） |
| `--lambda-nm-label LABEL=V` | 逐点覆盖窗宽；report 逐点记 λ 与三条判据 |
| `--q H,K` / `--g N:I,J` | 点集（可重复） |
| `--basis-from` / `--basis-px` | 1×1 倒格基矢来源（二选一，必须给一个） |
| `-L` / `--size-nm` | 直接给出画布视场（nm） |
| `--size-nm-from-log` | 从矫正 log 读视场（读 `# corrected canvas: … field of view <value> nm` 行） |
| 视场缺省 | 上面两个都没给时，沿用对标定报告的 `corrected_nm_per_px × N` |
| `--label NAME` | 按给出顺序给点改名（可重复，个数须与点数一致） |
| `--list-candidates` | 即使给了点集也写出候选表（此时串扰列对着已选点算） |
| `--denominator-max` | 候选表有理近似的分母上限（默认 12；语义见 §4） |
| `--ring-tol` | 候选表按半径聚环的相对容差（默认 0.05） |
| `--amplitude-fraction` | 掩码阈值（`fraction × median|ψ|`，默认 0.10，沿用 local-q-map） |
| `--delimiter` | 输入 CSV 的分隔符（默认 `,`） |
| `--steps-only LABEL` | 只给该点出五张步骤图 |
| `--no-figures` | 只写 h5 与 report，不出 PNG |
| `--replot` | **只出图模式**：读已有 `report.json` + 各 h5 + CSV（只为画布无窗 FFT 幅度），不重解调、不写 h5、不覆盖 `report.json`（只在同一个 log 末尾追加 `# replot` 段）；λ/基矢/FOV/anchors 全部从 report 取，因此 `--lambda-nm` 与 `--basis-from/--basis-px` 在此模式下不必填（给了就忽略并记 WARNING）；产物路径按 basename 在 outdir 下解析；产物缺失或 h5 形状/λ/标签与 report 不符 ⇒ **exit 5** 且不写任何 PNG |
| `--stm-lib` | `bragg_peak` 检测器的来源目录（默认仓库 `src`；只影响候选表路径） |

**非交互**：点集必须显式给。缺 `--q/--g` 时自动列出候选表、写 `q_candidates.json`，并以**非零码**退出（exit 2）。检测不到任何峰时写一张空表并以 exit 3 退出。

## 4. 候选表口径（`q_candidates.json`）

1. `bragg_peak.detect_bragg_peaks` 在矫正画布上找峰；
2. 每个峰用 1×1 基矢做有理近似归属 (h,k)（分母上限可配，默认 12），并给出两种写法；
3. 按半径聚成环（`--ring-tol`），环按半径升序编号 `ring0, ring1, …`（**不用** k 空间高对称点命名）。

`--denominator-max` 的语义是**逐分量**约束：h、k 各自的分母 ≤ 该上限（逐分量 `limit_denominator`）。两种写法的「n× 基」尺度是这两个分母的最小公倍数（LCM），**可以超过该上限**——例如 `--denominator-max 12` 下 (h,k) = (1/3, 1/5) 写作 `--g 15:5,3`（LCM = 15 > 12）。r3 主用例（分母恒为 3）不受影响；报告里的 `denominator_max` 字段只约束逐分量分母，不约束 LCM 尺度。

每行给出：(h,k)、两种写法、|q|、实测 |F|、方位角、Friedel 伙伴强度与对称残差、与已选点的串扰标记、**有理近似残差**（公度/非公度如实展示，非公度点不会被藏起来）、环内幅度残差。串扰列对每个已选点的 ±q 算最小 `|Δq|` 并与 `2/λ` 比：落在窗口内才算串扰；落在**一个 FFT 格子**（`dq_nm_inv`）以内的记为 `same_reflection`（那就是该已选反射本身，不是第二支）。样例：

```
# candidate ring1 #3: (h, k) = (1/3, 1/3)  [--q 1/3,1/3 | --g 3:1,1]  |q| = 17.0247 nm^-1
  |F| = 4.447e-07  azimuth = 29.8787 deg  Friedel partner |F| = 4.447e-07
  (asymmetry residual 0, detected True, distance 0.000 px)
  rational residual = 2.858e-03 nm^-1 (1.679e-04 of |q|, commensurate)
  ring amplitude residual = 0.196  cross-talk: none (min |dq| = 17.0276 nm^-1 vs 2 / lambda = 0.1538 nm^-1)
```

## 5. 配对与退化

- 输入是 q 的**集合**，每个 q 自动配 −q（同一个复场 ψ_q 里已经含着 −q，所以 B(r) 就是这一对的重构）。
- 同时显式给了 q 与 −q 时自动并成**一对**，只保留先给的那个作代表，第二个记为 alias 并在 log 里提示（`merged into the single pair`）。
- **自共轭点**（2q ≡ 0 mod G，含 q=0）只 WARNING，report 标 `phase_degenerate: true`（点与 pair 两处），产物照常写出。
- 自共轭点**不需要**专门的 ×2 修正：单反射自为搭档，解析式 `B = 2 Re[ψ e^{iq·r}]` 与原对是同一个写法（代码里因此没有 `phase_degenerate` 分支），选路仍逐点由三条判据决定。自检的单平面波恒等式（q 取 FFT 格点）实测解析路与双侧路都重构到 1e-15；非格点 q 的纯平面波两者都带泄漏/截断误差，所以「是否精确」由 `criteria` 与 `route` 决定，本条只声明没有特判。

## 6. 两条重建路与三条判据

默认解析式（**analytic**）：

```
B(r) = 2 Re[ psi_q(r) exp(i q.r) ]
```

逐点算三条适用条件，任一不满足就自动改用**双侧严格重建（two_sided）**——在原始 FFT 上同时抠 ±q 再反变换：

```
B(r) = Re{ IFFT[ FFT[T] * (G(k - q) + G(k + q)) ] },   G 为同一高斯窗
```

| 判据 | 定义 | analytic 要求 |
|---|---|---|
| Friedel 伙伴抑制 | `2 (λ|q|)²` 与 `exp(-2(λ|q|)²)` | `2λ²|q|² ≥ 6.9078`（伙伴权重 ≤ 1e-3） |
| 串扰 | 与其它点的 ±q 的最小 `|Δq|`（rad/nm）对 `2/λ` | `min|Δq| ≥ 2/λ`（单点无此项，记为 not defined） |
| 慢变 | `median|ψ(r+一个载波周期) − ψ(r)| / median|ψ|` | `≤ 1.0` |

report 与图注逐点标 `route = analytic | two_sided`，并把三条判据的数值、阈值、失败原因全部写进 `criteria`/`route_reason`。判据与 route 都可以从 report 单独复算。

两条路**不是**恒等：解析式在解调帧里正好在 q 处取样，双侧路在原始 FFT 网格上抠 ±q，差的是布拉格峰在网格上的插值。这个差别会被如实记进 `route_check`（逐点相关系数、rms 比、最大绝对差），自检把「相关系数 ≥ 0.9 且 rms 比在 [0.6, 1.4]」当作真实数据的交叉核对门。

## 7. 单位与口径

| 量 | 单位/口径 |
|---|---|
| `q_nm_inv` | rad/nm（= 2π × cycles/nm），沿用 `local-q-map` 的径向口径 |
| `q_px` | 有符号 fftshift 偏移（px），与 `bragg_peak` 同口径 |
| `field` | complex128，ψ_q(r) = FFT⁻¹{FFT[T·exp(−i q·r)]·exp(−λ²|k|²/2)} |
| `amplitude` | \|ψ_q\|（含 `A cos(q·r+φ)` 时 ≈ A/2）；另有 median/FWHM/max/min_valid |
| `theta` | arg ψ_q，弧度，(−π, π]，**无 q·r 斜坡**，附 circular median / R |
| `mask` | float 1 = 有效；无效 = 输入 NaN 区 ∪ \|ψ_q\| < fraction × median\|ψ\| |
| `B` | 与输入拓扑同单位（本数据集为 m） |
| `C(Δr)` | 相位相干，复数量：`C(Δr) = <exp(i[θ(r)−θ(r+Δr)])>`，用循环自相关 `A = IFFT(|FFT(u)|²)`、`N = IFFT(|FFT(mask)|²)`、`C = A/N` 估计（`u = exp(iθ)` 只在掩码内非零，掩码再按 0.65 λ 侵蚀排除周期 FFT 边界带）；径向剖面按 1 px lag 环、**配对数加权**的复平均 |
| 配对数下限 | `N(Δr) ≥ 0.5` 才算有效（配对数按定义是整数，< 0.5 是掩码 FFT 的浮点残差），否则该 lag 记 `nan`；真实 ≥1 的值不受影响 |
| `xi_nm` | `|C(Δr)|` 的指数衰减长度：对 `0 < Δr ≤ fit_range` 的**所有非空 bin** 做 `polyfit(Δr, log|C|, 1)`（不筛样本数）；slope ≥ 0 ⇒ `inf`，可用 bin < 5 ⇒ `nan` |
| `max_lag_nm` / `fit_range_nm` | 按视场自适应：`max_lag = L/√2`（L = 画布视场边长），`fit_range = (2/3)·max_lag`；`|C|` 锚点取 `0.10/0.25/0.50/0.75 × max_lag` 处的插值 |
| `inset_x_px` / `inset_y_px` | 有效掩码区到画布 x（左右列）/ y（上下行）边缘的最小距离（px） |
| `wrap_free_lag_nm` | `2 × min(inset_x_px, inset_y_px) × nm_per_px`：超过它，循环自相关会把画布远端的相位折进 `arg C`（回绕污染） |
| 割线长度 `cut_nm` | **被 cap 截断**：`cut_nm = min(fit_range_nm, min(L/2, 2·min(inset_x, inset_y))·nm_per_px)`；割线是 `cut_nm/nm_per_px` 个样本（lag 0 … n−1 px，与历史脚本切法一致），发生截断时 `cut_truncated = true` 且 log 逐点一行 WARNING。ξ 拟合窗与 `|C|` 锚点**不受** cap 影响 |
| `slope_rad_per_nm` | `arg C` 沿 +x / +y 一维割线（长度 = `cut_nm`）unwrap 后的线性斜率；x 分量 → ε_xx−ε_yy，y 分量 → ε_xy |
| `slope_per_angstrom` | `= slope_rad_per_nm / 10`（1 Å⁻¹ = 10 nm⁻¹） |
| `strain_equivalent` | `= |slope_per_angstrom| / (2 × A) = |slope_rad_per_nm| / (20 × A)`，赝规范 `A = 2β/(√3 a₀)`（**严格公式值，代码里由 `(β, a₀)` 现算，不硬编码**），取 `a₀ = 0.246 nm`、`β = 3` ⇒ `A = 14.081713883 nm⁻¹ = 1.408171388 Å⁻¹`、除数 `20A = 28.163427765 /nm`（report 记录 `a_nm/beta/A_per_strain_angstrom_inv/strain_divisor_per_nm`，selftest 会按同一份 report 里的 β/a₀ 复算并核对到 1e-12）。历史一次性脚本用的是**四舍五入**的 `A = 1.4082`（除数 2.8164），并把 rad/nm 当 rad/Å 用，所以 `ε_new/ε_old = 2.8164/(20A) = 0.1 × (1 + 2.0308e-5)`——旧 v2 对账的期望比值即 `0.1 × (1 ± 3e-5)` |

## 8. 产物

```
OUT/
  report.json                 机器可读报告（顶层 skill/input/lambda_nm/q/pairs）
  bragg_pair_lockin.log       全量日志（与 stdout 同文）
  q_candidates.json           候选表（缺 --q/--g 或 --list-candidates 时）
  <stem>_<label>.h5           每点：field / amplitude / theta / mask
  <stem>_<label>_{amplitude,theta,mask}.png      每点预览（无坐标轴、NaN 灰）
  <stem>_<label>_B.h5         每对：B(r) + mask（含 route/λ/两种写法等属性）
  <stem>_<label>_B.png        每对 B(r) 预览（1:1 像素、viridis、NaN 灰）
  B_pairs_summary.png         所有对的合图（共享绝对色标 + 逐对图注）
  steps_<label>/step1_fft_marks.png       FFT 标点（本点 ±q 与全部已选点）
  steps_<label>/step2_q_zoom.png          q 空间放大 + 高斯窗 1σ/2σ/3σ
  steps_<label>/step3_modulated_field.png 调制场 |X| 与 arg X（低通前）
  steps_<label>/step4_lowpassed_field.png 低通场 |ψ| 与 arg ψ（含 route）
  steps_<label>/step5_phase_coherence.png 该点场的相位相干（上 |C(Δr)| 曲线 + |C|=0/1 两条虚线；
                                          下 2D |C(dx,dy)|，±max_lag，viridis 0–1，中心十字，色条在数据区外）
```

`report.json` 顶层含 `coherence` 配置块（`definition/criterion/max_lag_nm/fit_range_nm/border_erosion_px/anchor_fractions/max_lag_rule/fit_rule/ramp_cut_rule/pair_floor/a_nm/beta/A_per_strain_angstrom_inv/strain_divisor_per_nm/strain_conversion/no_sample_count_filter`）；每个 `q` 元素含 `label/h/k/writings/q_nm_inv_abs/route/criteria/amplitude` 与 `coherence` 块（`n_valid_px/valid_fraction/border_erosion_px/inset_x_px/inset_y_px/wrap_free_lag_nm/ramp_cap_nm/anchors/xi_nm/lag_profile{lag_nm,C_real,C_imag}/profile_last_lag_nm/profile_last_pairs/ramp{x,y}{slope_rad_per_nm,slope_per_angstrom,strain_equivalent,cut_nm,cut_truncated}/ramp_cut_nm/ramp_cut_px/ramp_cut_truncated/ramp_fit_max_nm`）；`pairs` 每项含代表点、伙伴标签、alias、route、`route_check`、`B` 统计与产物路径。第 5 步不写 h5：`--replot` 由 h5 的 θ/mask + report 参数确定性重算。图注（步骤图与合图）逐点同时打印两种写法与 route；脚本在保存前会从 figure 上**读回**图注文本做断言，声明了却没画出来会直接报错。

## 9. 自检

```bash
MPLCONFIGDIR=<可写目录> .venv/bin/python <skill>/scripts/selftest.py
```

五族检查（全绿才 exit 0）：

1. **合成恒等式**：Friedel `ψ_{−q} = conj ψ_q`；`Σψ = ΣT e^{−iq·r}`（与 `localqmap.demodulated_sum` 的闭式独立比对）；圆滚 = 精确平移；单平面波 `B = A cos(q·r+φ)`（这条钉住解析式的因子 2）。
2. **产物与命令行契约**：两种写法互认、缺 `--q` 时的候选表、显式 −q 的合并、自共轭点告警、逐点 λ、产物清单、`--steps-only`、图注读回。
3. **真实数据交叉核对**：20251112 矫正画布上的三个 r3 点，解析路 vs 双侧路逐点比对（写进该次运行的 report）。
4. **相位相干（第 5 步）**，六条断言：① 常数相位场 ⇒ `|C| ≡ 1`（实测 max|1−|C|| = 2.2e-16，362 个 lag bin）；② 沿 +x / +y 注入 0.01 rad/nm ⇒ 对未注入基线的变化量按 1:1 复原（实测误差 2.5e-17 rad/nm）；③ `λ` 块随机化相位（块 = λ，偏移 0/±2π/3，固定种子）⇒ `|C|(max_lag)` 显著低于真实场（实测 0.0045 vs 1）；④ **已知真值 + 割线 cap**：与真实产物同形掩码（inset 73 px ⇒ cap 7.30 nm）下 cap 割线复原误差 0.0000 %，而 cap 禁止的 0.9·L 割线误差 79.89 %；⑤ **配对数下限**：小有效区（42436 px，reach < max_lag）使 73 个 lag 环真配对数为 0 ⇒ 这些环是 `nan`、`max|C| = 1`，且独立（非 FFT、直接切片计数）抽样确认它们一个真配对都没有、有限环最多含 42436 对；⑥ **应变常数自洽**：report 记的 `A_per_strain_angstrom_inv`（与 `strain_divisor_per_nm`）必须等于按同一份 report 里的 `beta`/`a_nm` 复算的 `A = 2β/(√3a₀)`（与 `20A`），相对差 ≤ 1e-12（实测两边都是 1.408171388267380 / 28.163427765347603，相对 0.000e+00）——钉住「常数是公式，不是手写数字」。每条都有「改坏对应实现 ⇒ 该条断言变红」的变异：`coherence-normalisation`（用像素数而不是像素对数归一）、`coherence-ramp-units`（割线用 px 而不是 nm）、`coherence-flat-bins`（所有像素塞进第一个 lag 环）、`coherence-no-cut-cap`（去掉 cap，割线跑到 fit_range：cap 那条断言实测从 0.0000 % 变 79.89 %）、`coherence-pair-floor`（配对数下限退回 `N > 0`：实测 `max|C|` 由 1 变 3.98，且 84 个本有真配对的 lag 环被 NaN 污染）、`coherence-hardcoded-strain-constant`（把应变常数写死成 14.081746 nm⁻¹：自洽那条断言实测从「记录值 = 复算值」变 1.408174600000000 vs 1.408171388267380、28.163492000000005 vs 28.163427765347603）。
5. **非空转与负对照**：空画布（全 NaN → exit 3）、零场（→ exit 4）、无峰候选表（→ exit 3）三条「被测对象置空必须报错退出」；十条负对照（去 ×2 / 禁用自动选路 / 候选表为空 / 模块置空 + 上面第 4 族的六条）都在 `/tmp` 副本上做，各须让**指定的那一条**断言变红（变异运行的 stdout 里该 FAIL 行的名字会被核对），并且同一断言在未变异树上必须是绿的。

真实数据路径：`--csv/--report`、环境变量 `BPL_SELFTEST_CSV/BPL_SELFTEST_REPORT`，或从 workspace 根起找 `data_processing/20251110/20251112_topo0009_1mV100pA_1000nm/`。找不到真实数据时该项**判红**（不静默跳过）。

## 10. examples/（历史一次性脚本：正文逐字节存档）

`examples/` 下 6 个脚本是 r3 lock-in 阶段写在数据目录里的一次性脚本，**正文逐字节存档**（去掉文件头 3 行后的 sha256 与移交时的原文件逐个相同）；它们的路径/STEM 是当时硬编码的，不被任何模块 import，不参与本 skill 的运行。

每个文件头固定 3 行（目的只有一个：在不改正文一个字节、也不动仓库 `pyproject.toml`/`ci.yml` 的前提下过 CI 的两道 ruff 门）：

```
# ruff: noqa   <- 文件级豁免 ruff check（正文是冻结的存档，不再按 lint 规则改）
# fmt: off     <- 让 ruff format 完全跳过该文件（正文不许被重排）
# 一行说明：逐字节存档、不被 import、不参与运行、正文冻结
```

`scripts/` 下的 4 个文件**不**用这种豁免，而是按仓库 ruff 配置（`select = E,W,F,I,B,C4,UP,YTT,Q,SIM,PTH,RUF`，忽略 `E501`）修到两道门全绿：改动只有格式/import 顺序/引号风格/多余的 `int(round(...))`/一个未使用的 import，归一化后的 AST 逐文件相同（`tmp/bpl_v2/check_lint_equivalence.py`）：

| 文件 | 当时做什么 |
|---|---|
| `make_B_r3_pairs.py` | 三个 r3 B(r)（本 skill 的 `_B` 产物即它的泛化，双侧路交叉核对口径相同） |
| `make_lockin_steps.py` | 早期合版步骤图 |
| `make_lockin_steps_figs.py` | 四张分离步骤图（本 skill 的 `steps_*/step1..4` 即它的泛化） |
| `make_theta_maps.py` | r3 相位图 |
| `make_r3_mask_preview.py` | 掩码/窗预览 |
| `make_phase_coherence.py` | 相位相干 C(Δr) 的历史 v2 口径（**已泛化进本 skill**：逐点 `steps_<label>/step5_phase_coherence.png` + report 的 `coherence` 块；旧脚本的 shuffle/常数相位对照曲线按用户裁定**不**再画；跨点整合仍不做） |

## 11. 与姊妹 skill 的分工

| skill | 负责 |
|---|---|
| `lawler-fujita-correction` / `affine-correction` | 几何矫正，产出矫正画布与标定报告（本 skill 的基矢来源） |
| `local-q-map` | 单个 q 的局域场引擎（本 skill 直接调用） |
| `bragg-pair-lockin`（本 skill） | 任意点集 → 逐点局域场 + 每对 B(r) + 逐点相位相干（C(Δr)、ξ、残余相位斜坡 = 等效应变）+ 五张泛化步骤图 + `--replot` 只出图（点→场→实空间→该点场的相位相干） |
| `phase-analysis` | 跨点整合、成对/三重积相位统计、相位直方图与图集（本 skill 不做） |

已知刻意的口径选择：本 skill **不**重测基矢取向（直接用报告给的 1×1 基矢；`local-q-map` 会用画布取向覆盖相差 >1° 的报告取向）——本数据集两者相差 0.0023°，远小于该阈值。若画布与报告取向明显不符，请先用 `local-q-map` 的取向诊断，再用 `--basis-px` 显式给基矢。
