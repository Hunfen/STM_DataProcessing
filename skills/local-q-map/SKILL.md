---
name: local-q-map
description: STM 矫正后拓扑 CSV 的局域 q 图（Gaussian 窗局域傅里叶滤波）：由矫正产物给出的 1×1 六方倒格基矢 {b1,b2} 表达任意 q(h,k)（h,k 可为分数），对每个 q 输出复数局域场 psi_q(r) 及其幅度 |psi_q|、解调相位 theta_q = arg psi_q（弧度，(-pi,pi]，无 q.r 斜坡）与有效掩码的 npy/png 全套产物与机器可读报告；含基矢来源自动识别（lawler-fujita / affine 报告、显式 --basis-px）、NaN 规则与跨环串扰/边界警告，并附一键数学 self-test。当用户给出矫正后拓扑 CSV 且要求「任意 q 的局域复数场/幅度/相位图」「局域傅里叶滤波」「(h,k) 表达 q」时使用。
---

# local-q-map（局域 q 图，v1.0）

**范围**：本 skill 只做**几何与信号处理**——由倒格基矢表达任意 q、在该 q 上做 Gaussian 窗局域傅里叶滤波、
给出复数局域场与由它派生的幅度/相位/掩码，并做数学自检。**不做物理解释**：脚本、图、log 与文档都不给物理结论。
**不做**相位解缠（theta 保持折叠）、不做环检测、不做矫正（基矢来自矫正产物的报告）。

**与 lawler-fujita-correction 的关系**：`lawler-fujita-correction` 用 lock-in 相位解位移场并重采样（矫正）；
本 skill 只在**已矫正**的 CSV 上做局域傅里叶分析，基矢直接读它的报告。两者共用「局域相位」这一数学对象，
但 **λ 的含义与默认值不同**（见 §2.3），且本 skill 只输出相位本身，不构造位移场。

## 1 输入与执行环境

| 项 | 说明 |
| --- | --- |
| 输入 | 正方形拓扑数值矩阵 CSV/txt（**矫正后画布**，可含 NaN；`np.loadtxt` 逐字读入，不做 flipud/去平面，与 `phase-analysis` 同口径） |
| 视场 L (nm) | 矫正画布的视场边长；`-L` 或 `--size-nm-from-log correction.log`（见 §4.1），两者都无则可用 `--basis-from` 报告里的 `corrected_nm_per_px` × 画布 |
| 倒格基矢 | 1×1 环六方倒格基矢 `{b1, b2}`，夹角 60°（`bragg_peak` 的 `_HEXAGONAL_UNIT` 约定）；来源见 §3 |
| 解释器 | 仓库 venv：`cd /path/to/STM_DataProcessing && MPLCONFIGDIR=<可写目录> PYTHONDONTWRITEBYTECODE=1 .venv/bin/python <脚本>`。**不要用 `uv run`** |
| 绘图 | `text.usetex=False`（mpl 3.10 + TeX Live 2026 有编码 bug），mathtext cm + Palatino；PNG 为无坐标轴裸图、`origin='lower'`、dpi=150 |

## 2 方法与约定

### 2.1 网格、单位与 q(h,k)

```
r ∈ [0,N)^2，数组坐标 (row = y, col = x)；画布 N x N；nm/px = L/N
q(h,k) = h * b1 + k * b2            任意实数 h,k（支持 '2/3,1/3' 分数写法）
```

三种 q 表示（同一向量，报告里都给）：

| 名称 | 单位 | 定义 |
| --- | --- | --- |
| `q_rad_px` | rad/px | 引擎实际使用的角频率：`b_rad_px = b_nm_inv * nm_per_px`（或 `--basis-px` 时 `b_rad_px = b_px * 2π/N`） |
| `q_px` | px（有符号 fftshift 偏移） | `q_px = q_rad_px * N / (2π)`，与 `bragg_peak` 的 `q_px` 同口径 |
| `q_nm_inv` | rad/nm | `q_nm_inv = q_rad_px / nm_per_px`；`|q|` 即 `|q_nm_inv|` |

环成员（只用半径比，**严禁**任何 k 空间高对称点命名）：

```
ring_1x1  ±(1, 0), (0, 1), (1, -1)
ring_r3   ±(1/3, 1/3), (2/3, -1/3), (1/3, -2/3)     半径 = ring_1x1 的 1/sqrt(3)
```

### 2.2 局域复场（引擎）

```
psi_q(r) = FFT^-1{ FFT[ T(r) * exp(-i q.r) ] * exp(-Lambda^2 |k|^2 / 2) }
```

`|k|` 是 FFT 网格的角频率（rad/px），因此实现里用 `Lambda_px = Lambda_nm / nm_per_px`；
`k = 0` 处窗权重恰为 1。对含 `A cos(q.r + phi_q)` 的 `T`：

```
|psi_q(r)| ≈ A_q / 2                    幅度图
theta_q(r) = arg psi_q(r) ∈ (-pi, pi]   **解调约定**：theta ≈ +phi_q，没有 q.r 斜坡
```

### 2.3 λ 的含义与默认值（易错，必须区分）

| | 本 skill `--lambda-nm` | `lawler-fujita-correction` `--lambda-nm` |
| --- | --- | --- |
| 默认 | **3.0 nm**（论文 SI §5.1 的 30 Å） | 30.0 nm |
| 含义 | 局域傅里叶的**实空间窗宽**：q 空间低通 `exp(-λ²k²/2)`，窗宽 `1/λ` | lock-in 低通保留的**畸变尺度**，决定可解应变上限 `|∇u| < 2π/(|K|λ)` |

`--basis-from` 指向的矫正报告若含 `lockin.lambda_nm`（lawler-fujita 报告），log 会**仅作参考**打印它，
并明确标注 "reference only, NOT the window of this skill"。两者不要混用。

### 2.4 三个精确恒等式（self-test 用）

```
sum_r psi_q(r)          = sum_r T(r) exp(-i q.r)     窗宽无关，精确
psi_{-q}(r)             = conj(psi_q(r))             T 实值（Friedel）
psi_q(r - delta)        = exp(-i q.delta) psi_q(r)   图像平移 delta（振幅图逐位平移）
```

第三条对**任意**数组都精确；但当 q 与画布不可约（`q_px` 非整数）时，`np.roll` 的环绕会带入相位不一致的
数据，因此 self-test 用可约基矢验证它。常数调制相位 φ0 使整幅 theta 图整体平移 φ0（同样精确）。

### 2.5 适用条件（写出来是为了让 λ 不被误用）

1. **可分离**：任意两个 q 的距离要远大于窗宽 `1/λ`（脚本按 `|Δq| < 2/λ` 报 CROSS-TALK 警告）；
2. **慢变**：`|∇φ|, |∇ln A| ≪ 1/λ`；
3. **反向旋转项可弃**：`2λ²|q|² ≫ 1`；
4. **有限画布采样项与「分量间调制泄漏」（两种量级，勿混用）**。合成 fixture 的成员半径取偶数整数 px，
   对每个 `{b1,b2}` 成员都有 `2·q_px_x ∈ ℤ`，因此**负频副本精确落在 FFT 的整数列上**、在窗邻域内
   精确缺席（除 `±(1/3,−2/3)` 一对，其副本落在 `k_x = 0` 列）：
   - **单成员自项**：单个成员的解调场与 `(A/2)e^{iφ}` 之差——全画布均值与逐点最大偏离都约 1e-14；
     `±(1/3,−2/3)` 例外，逐点偏离约 2.5e-3（N=512）。
   - **分量间调制泄漏（12 个 q 同时存在时主导）**：其余五个成员经窗与有限画布展宽漏进本 q，
     与「自项」无关。N=512 实测两个不同量级：
     **全局统计量**（各 q 的幅度中位数误差 6.1e-6、画布平均场的相位误差 0.058°）
     vs **逐点量**（worst pointwise ripple `max|psi(12 q) − psi(本 q 单独)|/median|psi|`
     = 3.63e-2 ≈ 18.6/N）。
   - **用法含义**：逐像素量（相位差图、三重积、逐像素相位统计）必须用**逐点**水位
     （实测 10–21/N，N=512 时 3.6e-2），不能用 6.1e-6 / 0.058° 当场的误差地板；
     只有全局量（幅度中位数、幅度加权圆均值/中位数）才用 6.1e-6 / 0.058°。
   真实数据 N≈1000–2000：逐点水位 ~1e-2 量级、全局量更小，两者都被噪声淹没。
   self-test check 2 对两种量级**都**有断言（`RIPPLE_REL_TOL` 与 `24/N` 上界）。

## 3 基矢来源（`--basis-from REPORT.json`，按 key 自动识别）

| 报告类型 | 判定 key | 取法 |
| --- | --- | --- |
| lawler-fujita-correction | `q_a_nm_inv` 与 `q_b_nm_inv`（120° 对） | `b1 = q_a`，`b2 = q_a + q_b = -q_c`；log 打印实测夹角与 `q_c` 闭合残差。**取向**默认由画布决定（见下），报告的 `q_a` 只提供半径与 60° 标签规范 |
| affine-correction | `orientation_deg` 且（`b1_measured_nm_inv_after` 或 `b1_ideal_nm_inv` 或 `implied_lattice_after.a_1x1_nm`） | `b1 = R(cos t, sin t)`，`b2 = b1` 逆时针转 +60°（复用 `bragg_peak.lattice_fit.rotate_basis`，导入失败时用同公式的本地副本）。半径键描述的是报告**锚定环**：`anchor_ring = r3` 时先换算回 1×1 半径（优先 `implied_lattice_after.a_1x1_nm` → `4π/(√3·a_1x1)`，否则 `√3 ×` 半径键）；无 `anchor_ring` 而 `rings_after` 给出 1 : 1/√3 环对时，半径键落在内环即判定为 r3 并同样换算。log 与 `basis_source.notes` 打印解码出的环身份；解出的 `|b1|` 仍落在报告的 r3 环上时（`rings_after` 交叉核对）打印 WARNING。**取向**默认由画布决定（见下），报告的 `orientation_deg` 只提供半径与 60° 标签规范 |
| 两者都不匹配 | — | **非零退出**，错误信息列出可识别的 key 与 `--basis-px` 逃生口 |

**取向语义（关键，review F6）**：报告只可靠地给出基矢的**半径**，不给出本画布的取向——
`affine-correction` 的 `orientation_deg` 与 LF 报告的 `q_a`/`q_b` 都是**矫正的输入画布**（INPUT frame）方向，
而本 skill 分析的是**矫正画布**（真实 topo4：报告 `orientation_deg = 30.7998°` 是输入帧，矫正画布自己的 1×1 环在 0.24°）。
因此默认（`--basis-orientation-from canvas`）由**矫正画布自身**定取向：

1. 对输入 CSV 跑一次 `bragg_peak.detect_bragg_peaks`（与姊妹 skill 同一调用、**同一数组**——不做
   flipud/去平面，与解调同口径；为效率限定 `q_max_px = 1.5 ×` 1×1 半径、`max_candidates = 512`，
   只找环_1x1 邻域；真实 2k 画布约 3–4 s，512² 合成画布约 0.1 s）；
2. 取半径带 `±15 %` 内成员（`index_hk` 标签**不使用**，只用成员方位角）的方位角，
   按六方环的 **60° 周期**做幅度加权圆均值，得到相对报告取向的方位差 `delta_deg`
   （log 打印成员数、方位差与 spread；成员逐个连同 `q_px` / 半径 / 方位角 / 幅度进 `basis_source.canvas_orientation.members`）；
3. `|delta_deg| > 1°` 时把 `{b1, b2}` **整体刚性旋转** `delta_deg`（半径、60° 关系、其余所有报告数值不变，
   `basis_source.notes` 与 log 记录该旋转）；`|delta_deg| ≤ 1°` 时**报告取向逐位保留**（已有产物不变）；
4. 画布上找不到 1×1 成员（或检测器不可用）时保留报告取向，并在 `basis_source.orientation_source` 与 log 里写明原因。

六方环的取向只在 **60° 周期**内可测，所以「哪个成员叫 `(1,0)`」本质上是标签规范（gauge）：
差接近 30° 时两支等价，实现取「离报告 b1 最近的画布成员」这一支（因此 (h,k) 标签可能相对报告整体换一个成员，
log 打印解析出的 b1 角度便于核对）。`--basis-orientation-from report` 可强制使用报告自身框架（对照/复现用；
对旋转的 affine 报告会落回错误取向，此时下面的无条件校验会拦住它）。

**无条件基矢-画布校验（两者分支与 `--basis-px` 都做）**：任何产物写出之前，把解析出的 1×1 基矢与画布比对——
半径带 `±10 %`（annulus）内画布谱幅度的最强点即画布自己的环；要求六个 ring_1x1 成员处谱幅度 ≥ 该峰值的 20 %
（即 `|psi|` 之比：`psi_q = ifft2(window · FFT)`，1/N² 因子在比值中约掉），且成员的弧距离小于
`1.79 × N/(2πλ_px)` px（`1.79 = sqrt(2 ln 5)` 正是窗把成员衰减到同一 20 % 的偏移）。失配时：
log 打 `WARNING: basis-vs-canvas check: MISMATCH ...`（同时给出解析取向与实测取向、比值、偏移与阈值），
报告记 `warnings.basis_ring_mismatch = true` 与全套数值；**默认的** `--strict` 则**不写任何逐 q 产物**、退出码 **3**
（`--no-strict` 降级为「只 WARNING + 报告标志、仍出产物」）。
真实 topo4 的对照：报告框架下成员只达环峰 0.494 %、偏 125 px ⇒ 被拦；画布取向下达 100 %、偏 0.2 px ⇒ 通过。

**逃生口**：`--basis-px 'b1x,b1y;b2x,b2y'`——矫正画布上的**有符号 fftshift 偏移（px）**。

**单位换算**：`q_rad_px = q_nm_inv * corrected_nm_per_px`（优先用报告里的 `corrected_nm_per_px`，
否则用画布 `L/N`）；log 会打印实际使用的 nm/px 与来源。

## 4 CLI 与输出契约

### 4.1 命令行

```bash
cd /path/to/STM_DataProcessing
export MPLCONFIGDIR=<可写目录> PYTHONDONTWRITEBYTECODE=1

.venv/bin/python skills/local-q-map/scripts/stm_local_q_map.py \
  CORRECTED.csv -o OUT \
  --basis-from OUT_CORRECTION/correction_report.json \
  --q 1,0 --q 0,1 --q 1,-1 --q 1/3,1/3 --q 2/3,-1/3 --q 1/3,-2/3 \
  -L 50   # 或 --size-nm-from-log correction.log
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--basis-from` / `--basis-px` | 必给其一 | 基矢来源（互斥） |
| `--q 'H,K'` | 必给至少一个 | 可重复；分数写法 `2/3,1/3`；按给出顺序标为 `q0, q1, ...`，`h,k` 原样进报告 |
| `-L` / `--size-nm-from-log` | — | 视场来源，优先级见下 |
| `--lambda-nm` | 3.0 | 局域傅里叶窗宽（nm），见 §2.3 |
| `--amplitude-fraction` | 0.10 | 掩码阈值 = 该比例 × `median(|psi|)` |
| `--basis-orientation-from` | `canvas` | 报告基矢的**取向**来源：`canvas` 用矫正画布自身测（见 §3）；`report` 强制用报告自身框架（对照用；旋转的 affine 报告会因此取错取向） |
| `--strict` / `--no-strict` | `--strict` **开** | 基矢-画布校验失配时**不写逐 q 产物**并以退出码 **3** 结束；`--no-strict` 降级为「只 WARNING + 报告标志」（仍出产物、退出 0） |
| `--delimiter` | `,` | CSV 分隔符 |
| `--no-figures` | 关 | 只写 npy，不写 PNG |
| `--stm-lib` | 仓库 `src` | `bragg_peak` 约定与 `rotate_basis` 的来源 |

**视场优先级（与 `phase-analysis` 一致并多一级）**：
`-L` → `--size-nm-from-log`（`# corrected canvas: ... field of view <X> nm` 行胜出，否则取最后一条
`field of view <X> nm`，都没有则报错退出）→ `--basis-from` 报告的 `corrected_nm_per_px` × 画布
→ 报告的 `corrected_field_of_view_nm` → 都没有则报错退出。

### 4.2 产物（`-o OUT`，`stem` = 输入 CSV 的文件名主体）

每个 q（`j = 0, 1, ...`）：

| 产物 | 内容 | PNG |
| --- | --- | --- |
| `<stem>_q{j}_field.npy` | **主产物**：`complex128` 的 `psi_q(r)` | — |
| `<stem>_q{j}_amplitude.npy` / `.png` | `|psi_q|`（float64，全有限） | inferno |
| `<stem>_q{j}_theta.npy` / `.png` | 弧度、折叠在 `(-pi, pi]` | twilight，`vmin=-180, vmax=180`，值 = `np.degrees(np.angle(np.exp(1j*data)))` |
| `<stem>_q{j}_mask.npy` / `.png` | **有效掩码**（float 0/1，**1 = 有效**） | gray，vmin=0，vmax=1 |

共享：`local_q_map.log`、`local_q_map_report.json`。

**NaN 规则**：引擎把输入 NaN 像素填 **0** 后再做 FFT；掩码 =
`（输入 NaN 区域）∪（|psi_q| < amplitude_fraction × median(|psi_q|)）`，报告里 `mask_coverage_fraction`
是**有效像素占比**（与 `lawler-fujita-correction` 的 `lockin.mask_coverage_fraction` 同义），
`masked_fraction = 1 - 它`。幅度/相位 PNG 里无效像素画成 NaN（bad 灰），但**所有 npy 都保持有限**。

### 4.3 报告 JSON

顶层：`skill`、`skill_version`、`input`、`canvas_px`、`field_of_view_nm`、`nm_per_px`、
`field_of_view_source`、`nan_fraction`、`lambda_nm`、`lambda_px`、`amplitude_fraction`、
`basis_source{kind,path,detected_keys,notes,b1_nm_inv,b2_nm_inv,b1_rad_px,b2_rad_px,b1_px,b2_px,
abs_b1_nm_inv,b1_angle_deg,b2_angle_deg,angle_between_deg,basis_nm_per_px,basis_conversion,
orientation_source,orientation_from_canvas,orientation_delta_deg,canvas_orientation{status,
delta_deg,orientation_deg,n_members,spread_deg,member_angles_deg,members[...],radius_px,
band_fraction,patch_half,max_candidates,detector,message},basis_canvas_check{band_fraction,
radius_px,member_half_px,strength_fraction,offset_factor,window_radius_px,annulus_pixels,
annulus_peak_px,annulus_peak_radius_px,annulus_peak_angle_deg,measured_angle_deg,
resolved_angle_deg,annulus_strength,member_strength,strength_ratio,worst_offset_px,
offset_tolerance_px,mismatch,reason,summary},
report_lockin_lambda_nm,...}`、`warnings{cross_talk_pairs,cross_talk,boundary_warning,border_band_px,
basis_warnings,basis_ring_mismatch,basis_ring_mismatch_rule,basis_ring_strength_ratio,
basis_ring_member_strength,basis_ring_annulus_strength,basis_ring_annulus_peak_px,
basis_ring_measured_angle_deg,basis_ring_resolved_angle_deg,basis_ring_worst_offset_px,
basis_ring_offset_tolerance_px,nm_per_px_mismatch,nm_per_px_ratio,canvas_nm_per_px,basis_nm_per_px,...}`、
`rings{ring_1x1_members_hk,ring_r3_members_hk,naming}`、`conventions{...}`、`q[...]`、`written`。

每个 q 条目：`label`、`h`、`k`、`q_px`、`q_rad_px`、`q_nm_inv`、`q_nm_inv_abs`、`lambda_nm`、`lambda_px`、
`amplitude_fraction`、`amplitude_threshold`、`nan_fraction`、`mask_coverage_fraction`、`masked_fraction`、
`n_valid_pixels`、`amplitude{median,fwhm,max,min_valid}`、
`theta{circular_mean_deg, circular_mean_all_pixels_deg, mean_identity_error_deg, circular_median_deg,
circular_median_span_deg, n_median_minimisers, resultant_R}`、`artifacts{...}`、`conventions{...}`。

口径说明：

- **幅度加权圆均值 = `arg sum psi`**（权重取自身幅度时，`Σ w_i e^{iθ_i} = Σ psi_i`）：报告里
  `circular_mean_deg` 是有效像素上的值，`circular_mean_all_pixels_deg` 是全画布值（后者才与恒等式对应）；
  `mean_identity_error_deg` 是「全画布 `arg sum psi`」与「`arg sum T exp(-i q.r)`」之差（自检，机器精度）。
- `resultant_R = |Σ_valid psi| / Σ_valid |psi|`（幅度加权圆集中度）。
- `circular_median_deg` 是加权圆中位数（最小化 `Σ w_i |wrap(t - θ_i)|`）；`amplitude.fwhm` 是幅度分布
  直方图的半峰宽（无下降沿时记 NaN，属正常结果）。
- `conventions` 内写明 theta 的符号/单位/PNG 口径与掩码语义。

### 4.4 log 关键行（契约行）

```
# local-q-map v1.0
# input: <csv>
# canvas <n> x <n> px, field of view <L> nm (<nm/px> nm/px), NaN <..> %
# field of view source: <...>
# basis source: <kind> (<report path>)
# basis keys used: <...>
# basis note: ...
# basis b1 = (...) nm^-1 = (...) rad/px = (...) px, |b1| = ... at ... deg
# basis b2 = ... ; angle between b1 and b2 = ... deg (60 deg expected)
# basis nm/px used for the rad/nm -> rad/px conversion: ... (<provenance>)
# basis orientation source: <corrected canvas (...) | report orientation (...)>
# basis orientation: <n ring_1x1 member(s) within 15 % of <R> px; the amplitude-weighted azimuth difference to the report orientation is <delta> deg (spread <..> deg)>
# basis orientation: the report orientation is replaced by the corrected canvas (rotation <delta> deg applied, radius unchanged)   [仅当 |delta| > 1°]
# basis-vs-canvas check: annulus <R> px +/- 10 %, annulus peak <..> at <..> px = <..> deg (folded <..> deg), the resolved members reach <..> -> ratio <..> (threshold 0.2), worst member offset <..> px (tolerance <..> px) -> OK
# WARNING: basis-vs-canvas check: MISMATCH -- <resolved 与 measured 两个取向、比值、偏移与阈值>
# basis-source report lockin.lambda_nm (reference only, NOT the window of this skill): 30.0000 nm
# window lambda = 3.0000 nm = 30.7200 px, amplitude fraction = 0.1000
# boundary: the periodic FFT pollutes the demodulated maps within about 0.65 * lambda = ... px of the border
# q0: (h, k) = (1, 0) -> q_px (...), |q| = ... nm^-1, amplitude threshold ...
# q0: amplitude median ..., FWHM ...; theta circular mean ... deg, R = ..., mean-identity error ... deg
# q0: mask coverage ... % (nan fraction ... %)
# written: ...
```

### 4.5 警告

| 触发 | 行为 |
| --- | --- |
| 任意两个 q 的 `|Δq|` < `2/λ`（单位 rad/nm，rad 无量纲） | log 打 `WARNING: ... CROSS-TALK`，报告记 `warnings.cross_talk_pairs` 与 `warnings.cross_talk` |
| `λ > L/4` | log 打 `WARNING: lambda ... exceeds L / 4 ...`，报告记 `warnings.boundary_warning = true` |
| 解出的 `|b1|` 仍落在报告的锚定 r3 环上（`rings_after` 交叉核对） | log 打 `WARNING: the resolved |b1| ... is the r3 ring, not the ring_1x1 ring`，报告记 `warnings.basis_warnings`（同样的字符串也在 `basis_source.basis_warnings`） |
| `|basis_nm_per_px / (L/N) − 1| > 1e-3`（画布与基矢来源的 nm/px 不一致） | log 打 `WARNING: nm/px mismatch ...` 并分别给出画布值与报告值及各自来源，报告记 `warnings.nm_per_px_mismatch = true`（附 `nm_per_px_ratio`、`canvas_nm_per_px`、`basis_nm_per_px`）；仍按画布值报 `q_nm_inv` |
| 解析出的 1×1 基矢与画布**失配**（成员处谱幅度 < annulus 峰的 20 %，或成员弧偏移 > `1.79 × N/(2πλ_px)` px） | log 打 `WARNING: basis-vs-canvas check: MISMATCH ...`（给出解析取向、实测取向、比值、偏移与阈值），报告记 `warnings.basis_ring_mismatch = true` 与 `basis_source.basis_canvas_check`；**默认的** `--strict` 即**不写任何逐 q 产物**、退出码 **3**（`--no-strict` 降级为只告警） |
| 画布上测不到 1×1 成员（或检测器不可用） | 取向保留报告值，`basis_source.orientation_source` 与 log 写明 `no_members` / `no_detector`；基矢-画布校验仍照常执行（无证据时不误报） |
| 边界污染 | **不作硬性剔除**：log 与报告给出边界带 `0.65 λ` px（周期性 FFT 把对边数据混入解调场），用户自行裁边 |

## 5 一键 self-test（11 项）

```bash
cd /path/to/STM_DataProcessing
MPLCONFIGDIR=<可写目录> PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python skills/local-q-map/scripts/selftest.py
# 可选：--workdir DIR、--size N（默认 512）、--keep、--stm-lib DIR
```

合成真值：六方基矢 `|b1| = 2·round(0.195 N) px`（N=512 时 200 px；**偶数整数 px**，随 `--size` 缩放，
故 (1,0) 与画布严格可约、且半径在 Nyquist 以内）、L = 50 nm 的 6 个调制
（ring_1x1 三个 + ring_r3 三个，各自 A 与 φ 不同），以及它们的 6 个 Friedel 伙伴，共 **12 个 q**。
实图像的 Friedel 对不可能独立给幅相，因此 `-q` 的相位严格是 `+q` 的相反数——self-test 按此核对。

| # | 检查 | 验收阈值（实测） |
| --- | --- | --- |
| 1 | 单平面波：`|psi| = A/2` 且 `theta = +phi0`（钉住符号约定，且落在可约成员上，采样项为零） | 相对 ≤ 1e-3（4.4e-16）；相位 ≤ 0.01°（5.3e-13°） |
| 2 | 12 个 q 各恢复自己的 A 与相位（**全局统计量**：幅度中位数误差、画布平均场相位误差）；六个调制之间的**逐点**调制泄漏；均值串扰 | 幅度 ≤ 1e-3（6.1e-6）；相位 ≤ 0.5°（0.058°）；逐点 ripple ≤ max(5e-2, 24/N)（3.63e-2 = 18.6/N）；均值串扰 ≤ 2e-2（5.5e-3，解析窗界 ≈ 0） |
| 3 | 同一向量用 `(h,k)` 与显式 `--basis-px` 给出，映射逐位一致 | 相对 ≤ 1e-12（0） |
| 4 | 基矢解析：lawler-fujita 报告 / affine 报告（两级半径退化）/ affine 报告 `anchor_ring=r3`（含无 `anchor_ring` 时用 `rings_after` 的 1 : 1/√3 环对推断，以及 r3 落点告警的交叉核对）/ **旋转 affine 报告**（`orientation_deg` 偏 30.8°，默认被画布重新取向且 `|psi|` 落在峰上；`--basis-orientation-from report` 复现失配 → WARNING + 标志 + `--no-strict` 出产物（exit 0）、默认 `--strict` 退出 3 且不写逐 q 产物）/ 无匹配 → 非零退出 | 相对 ≤ 1e-12；取向折到画布环成员（≤ 0.01°）；旋转支 `|psi|` 中位数 0.425（报告框架 2.4e-9，差 1.8e8 倍）；退出码 ≠ 0 且信息明确 |
| 5 | 规范律：常数 φ0 使 theta 图整体平移 φ0；平移图像使各图平移 δ（含精确规范常数 `exp(-i q.δ)`） | ≤ 0.01°（8.4e-12°）；相对 ≤ 1e-9（6.0e-14） |
| 6 | Friedel 恒等式 `theta_q + theta_-q = 0 mod 2π` | ≤ 1e-6°（2.5e-14°） |
| 7 | 均值恒等式 `arg sum psi_q = arg sum T e^{-i q.r}` | ≤ 1e-9°（0） |
| 8 | NaN：fill-0 与 fill-plane 的谱只差缺口本身；fill 值扰动 < 缺口占比；NaN 区全部被掩码标记，npy 全有限 | 谱相对 ≤ 1e-7（2.3e-16）；扰动 < 缺失占比（2.2e-5 < 6.1e-3） |
| 9 | 确定性：两次运行产物**逐字节相同** | 14/14 文件 |
| 10 | 视场来源：`--size-nm-from-log` 三支 + `--basis-from` 报告的 `corrected_nm_per_px` × 画布（不给 -L） | 103.3691 nm / 42.5 nm / 退出码 ≠ 0；报告支：视场 = `per_px × N`、`nm/px = per_px`、`λ_px = λ/per_px`、无边界警告 |
| 11 | 端到端：单 q 与多 q 两次完整运行 + 产物/报告/log 字段断言（含基矢来源的取向字段与 `basis_canvas_check`）+ 警告支（串扰、边界、nm/px 不一致、以及一致时不告警）+ **基矢-画布校验通过** + 加权圆中位数 | 21/21 文件、字段齐全、log 契约行齐全；`-L` 差 2 倍时 `nm_per_px_mismatch = true`；基矢-画布成员幅度比 100 %（阈值 20 %）、成员偏移 0 px（容差 4.75 px）、`basis_ring_mismatch = false`；双峰幅度样本上报告值 = 幅度加权中位数（两种定义差 60.5°） |

退出码 0 = 11 项全过（本版 46 条断言）；每项都打印实测值与其阈值。

## 6 已知限制

1. **相位折叠**：theta 保持 `(-pi, pi]`，本 skill 不解缠（与 `lawler-fujita-correction` 的 theta 产物同口径）。
2. **边界**：周期 FFT 使边界 `0.65 λ` 带内的解调场混入对侧数据；λ 越接近视场，可信区越小（`λ > L/4` 报警告）。
3. **有限画布采样项**：q 与画布不可约时存在 `~1/N` 的残差（见 §2.5 第 4 条），不是窗串扰。
4. **掩码语义**：`_mask.npy` 是**有效掩码**（1 = 有效）；`Σq=0` 之类的组合量必须只在有效像素上算。
5. **基矢：半径来自矫正产物，取向来自矫正画布**。本 skill 不重测环半径——报告给错半径，q 就错
   （log 会打印解析出的 b1/b2 与半径来源供核对）；报告锚定在 r3 环时半径键不是 1×1 半径，代码按
   `anchor_ring` 换算并用 `rings_after` 交叉核对；画布 nm/px 与报告 nm/px 不一致时打 `nm/px mismatch` 警告
   （仍按画布值报 `q_nm_inv`）。报告的 `orientation_deg` / `q_a` 属**输入帧**，不用于取向（§3）；
   任何残留的基矢-画布失配由无条件校验拦住（WARNING + 报告标志；默认 `--strict` 直接非零退出，`--no-strict` 降级为告警）。
6. **60° 标签规范（gauge）**：六方环只能把取向定到 60° 周期，所以「哪个成员叫 `(1,0)`」的标签规范由报告的
   取向提供，实现取「离报告 b1 最近的画布成员」；差接近 30° 时另一支与它等价（六个成员方向集合相同，
   `(h,k)` 标签整体换一位）。跨报告 / 跨画布比较 `(h,k)` 标签时必须核对 log 打印的 `b1` 角度与
   `basis_source.orientation_from_canvas`。
7. **窗口判据**：`|Δq| < 2/λ` 只报警告不中止；窗宽 `1/λ` 与「保留尺度 λ」是同一件事的两种说法，勿与 LF 的 λ 混用。

## 7 修改记录

2026-09（v1.0）：首个版本。§2 的单一数学定义（`psi_q = FFT^-1{FFT[T e^{-i q.r}] e^{-λ²|k|²/2}}`，
λ 默认 3.0 nm、解调约定 `theta ≈ +φ_q`）；`(h,k)` 表达任意 q 与 ring_1x1 / ring_r3 成员；
基矢来源自动识别（lawler-fujita / affine / `--basis-px`）与单位换算；NaN 填 0 规则与掩码；
每 q 的 field/amplitude/theta/mask 全套 npy+png 与机器可读报告；跨环串扰与边界警告；
11 项一键 self-test。见 `CHANGES.md`。

2026-09（v1.0 复核修复轮）：① `--basis-from` 报告的 `corrected_nm_per_px` 现在**乘以画布**得到视场
（此前误把 per-px 当视场，λ_px 会超出画布、局域场退化为全场解调、|q| 与边界警告全错）；
② affine 报告按 `anchor_ring` 解码半径（锚定 r3 环 → 换算回 1×1 半径），并用 `rings_after` 交叉核对
（仍落在 r3 环上则告警）；③ `circular_median_deg` 按幅度加权（与 §4.3 定义一致）；
④ 画布 `L/N` 与基矢来源 nm/px 不一致时新增 `nm/px mismatch` 警告。self-test 断言由 33 条扩到 43 条（仍 11 项）。
⑤ §2.5 第 4 条按「有限画布采样项 vs 分量间调制泄漏」两套量级重写（逐点 3.63e-2 ≈ 18.6/N 与全局
6.1e-6 / 0.058° 同时给出，并说明逐像素用法必须用逐点水位）；self-test check 2 新增逐点 ripple 断言
（`RIPPLE_REL_TOL` = 5e-2，配合 `24/N` 缩放，仍 11 项）。

2026-09（v1.0 复核修复轮 2 → 3，F6 blocker + F7 low）：报告的 `orientation_deg`（affine）与
`q_a`/`q_b`（LF）都属矫正的**输入帧**，直接拿来做基矢取向会在矫正画布上静默给出噪声（真实 topo4：
成员偏 128 px、`|psi|` 2.9e-15、exit 0 无警告）。修复：① **取向改由矫正画布自身测量**
（`bragg_peak.detect_bragg_peaks` 限定在 1×1 半径邻域，成员方位角按 60° 周期做幅度加权圆均值；
差 > 1° 时把 `{b1,b2}` 整体旋转过去，≤ 1° 时报告取向逐位保留；测不到时保留报告取向并写明原因），
新增 `--basis-orientation-from canvas|report`；② **无条件基矢-画布校验**（annulus 峰值 vs 六成员谱幅度 +
成员弧偏移，两分支与 `--basis-px` 都做），失配 ⇒ WARNING + `warnings.basis_ring_mismatch` +
`basis_source.basis_canvas_check` 全套数值，默认开的 `--strict` ⇒ 不写逐 q 产物并退出 **3**（新增 `--no-strict` 降级为告警）；
③ 文档写清取向/帧语义、`--basis-px` 工作流与 60° 标签规范，修正断言条数（43 → 46，仍 11 项），
并写明落盘时排除 `scripts/__pycache__/`（`py_compile` 验证步骤的副产物，非 skill 内容）。
self-test check 4 新增「旋转 affine 报告」fixture（默认重新取向 + 强制报告框架的失配：`--no-strict` 出产物且带标志、默认 `--strict` 退出 3 不写产物），
check 11 新增基矢-画布校验通过的断言。
