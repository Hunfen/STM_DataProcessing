# CHANGES —— local-q-map

## 2026-09（v1.0，首个版本）

### 目标

对**矫正后**拓扑 CSV，用矫正产物给出的 1×1 六方倒格基矢 `{b1, b2}` 表达任意 `q(h,k)`，
做 Gaussian 窗局域傅里叶滤波，输出每个 q 的复数局域场与由它派生的幅度图、解调相位图与有效掩码。
只做几何与信号处理，不做物理解释；环只用 `ring_1x1` / `ring_r3` 命名。

### 单一数学定义（写进 SKILL.md §2）

```
psi_q(r) = FFT^-1{ FFT[ T(r) exp(-i q.r) ] * exp(-Lambda^2 |k|^2 / 2) }
Lambda  = --lambda-nm，默认 3.0 nm（论文 SI §5.1 的 30 Angstrom）
|psi_q| ≈ A_q / 2        theta_q = arg psi_q ∈ (-pi, pi] ≈ +phi_q（解调约定，无 q.r 斜坡）
```

- `|k|` 为 FFT 网格角频率（rad/px），实现用 `Lambda_px = Lambda_nm / nm_per_px`；`k = 0` 权重恰为 1。
- **与 `lawler-fujita-correction` 的 λ 严格区分**：本 skill 的 λ 是局域傅里叶窗宽（默认 3 nm），
  LF 的 λ 是 lock-in 保留的畸变尺度（默认 30 nm）。`--basis-from` 指向 LF 报告时，
  log 仅作参考打印报告的 `lockin.lambda_nm` 并标注 "reference only"。

### 新增内容

1. **q 的三种表示**：`q_rad_px`（引擎用）、`q_px`（有符号 fftshift 偏移，与 `bragg_peak` 同口径）、
   `q_nm_inv`（rad/nm）；`q(h,k) = h b1 + k b2`，`h,k` 为任意实数（支持 `2/3,1/3` 写法）。
   环成员：`ring_1x1 = ±(1,0),(0,1),(1,-1)`；`ring_r3 = ±(1/3,1/3),(2/3,-1/3),(1/3,-2/3)`。
2. **基矢来源自动识别（key-based）**：
   - lawler-fujita 报告：`b1 = q_a_nm_inv`，`b2 = q_a + q_b = -q_c`；log 打印实测 120° 夹角与 `q_c` 闭合残差；
   - affine 报告：`b1 = R(cos t, sin t)`，`R` 依次退化 `b1_measured_nm_inv_after` → `b1_ideal_nm_inv`
     → `4π/(√3·a_1x1)`（`implied_lattice_after.a_1x1_nm`），`b2 = b1` 逆时针 +60°
     （复用 `bragg_peak.lattice_fit.rotate_basis`，导入失败时用同公式本地副本）；
   - 都不匹配：非零退出 + 明确错误信息；
   - 逃生口 `--basis-px 'b1x,b1y;b2x,b2y'`（画布上有符号 fftshift 偏移，px）。
   - `q_rad_px = q_nm_inv * corrected_nm_per_px`（优先报告自己的 nm/px，否则画布 `L/N`）。
3. **视场来源**：`-L` → `--size-nm-from-log`（corrected canvas 行胜出，否则最后一条 `field of view`，
   都没有则报错退出）→ 报告的 `corrected_nm_per_px` × 画布 → 报告的 `corrected_field_of_view_nm`
   → 报错退出。（比 `phase-analysis` 多两级报告回退，使 `--basis-from` 可单独使用。）
4. **NaN 与掩码**：引擎把 NaN 填 0 后做 FFT；掩码 = 输入 NaN 区域 ∪ `|psi| < amplitude_fraction ×
   median(|psi|)`；所有 npy 保持有限；报告 `nan_fraction` 与 `mask_coverage_fraction`（有效占比）。
5. **产物**（每 q）：`<stem>_q{j}_field.npy`（**主产物**，complex128）+ `_amplitude.npy/.png`（inferno）
   + `_theta.npy/.png`（弧度；PNG 为 `degrees(angle(exp(1j·data)))`，twilight，`±180`）
   + `_mask.npy/.png`（有效掩码，gray）；共享 `local_q_map.log`、`local_q_map_report.json`。
6. **报告字段**：顶层设置 + 基矢来源全字段 + 警告 + `rings` 命名 + `conventions`；
   每 q 的 `h,k,q_px,q_rad_px,q_nm_inv,|q|,lambda_nm,lambda_px,amplitude{median,fwhm,max,min_valid},
   theta{circular_mean_deg,circular_mean_all_pixels_deg,mean_identity_error_deg,circular_median_deg,
   resultant_R},nan_fraction,mask_coverage_fraction,artifacts,conventions`。
   「幅度加权圆均值 = `arg sum psi`」；`mean_identity_error_deg` 是它与 `arg sum T e^{-i q.r}` 之差。
7. **警告**：任意两 q 的 `|Δq| < 2/λ`（rad/nm）→ CROSS-TALK WARNING；`λ > L/4` → 边界 WARNING；
   边界污染带 `0.65 λ` px 只报告不裁剪。
8. **一键 self-test（11 项）**：单平面波幅相/符号约定、12 个 q 的幅相（全局）与逐点调制泄漏、
   `(h,k)` 与 `--basis-px`
   逐位一致、基矢解析（lawler-fujita / affine / affine `anchor_ring=r3` / 无匹配）、规范律（常数 φ0 与图像平移，
   含精确规范常数 `exp(-i q.δ)`）、Friedel 恒等式、
   均值恒等式、NaN 规则与掩码、两次运行逐字节确定性、视场来源（`--size-nm-from-log` 三支 +
   报告 `corrected_nm_per_px` × 画布）、端到端产物契约（含 nm/px 不一致告警与幅度加权圆中位数）。

### 复核修复轮（v1.0，review round 1 → 2）

独立复核（review r1）确认引擎数学与约定无误，但 CLI 的「视场 / 基矢解析链」有 4 处缺陷，本轮修复：

1. **F1（blocker）报告的 nm/px 被当成视场**：`--basis-from` 且不给 `-L` / `--size-nm-from-log` 时，
   `field_of_view()` 直接返回报告的 `corrected_nm_per_px`（per-px）而不是它 × 画布。后果：
   视场 0.0488 nm、`nm/px` 2.3e-5、`λ_px = 130068`（> 画布），窗权处处 ≈ 1，**主产物 field.npy
   退化为全场解调场**，`|q|` 报成 62387 nm⁻¹（真值 29.49），并误报边界警告。
   现改为 `size_nm = corrected_nm_per_px × n_px`（优先级 `-L` → `--size-nm-from-log` →
   报告 per-px × 画布 → 报告 `corrected_field_of_view_nm` → 报错），self-test 新增该分支断言。
2. **F2（high）affine 报告的半径键是「锚定环」半径**：真实 affine 报告带 `anchor_ring`
   （topo4 为 `r3`），`b1_measured_nm_inv_after` 是 r3 环半径（17.076 nm⁻¹），不是 1×1 环半径
   （29.4635 nm⁻¹），原先直接当 1×1 用，`(1,0)` 被投到 r3 环上。现按 `anchor_ring` 解码：
   r3 时优先 `implied_lattice_after.a_1x1_nm` → `4π/(√3·a_1x1)`，否则 `√3 ×` 半径键；无
   `anchor_ring` 时用 `rings_after` 的 1 : 1/√3 环对推断；解出的 `|b1|` 仍落在 r3 环上
   （`rings_after` 交叉核对）则 WARNING。log 与 `basis_source.notes` 打印解码出的环身份。
3. **F3（medium）`circular_median_deg` 未加权**：SKILL.md §4.3 定义为幅度加权圆中位数
   （最小化 `Σ w_i |wrap(t − θ_i)|`），CLI 却传了无权重的 `theta[valid]`。现传
   `weight=amplitude[valid]`（与 `phase-analysis` 的 `circ_median(phi, w)` 同口径）；
   双峰幅度合成样本上两种定义差 60.5°，self-test 断言报告值 = 加权值。
4. **F4（low）画布与基矢来源的 nm/px 无一致性保护**：现比较 `basis_nm_per_px` 与 `L/N`，
   `|ratio − 1| > 1e-3` 时打 `WARNING: nm/px mismatch ...`（给出两个值与各自来源），
   报告记 `warnings.nm_per_px_mismatch`（附 `nm_per_px_ratio`、`canvas_nm_per_px`、`basis_nm_per_px`）。

self-test 断言由 33 条扩到 **43 条**（仍 11 项，全部通过）；新增 F1/F2/F3/F4 四个分支的直接覆盖。

### 复核修复轮（v1.0，review round 2 → 3：F6 blocker + F7 low）

独立复核（review r2）确认 F1–F5 全部修好且引擎数学无误，但 F2 的「半径」修复只解决了一半：
**报告的取向同样不属于矫正画布**。

- **F6（blocker）affine 报告的 `orientation_deg` 是输入帧**。真实 topo4 的 affine 报告
  `orientation_deg = 30.7998°` 与其**输入**画布的环一致，而**矫正画布**自己的 1×1 环（`rings_after[0]`
  = 242.477 px）在 0.24°；LF 报告的 `q_a`/`q_b` 同属输入帧的 lock-in 方向。后果：解析出的基矢
  三个 ring_1x1 成员偏离真实峰 ~128 px，CLI **exit 0 且无任何警告**地输出噪声地板
  （`|psi|` 2.9e-15…6.3e-15、R 0.066…0.314），而同一画布用 `--basis-px '242,1;121,209.6'` 给的是
  1.96e-12、R 0.72（~670 倍且相干）。修复分两部分：
  1. **取向从矫正画布自身导出**：新增 `localqmap.canvas_ring_orientation()`——对输入 CSV（同一数组、
     不 flipud/去平面）调用 `bragg_peak.detect_bragg_peaks`（与姊妹 skill 同一调用；为效率限定
     `q_max_px = 1.5 ×` 1×1 半径、`max_candidates = 512`），取半径带 ±15 % 内成员的方位角，
     按六方环 60° 周期做幅度加权圆均值，得到相对报告取向的 `delta_deg`；`|delta| > 1°` 时用
     `localqmap.rebase_orientation()` 把 `{b1,b2}` **整体刚性旋转**（半径、60° 关系、所有报告数值不变），
     `|delta| ≤ 1°` 时报告取向**逐位保留**（已有产物与断言不变）。测不到成员时保留报告取向并在
     `basis_source.orientation_source` / log 写明 `no_members` / `no_detector`。
     新增 `--basis-orientation-from canvas|report`（默认 `canvas`，`report` 为对照/复现用）。
     真实数据：topo4 现落在环上（`|psi|` 1.79e-12、R 0.44–0.63），topo0009 LF 的 `delta = +0.0026°`
     落在 1° 容差内 ⇒ 取向逐位不变（round 2 的 LF 结论保持）。六方环只能定到 60°，故差接近 30° 时
     标签规范（哪个成员叫 (1,0)）本质歧义，实现取「离报告 b1 最近的画布成员」，写进 SKILL §6 第 6 条。
  2. **无条件基矢-画布校验**：新增 `localqmap.basis_canvas_check()`，对**两个分支与 `--basis-px`**
     都在写产物前比对——半径带 ±10 %（annulus）内画布谱幅度最强点即画布自己的环；六成员处谱幅度
     必须 ≥ 该峰的 20 %（`|psi|` 之比，1/N² 因子约掉），成员弧偏移必须 < `1.79 × N/(2πλ_px)` px
     （`1.79 = sqrt(2 ln 5)`，正是窗把成员衰减到 20 % 的偏移）。失配 ⇒ log
     `WARNING: basis-vs-canvas check: MISMATCH ...`（给出解析取向、实测取向、比值、偏移与阈值）+
     `warnings.basis_ring_mismatch = true` + `warnings.basis_ring_*` 全套数值 +
     `basis_source.basis_canvas_check`；新增 `--strict`（**默认开**，配 `--no-strict` 逃生口）⇒ **不写任何逐 q 产物**
     并以退出码 **3** 结束，`--no-strict` 则降级为「只 WARNING + 报告标志、仍出产物」。
     无证据的情形（annulus 内无 FFT 像素、常值画布）记 `mismatch = false` 并写明原因，不误报。
- **F7（low）文档、条数与落盘**：SKILL.md §3/§4.1/§4.4/§4.5/§6 与 README 写清「报告给半径、画布给取向」
  的帧语义、`--basis-px`/`--basis-orientation-from`/`--strict` 用法与 60° 标签规范；
  断言条数改为与实际输出一致（43 → 46）；落盘清单排除 `scripts/__pycache__/`（`py_compile` 验证步骤的
  副产物），README §1 与 SKILL §7 写明。

self-test：check 4 新增「旋转 affine 报告（`orientation_deg` 偏 30.8°）」fixture——默认路径断言
`|b1|` 不变、b1 折到画布环成员、`|psi|` 中位数 0.425（报告框架 2.4e-9，差 1.8e8 倍）且无失配；
强制报告框架断言 `basis_ring_mismatch = true`、log 有 MISMATCH 行、成员幅度比 < 20 %、`--no-strict` 仍出 4 个 npy 且 exit 0，
而默认 `--strict` 退出 3 并**不写任何 npy**。check 11 新增「端到端通过基矢-画布校验」与取向字段齐全的断言。
断言由 43 条扩到 **46 条**（仍 11 项，全部通过）。

### 复核修复轮补充（F5，verifier 附加 finding，medium）

check 2 此前只断言**全局**统计量（幅度中位数误差 6.1e-6、画布平均场相位误差 0.058°），
而同一 check 算出的**逐点**污染 worst pointwise ripple = 3.63e-2（≈18.6/N，N=512）只打印、不断言，
下游逐像素用法会误把它当成 6e-6/0.06° 的误差地板。修复：

1. **新增断言**：check 2 增第 4 条「the pointwise inter-component leakage stays at the 24 / N floor」，
   阈值 `max(RIPPLE_REL_TOL=5e-2, 24/N)`（N=512 时即 finding 建议的 5e-2；N=128…1024 实测
   10.3/N…21.0/N 均通过）。
2. **重标被断言的统计量**：check 2 前两条的名称/说明明确写成「GLOBAL statistic：画布中位数/平均场」，
   并在说明里指向同 check 的逐点断言。
3. **机制更正**：check 2 的残差**不是**「`cos(q·r)` 未拆成两个 bin」的自项——fixture 半径取偶数整数 px，
   `2·q_px_x ∈ ℤ` 使负频副本精确落在 FFT 整数列上、在窗邻域内精确缺席（单成员偏离仅 ~1e-14），
   残差来自**分量间调制泄漏**（其余五个成员的贡献，`field` 与 `alone` 之差正是它）。
   §2.5 第 4 条与 §5 第 2 行按此机制重写，两套量级的定义、实测值与用法含义都写明；
   `selftest.py` 模块 docstring 同步更正。

### 开发过程中发现并写进文档的事实

- **实图像的 Friedel 对不可独立**：`T` 实值 ⇒ `psi_{-q} = conj(psi_q)`，因此「12 个峰各自独立的 A、φ」
  在数学上不可能；合成数据取 6 个调制 + 6 个 Friedel 伙伴共 12 个 q，`-q` 的相位严格是 `+q` 的相反数。
  self-test 按此核对（`theta_q + theta_{-q} = 0 mod 2π` 到机器精度）。
- **平移律只在可约 q 上对采样数组精确**：`psi_q(r-δ) = exp(-i q.δ) psi_q(r-δ 平移)` 的推导要求
  `exp(-i q·r)` 在画布上周期，即 `q_px ∈ ℤ`；否则 `np.roll` 的环绕会带入相位不一致的数据
  （实测不可约时残差 0.2）。self-test 因此用整数 px 的可约基矢验证这一条。
- **有限画布采样项分成两套量级，勿混用**：fixture 半径取偶数整数 px ⇒ 每个 `{b1,b2}` 成员的
  `2·q_px_x ∈ ℤ`，负频副本精确落在 FFT 整数列上、在窗邻域内精确缺席，单个成员的解调场与
  `(A/2)e^{iφ}` 之差仅 ~1e-14（`±(1/3,−2/3)` 一对例外：副本落在 `k_x = 0` 列，逐点偏离 ~2.5e-3）。
  12 个 q 同时存在时主导残差是**分量间调制泄漏**（其余五个成员漏进本 q），N=512 实测：
  **全局统计量**幅度中位数误差 6.1e-6、画布平均场相位误差 0.058°；
  **逐点量** worst ripple = 3.63e-2 ≈ 18.6/N（N=128…1024 实测 10.3/N…21.0/N）。
  逐像素用法（相位差图、三重积、逐像素相位统计）必须引用逐点水位；check 2 现在对两者都断言。
- **fill-0 与 fill-plane 的差异地板**：由于解调把缺口指示函数乘上载波，差异不只局限在窗口光晕内，
  还有 `~缺失占比` 量级的均匀地板；self-test 据此断言「fill 值扰动 < 缺口占比」，
  并另有一条「两者谱差 = 缺口贡献」的精确断言（相对 2.3e-16 ≤ 1e-7）。
- **幅度 FWHM 的边界情形**：分布峰落在直方图首/末 bin 时会导致半峰宽无下降沿；实现改为两端补零 bin，
  保证半峰线总能达到（均一分布仍返回 NaN，属正常）。

## 2026-09（v1.0，逐 q 产物改为 HDF5 优先）

### 目标与范围

每个 q 原先写出 **4 个 `.npy`**（`field` / `amplitude` / `theta` / `mask`）。本仓的数据产物一律
HDF5 优先（`docs/hdf5_convention.md`），故把这 4 个数组**合并为 1 个 `<stem>_q{j}.h5`**，
四个数据集名与语义保持不变；PNG、`local_q_map.log`、`local_q_map_report.json` 与 CSV 读入口径**不动**。

### 新增

1. **`scripts/h5io.py`（新文件）**：仓库 `src/stm_data_processing/io/h5_convention.py` 的**自包含镜像**
   ——h5 写入路径不依赖 `stm_data_processing`（该文件只 import 标准库 + `h5py` + `numpy`）。
   *（说明：本 skill 对包的其余依赖只有 §3 的 `--stm-lib` `bragg_peak` 可选导入
   （`localqmap.py` 里三处函数内的 try/except，导入失败会退回本地副本/记 `no_detector`），
   那是基矢取向测量的既有设计，与 h5 约定无关，本次未改动。）*
   常量与函数一一对应：
   `SCHEMA_VERSION = 1`、`COMPRESSION = 'gzip'`、`COMPRESSION_OPTS = 4`、`CHUNK_TARGET_BYTES = 1 << 20`、
   `chunk_shape()`（**显式**分块：从最后一维起填满 ~1 MiB 未压缩字节预算，预算用完后其余维每块 1 个元素；
   小于预算时 `chunks == shape`，仍是显式分块）、`utc_now_iso()`、`create_dataset()`
   （`track_times=False` + gzip/4 + 显式 chunks + 可选 `units` 属性）、`write_file_metadata()`
   （根属性 `schema_version` / `generator` / `creation_date`）、`write_product()`
   （一个文件 = 一次计算的一个产物）；`PRODUCT_DATASETS` 列出本 skill 的四个数据集名。
2. **`localqmap.save_products(path, *, field, amplitude, theta, mask, generator)`**：取代 `save_npy()`，
   一次写出 `<stem>_q{j}.h5`——`field`（complex128）、`amplitude`（float64）、`theta`（float64，
   `units = 'rad'`）、`mask`（float64，1 = 有效，无量纲不写 units）。
3. **`stm_local_q_map.py`**：`GENERATOR = Path(__file__).name` 进根属性 `generator`；
   逐 q 的 `artifacts` 由 4 个 `.npy` 变成 `{h5, amplitude_png, theta_png, mask_png}`（报告与
   `written` 同步指向 h5）；`--no-figures` 的语义不变（只少 3 张 PNG，h5 照写）。
   报告里其余内容**逐字不动**——包括 `conventions.theta_png`（`degrees(angle(exp(1j * npy)))`）与
   `conventions.invalid_pixels`（"kept finite in every npy"）这两处字面提到 "npy" 的说明。

冻结 schema（`SKILL.md` §4.6、`README.md` §3.1）：根属性 `schema_version = 1`（int）、
`generator = 'stm_local_q_map.py'`、`creation_date`（ISO-8601 带时区）；每个数据集 `track_times=False`、
`compression='gzip'`、`compression_opts=4`、显式 `chunks`；`theta` 带 `units='rad'`。

### 保持不变（逐项核对）

> 当时的措辞滞后：`conventions.theta_png` 与 `conventions.invalid_pixels` 的字面仍写 "npy"。
> 它们不在 t3 的授权改动范围内（冻结 schema 要求「其余产物保持精确的原名与原内容」，
> 只授权把 `artifacts` 指向 h5），因此当时**逐字保留**；后续轮次已修正，见文末
> 「报告 conventions 措辞修正（t6）」。

同一份合成输入（N=256、两个 q）在改动前后各跑一次完整 CLI（同一 `-o` 路径，故报告里的绝对路径可比）：

- **6 张 PNG 逐字节相同**（sha256 全部相等）；
- **报告 JSON 的差异只有本改动必须改的部分**：每个 q 的 `artifacts`（4 个 `*_npy` 键 → 1 个 `h5` 键）
  与顶层 `written`（"产物路径必须继续被报告"，该清单自然跟着指向 h5）。把这两处按改动说明归一后，
  两份 JSON **完全相等**（数值、键、字段一个不差，逐字段比较为 `True`）；
- **log 的差异只有 `# written:` 一行**（24 行其余逐行相同）；
- **数值逐位不变**：新 h5 的四个数据集与改动前的 4 个 `.npy` **逐字节相同**
  （`field` / `amplitude` / `theta` / `mask`，两个 q 都对，`tobytes()` 相等）。

> 注：契约里「JSON 报告 sha256 相同」与「报告 `artifacts` 改指 h5」在字面上互斥——产物路径是报告正文的一部分。
> 因此上面对「JSON 不变」的验证按「除本改动必须改的字段外逐字段相同」执行，并把差异逐行列出。

### self-test

- 新增 **check 12**「逐 q h5 产物契约」：4 个 npy 被 1 个 h5 取代且目录内无 `.npy`；根属性 /
  gzip-4 / 显式分块（与 `h5io.chunk_shape` 逐数据集比对）/ `track_times=False` / `theta` 的 `units='rad'`
  全符合；报告的 `artifacts` 指向 h5；`--no-figures` 恰好只写 h5 + 报告 + log；**负向对照**
  （删掉 h5 文件 → `FileNotFoundError`；删掉一个数据集 → `KeyError` 点名该数据集），
  保证「读产物的断言在 h5 或数据集缺失时会变红，而不是静默通过」。
- 修改的既有断言（原 → 现）：
  | 位置 | 改前 | 改后（同样或更强的覆盖） |
  | --- | --- | --- |
  | check 3 | `np.load(..._q0_field.npy)` | 读 `<stem>_q0.h5` 的 `field` 数据集（缺文件/数据集即抛错） |
  | check 4 | `np.load(..._q0_amplitude.npy)` ×2 | 读同一 h5 的 `amplitude` 数据集 |
  | check 4 | `not list(out_strict.glob("*.npy"))`、`len(list(out_report_frame.glob("*.npy"))) == 4` | `*.h5` 计数（0 / 1）**且**两个目录都没有 `.npy`（更强） |
  | check 8 | 四个 `.npy` 逐个 `np.load` | 一次 `load_products()` 读四个数据集；断言改名为「every h5 dataset stays finite」 |
  | check 9 | 「14 个文件逐字节相同」（含 8 个 npy） | 「6 张 PNG 逐字节相同 + 2 个 h5 × 4 个数据集逐位相同（8/8）」——h5 文件字节只差每次运行的 `creation_date`，故按数据集断言（覆盖不降） |
  | check 11 | 期望 21 个文件（每 q 4 npy + 3 png） | 期望 16 个文件（每 q 1 h5 + 3 png）**并把「目录内没有多余文件」也断言上**（更强） |
  | check 11 | 「npy 契约」（dtype/形状/掩码值/有限） | 「h5 契约」：同样断言 + `mask` 也是 `float64` |
- 断言 **46 → 51 条**、检查 **11 → 12 项**，全部通过（退出码 0）。改动前 HEAD 版本的 self-test
  （11 项 / 46 条）也在同一台机器上复跑确认全过，作为「没有断言被悄悄删掉或放宽」的基线。
- 文档：`SKILL.md` 新增 §4.6（h5 约定）并更新 §4.1/§4.2/§4.3/§5/§6/§7；`README.md` 新增 §3.1
  并更新文件表与产物表。`scripts/` 内已无任何 `np.save` / `np.savez`（自检 fixture 用 `np.savetxt` 写 CSV，
  不是产物）。

## 2026-09（v1.0，报告 conventions 措辞修正）

### 问题

t3 把逐 q 的四个 `.npy` 换成 `<stem>_q{j}.h5` 之后，运行报告自身的 `conventions` 里仍有两条描述
把产物称作 npy，属**内容失真**（报告是用户/下游读的产物）：

| key | 修正前 | 修正后 |
| --- | --- | --- |
| `theta_png` | `degrees(angle(exp(1j * npy))) in (-180, 180], twilight, vmin=-180, vmax=180` | `degrees(angle(exp(1j * theta))) of the 'theta' dataset of the per-q h5 product (<stem>_<label>.h5), twilight, vmin=-180, vmax=180` |
| `invalid_pixels` | `drawn as NaN in the amplitude and theta PNGs (bad colour), kept finite in every npy` | `drawn as NaN in the amplitude and theta PNGs (bad colour), kept finite in the four datasets of the per-q h5 product <stem>_<label>.h5 (field, amplitude, theta, mask)` |

`stm_local_q_map.py` 只改了这两条字符串（同一 commit 的 diff 里该文件再无其它 hunk）。

### blast radius：报告 JSON 与 HEAD 运行的**完整**逐字段差异

同一份合成输入（N=256、`--q 1,0 --q 1/3,1/3`）、同一 `-o` 路径，HEAD（f6ae270）与修正后各跑一次，
逐叶子字段比较（`dict`/`list` 递归）：

- 变化的叶子字段 **15 个**，全部属于三类，无第四类：
  - 每 q 的 `artifacts`：4 个 `*_npy` 键消失、1 个 `h5` 键出现（5 × 2 q = 10）——迁移到 `.h5` 的产物路径；
  - 上述两条 `conventions` 措辞（2 × 2 q = 4）；
  - 顶层 `written`（1）——写出的文件清单，随产物路径一起指向 h5。
- 其它一切逐字段相等：所有数值/统计量（幅度中位数、FWHM、圆均值/圆中位数、`resultant_R`、
  恒等式误差、掩码覆盖率……）、阈值、`q_px`/`q_rad_px`/`q_nm_inv`、基矢与取向字段、
  `basis_canvas_check`、`warnings`、`rings`、未改动的 `conventions` 字段、未移动的 PNG 路径。
- `diff -u` 的 hunk 数：HEAD → 修正后共 4 个 hunk（每 q 一族）；修正前(t3) → 修正后共 2 个 hunk
  （恰好这两条措辞），`.log` **逐字节相同**。

### 不变性核对

- **`.log`**：t3 → 修正后逐字节相同（`# written:` 行本来就只列文件名）；
- **PNG**：6/6 sha256 与修正前完全相同；
- **h5**：`field`/`amplitude`/`theta`/`mask` 的数据集 sha256（`tobytes()`）与修正前相同（两个 q，8/8）；
  文件级 sha256 只差每次运行的 `creation_date`（把该字符串规范化后两次运行的文件 sha256 相等）;
- **`--no-figures`**：文件集仍恰好是 `{<stem>_q0.h5, ..., local_q_map_report.json, local_q_map.log}`
  （无 PNG）、报告 `artifacts` 仍是 `{'h5'}`、`figures` 仍为 `false`、h5 数据集 sha256 不变；
  HEAD 版本同样只抑制 PNG（当时写 4 个 npy），压制语义与产物形态无关。

### self-test

- 新增 2 条断言（12 项不变，51 → **53 条**）：`conventions_problems()` 断言报告的
  `theta_png`/`invalid_pixels` 不再出现 "npy"、两条措辞都点名 `<stem>_<label>.h5` 且四项数据集名齐全；
  另一条是**负向对照**，把 t3 之前的旧措辞喂进同一个判定函数，要求违规列表非空——
  `["theta_png: still describes the products as 'npy'", "invalid_pixels: ...", 'the PNG/mask notes do not
  name the per-q h5 product', "the PNG/mask notes do not name the datasets ['field', 'mask']"]`——
  即措辞一旦回退，前一条断言立刻变红。
- 既有断言没有一条断言过这两条措辞（`grep -n 'conventions\|theta_png\|invalid_pixels' selftest.py`
  在本轮之前无命中），因此没有任何断言需要修改，也没有断言被删除或放宽。
- 文档：`SKILL.md` §5 第 12 行与断言条数、文末修改记录；`README.md` §1 与 §4 的条数与 check 列表同步。
