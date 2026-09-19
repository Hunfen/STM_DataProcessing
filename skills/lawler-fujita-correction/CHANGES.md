# CHANGES — lawler-fujita-correction

## 2026-09（v1.0 修复）：LF 产物文件名对齐文档 + 方向数措辞

验证者 findings F1/F2 的修复（不改算法、不改数值结果）：

- **F1（产物文件名）**：SKILL.md §3 与 README.md 一直承诺 `<stem>_lf_theta_a.npy`、`<stem>_lf_u_x.npy`、
  `<stem>_lf_mask.npy` 等带 `_lf_` 标记的名字，而脚本原先写的是不带标记的 `<stem>_theta_a.npy`……
  **决定：让脚本对齐文档**——`write_artifacts()` 现在写 `<stem>_lf_<map>.{npy,png}`（9 张图共 18 个
  文件），`correction_report.json` 的 `lf_artifacts` 路径同步指向这些字面名字；核心产物
  （`<stem>_corrected.csv`、`_corrected_fft2.npy`、`_corrected.png`、`_corrected_fft.png`、
  `correction.log`、`correction_report.json`）与包名（`FILE` + 同名 `.npz`）不变。
  自检**新增第 17 项**：在拟合输出目录里按**字面文件名**核对 18 个文档名是否存在
  （不再只信报告 JSON 里的路径），因此文档与代码一旦再次漂移，自检会失败。
- **F2（方向数措辞）**：原来无论几个方向都打印「the three directions are exactly N deg apart」
  （且该 f-string 里有一处字面量残留）。现在 log 分两句报告实际几何：
  `# reference hexagon: ... orientation <φ> deg (measured members deviate <d> deg from the exact
  <pair_angle> deg spacing of this direction set)` 与
  `# lock-in directions: <n> (<three|two>) reference wave vector(s), pairwise spacing exactly
  <pair_angle> deg (<Q_a + Q_b + Q_c = 0 | Q_a + Q_b does not close: no third direction>)`。
  默认 120° → 3 个方向、`Q_a + Q_b + Q_c = 0`；`--pair-angle 60` → 2 个方向、间距 60°、明说
  `Q_a + Q_b` 不闭合，第三方向诊断 `available = false`，且**不写** `_lf_theta_c` /
  `_lf_amplitude_c`（该模式 14 个 `_lf_` 文件）。报告新增 `n_lockin_directions` 与
  `direction_spacing_deg` 两个字段。
- 文档同步：SKILL.md §3 产物表（`_lf_` 标记 + 三方向 18 文件 / 两方向 14 文件的区别）、§2.3、
  §5（16 → 17 项，新增第 6b 行）；README.md 文件表与输出段。
- 验收：自检 **17/17** 通过（原 16 项全部不变，新增的字面文件名断言 18/18 通过）；
  `--pair-angle 60` 实跑 exit 0、措辞与产物数如上述；`selftest_lf/out/` 中 `_lf_` 文件 18 个、
  非预期文件名 0 个。

## 2026-09（v1.0）：新建 skill —— Lawler–Fujita 位移场矫正 + 可迁移包

用户要求：实现 Fujita et al., PNAS 2014（SI Text §4）的 Lawler–Fujita 晶格相位矫正——
输入拓扑图 → 矫正数据 + 复数 FFT2 + **可迁移**的矫正包；结构与约俗对齐已有的几何矫正 skill
（`affine-correction`，由 `topo-correction` 改名而来）。

### 算法（`scripts/stm_lf_correct.py` + `scripts/lf_lib.py`）

- 模型与符号（脚本内固定、由自检端到端验证）：
  `T(r) = ideal(r − u(r))`；`psi_i = lowpass[T·exp(−i Q_i·r)]`；`theta_i = arg psi_i = −Q_i·u`；
  `u = Q⁻¹(θ̄ − θ)`，`θ̄ = 0`（有效像素上相位均值为 0）；`corrected(r) = T(r + u(r))`。
- lock-in：**q 空间高斯 `exp(−λ²|k|²/2)`**，等价于论文中半径 `δq = 1/λ` 的圆盘；λ 由
  `--lambda-nm`（默认 30.0 nm）给出。掩码在未位移的 FFT 索引网格上用环绕距离计算。
- **可解性条件**（写进文档）：`|∇u| < 2π/(|K|λ)`；λ = 30 nm ↔ 0.7 % 应变。λ ≲ 1 nm 时掩码半径
  接近邻近 Bragg 峰间距，邻近环泄漏（实测第三个方向出现 100° 量级相位错误），故不可用。
- **相位解缠用最小二乘（Poisson + DCT，Neumann 边界）**：复数 lock-in 场存在零点，逐行逐列顺序
  解缠会把一个 2π 位错沿整行传出（实测该方向相位误差 130°）；最小二乘把影响限制在局部
  （同一数据 0.18°）。解缠前用与掩码同一阈值把低幅度像素的折叠相位填成最近可靠像素的值。
- **带宽自检**：参考波矢与测得峰位之间必有偏移（数据的平均应变），而掩码半径是 `1/λ`；当
  `λ · |q_measured − q_ref|/L > 1` 时掩码里没有真正的 Bragg 峰，解出的场不是晶格畸变。脚本打印
  该诊断、报 WARNING，并把它记进报告（`lockin.band_warning`）。
- **参考六方**：半径取**理想值** `2L/(√3 a)`（由 `--a` 决定），取向取数据——三个成员相对「恰好
  120° 间隔」的平均偏差加到最强成员方位角上；三个方向严格等间隔，于是 `Q_a + Q_b + Q_c = 0`
  精确成立，第三方向自检检验 lock-in 而非参考几何。测得峰位（被平均畸变平移过）另存
  `q_measured_px` 供核对。
- **规范自由度写进文档**：位移场在「刚性平移 + 参考六方的均匀应变」下不可分辨（数据只给取向，
  平均剪切会偏一点）；自检同时报原始误差与去仿射规范后的误差（实测 0.054 nm / 0.0031 nm）。
- 重采样：`scipy.ndimage.map_coordinates`（`order=3`、`cval=NaN`），画布
  `n_out = n + 2(⌈max|u|⌉ + pad)`，nm/px 不变，矫正后视场 `L·n_out/n`；输入 NaN 先填平面均值再插值，
  缺测掩码用一阶插值传递（避免 NaN 污染三次样条前置滤波）。有效掩码之外的像素不施加位移。
- 回退：找不到可用 1x1 环（半径不符/成员 < 6）或有效掩码 < `--min-coverage`（默认 0.50）时
  `method = "identity_fallback"`、`fallback = true`，**不做任何矫正**，输入原样输出 + log WARNING。

### 产物与契约

- 输出与 affine skill 同类：`<stem>_corrected.csv`、`<stem>_corrected_fft2.npy`（complex128、
  Hanning 窗 + NaN 平面填充）、`<stem>_corrected.png`（gwyddion）、`<stem>_corrected_fft.png`
  （inferno 对数）、`correction.log`、`correction_report.json`。
- log 含两条契约行：`# canvas ... field of view <L> nm (...)` 与
  `# corrected canvas: <n_out> x <n_out> px, field of view <X> nm (...)`。
- **LF 附加产物**（各 npy + 预览 png）：`theta_a/b/c`（解缠相位，弧度；png 折叠到 (−180,180] 度）、
  `amplitude_a/b/c`、`u_x`/`u_y`（nm）、`mask`。
- 报告含：Q_a/Q_b/Q_c（px 与 nm⁻¹）、测得峰位、参考六方取向与成员偏差、λ、幅度阈值、掩码覆盖率、
  `gauge`（θ̄ = 0）、u 统计（rms/max nm）、第三方向一致性（全掩码 + 边界带内）、
  矫正前后重检测的 1x1 环各向异性与 1x1/r3 环对比值、画布/视场/NaN 比例、`method/fallback`。
- 边界：解调相位在边界约 `0.65λ` 带内被周期化 FFT 污染（实测污染剖面），**不作硬性剔除**（大 λ 时
  会判死整幅图），而在第三方向诊断里剔除该带并同时报告两个数字（0.18° vs 7.41°）。

### 可迁移包

- `--save-transform FILE`：`FILE` 为变换 JSON（schema `lawler-fujita-correction-transform`、
  `schema_version = 1`：Q 矩阵/λ/a/参考网格与视场/pad/order/幅度阈值/method/fallback/路径/迁移约束），
  同名 `.npz` 存 `u_x`、`u_y`（nm，参考网格）与 `valid` 掩码。
- `scripts/stm_lf_apply.py`：目标 CSV + `--transform bundle.json` + `-L`（目标自己的视场）→
  **只做重采样**（不重新检测、不重跑自检）；同网格时逐位使用位移场（因此与拟合段逐位一致，
  实测 max|Δ| = 0）；不同像素数按物理位移（nm，各自 nm/px）插值到目标像素坐标并重新计算画布；
  掩码随场传递；视场不匹配（相对差 > `--fov-tol`，默认 1e-3）打 WARNING 并记 `fov_warning`；
  `identity_fallback` 包把目标原样复制 + WARNING。产物同拟合段 + `apply_report.json`。
- 迁移语义依据论文 SI：同一扫描的形貌图与谱学图共享同一畸变，把形貌图求得的位移场套用到谱学数据
  是论文自身的做法；因此约束是「同一扫描/同一视场」，而非「同一像素数」。

### self-test（`scripts/selftest.py`，16 项，约 70 s）

合成真值：a = 0.246 nm、L = 50 nm、n = 1024 的六方点阵（1x1 三成员 + r3 三成员），注入已知光滑
位移场（1.0 nm 高斯包 σ = 30 nm + 0.4 nm 线性漂移，零均值）。注入幅度受**可见性条件**约束：
Bragg 峰宽 ≈ `|∇u| × 半径`，环聚类容差是半径的 2–3 %，故 `|∇u| ≲ 2 %`（实测 0.025 仍可检出、
0.03 起环碎裂），在 50 nm 视场内相当于 ≲1 nm。检查：注入场可见性、拟合跑通、报告字段、两条契约行、
9 张产物、恢复误差（原始 0.054 nm 与去规范 0.0031 nm，阈值 `max(10 % 注入 rms, 0.1 nm)`）、
非平凡性、矫正后 1x1 环各向异性（1.72 % → 0.012 %）、1x1/r3 环对比值（−0.0009 %）、
第三方向一致性（边界带内 0.18°，阈值 2°）、包 JSON + npz、同网格 apply 逐位复现、
不同像素数（800）apply 重采样、无晶格回退。另有 informational：默认 λ = 30 nm 只保留 63.7 % 的
位移（默认低通对应远小于 1 % 应变的畸变）。

### 已知限制

相位结点像素相位不可靠（掩码剔除 + 解缠填充）；边界 0.65λ 带内相位被污染；位移场含平动 + 均匀
应变自由度；应变 ≳ 2–3 % 时检测找不到成形环 → 回退；apply 段不重跑检测与自检，可信度等于来源图。
