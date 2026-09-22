# CHANGES — lawler-fujita-correction

## 2026-09（修正）：bundle h5 的 `nm_per_px` units 由 `'nm'` 改为 `'nm/px'`

`t2` 的独立验证（`var/verify_skills_h5/report.md` §14.2.1）指出：`--save-transform` 的 bundle
h5 里 `nm_per_px` 被写成 `units = "nm"`，而它是**像素尺度**（nm per pixel，两个长度的比值），
属性与数据集名字互相矛盾——信任 `attrs["units"]` 的消费者会按错比例换算。迁移前的 `.npz`
没有任何 units 属性，这个属性是本次 h5 化新引入的，因此由本次修正负责。

- `stm_lf_correct.py`：bundle 的 `nm_per_px` 数据集写 `units = "nm/px"`；同一处的注释把六个成员的
  units 一一讲清（见下）。**只改这一个字符串**：数据集名、dtype（`u_x`/`u_y`/`field_of_view_nm_reference`/
  `nm_per_px` float64、`valid` bool、`n_px_reference` int64）、值、chunking 与其余属性都没动。
- **同一次通检本 skill 写出的全部 units 属性**（结论：除 `nm_per_px` 外都正确，未改动）：

  | 数据集 | 量 | 属性 | 依据 |
  | --- | --- | --- | --- |
  | `<bundle>.h5: u_x` / `u_y` | 位移分量 | `"nm"` | 物理长度；数据集名与报告 `displacement.component_order` 都写明 nm |
  | `<bundle>.h5: field_of_view_nm_reference` | 参考图视场（长度） | `"nm"` | 物理长度 |
  | `<bundle>.h5: nm_per_px` | 像素尺度 | `"nm/px"`（**本次修正**，原为 `"nm"`） | 比值量；名字已写明，属性必须与名字一致 |
  | `<bundle>.h5: n_px_reference` | 像素数（计数） | **无** | 计数不是物理量，按仓库约定（无量纲量不写 `units`）不写属性；修正前已如此，保持 |
  | `<bundle>.h5: valid` | 有效掩码 0/1 | **无** | 无量纲 |
  | `<stem>_lf.h5: theta_a/b/c` | 局域相位 | `"rad"` | 解缠相位，弧度 |
  | `<stem>_lf.h5: u_x` / `u_y` | 位移分量 | `"nm"` | 物理长度 |
  | `<stem>_lf.h5: amplitude_a/b/c` | lock-in 幅度 | **无** | 与输入拓扑同单位（任意单位，不是物理单位），不写属性 |
  | `<stem>_lf.h5: mask` | 有效掩码 0/1 | **无** | 无量纲 |
  | `<stem>_corrected.h5: corrected` | 矫正后拓扑 | **无** | 输入拓扑是任意单位（CSV 无单位），不写属性 |
  | `<stem>_corrected.h5: fft2` | 复数 FFT2 | **无** | 原始 FFT，任意单位，不写属性 |

- **selftest 的 gate**：`bundle_units`（`check_h5_schema` 的期望表）把 `nm_per_px` 钉成 `"nm/px"`
  （`valid`、`n_px_reference` 钉成 `None` = 必须**没有** units 属性）；把脚本里的字符串改回
  `"nm"`，`bundle JSON + u-field h5 written` 立即变红
  （`nm_per_px: units 'nm' != 'nm/px'`），self-test 退出码非零。断言数量不变（**20 项**），
  没有删除或削弱任何断言；PASS 行现在也照出每个数据集的 units，便于人工核对这条元数据。
- 验收：同一输入、同一输出目录下，修正前后两版脚本的全部 CSV / PNG / JSON / `.log` 产物
  **sha256 逐字节相同**；三个 h5 产物的数据集名、dtype、值（逐字节）、chunking、compression 与
  其它属性全部相同，差异**只有** `<bundle>.h5` 的 `nm_per_px` 的 `units`（`'nm'` → `'nm/px'`）；
  `creation_date` 每次运行本就不同（已文档化），不构成差异。详见修正证据。

## 2026-09：数组侧产物改为 HDF5（`<stem>_corrected.h5` / `<stem>_lf.h5` / `<bundle>.h5`）

目标：仓库的数据产物 **HDF5 优先**——LF 的相位/幅度/位移/掩码此前是 9 个 per-map `.npy`，
可迁移包的 u 场此前是 `.npz`，现在分别是**一个** `<stem>_lf.h5` 与 `<bundle>.h5`；
矫正数组另外写 `<stem>_corrected.h5`。inter-skill 的 CSV 契约、图、报告与 log 一个字节不动。

- **新增自包含模块 `scripts/h5io.py`**（镜像 `src/stm_data_processing/io/h5_convention.py` 的约定，
  skill **不** import `stm_data_processing`）：`SCHEMA_VERSION = 1`、`COMPRESSION = "gzip"`、
  `COMPRESSION_OPTS = 4`、`CHUNK_TARGET_BYTES = 1 << 20`、`chunk_shape()`（从末维起填满 ≤ 1 MiB）、
  `create_dataset()`（`track_times=False` + gzip/4 + 显式 chunks + 可选 `units`）、
  `write_file_metadata()`（root `schema_version`/`generator`/`creation_date`）、`write_file()`。
- **`<stem>_corrected.h5`**（`stm_lf_correct.py` 与 `stm_lf_apply.py` 都写）：数据集 `corrected`
  （`float64`，与 CSV 同一个数组）与 `fft2`（`complex128`，与保留的 `.npy` 同一个数组）；
  root `generator` = 产生脚本名，附 `input`（apply 段另加 `transform_file`）、`field_of_view_nm`、
  `nm_per_px`。两个数组无物理单位 → 不写 `units`。
- **`<stem>_corrected_fft2.npy` 保留**：`skills/phase-analysis/` 通过 `--fft2` 读它。
- **9 个 per-map `.npy` → 一个 `<stem>_lf.h5`**：数据集名与原来一一对应（`theta_a/b[/c]`、
  `amplitude_a/b[/c]`、`u_x`、`u_y`、`mask`），文件名保留文档化的 `_lf_` 标记；
  `units`：相位 `"rad"`、`u_x`/`u_y` `"nm"`、幅度与掩码无量纲（不写）。每张图的
  `<stem>_lf_<name>.png` 预览**照旧**。`correction_report.json` 的 `lf_artifacts` 登记项随之从
  `{"npy", "png"}` 变为 `{"h5", "dataset", "png"}`（这是本次唯一改动的报告字段，其余字段与数值逐字节不变）。
- **`--save-transform` 的 u 场 `.npz` → `.h5`**：同样 6 个成员（`u_x`、`u_y`（`nm`）、`valid`
  （bool，无量纲）、`n_px_reference`、`field_of_view_nm_reference`（`nm`）、`nm_per_px`（`nm`）；
  HDF5 的标量不能分块/过滤，因此后三个存成**单元素数据集**（名字与 npz 时代一致）。
  bundle JSON 的 `u_field_file` 指向 `.h5`（这是 JSON 里唯一改动的字段），schema 与其余字段不变。
- **`stm_lf_apply.py`**：`load_bundle()` 由 `np.load(u_path)` 改为用 `h5py` 读该 `.h5`
  （缺文件/缺数据集/不是 HDF5 都转成带原因的 `ValueError` → 退出码 2），并写出自己的
  `<stem>_corrected.h5`；`apply_report.json`、`correction.log` 与 `<stem>_corrected_fft2.npy` 不变。
- **未改动的产物**：`<stem>_corrected.csv`、全部 png、`correction.log` 的其余所有行（含两条
  跨 skill 契约行与 `# written:` 行——它只列仍在写出的产物，无过时描述）、
  `correction_report.json` 除 `lf_artifacts` 外的全部内容。
- `lf_lib.save_npy()` 随 per-map npy 一起消失（本次改动产生的孤儿函数，已删除）。
- **self-test 17 → 20 项**（原有 17 项的意图与阈值不变）：
  第 5 项（原「9 张 npy + png 齐全」）改为「一个 `<stem>_lf.h5` 的 9 个数据集 + 每张 png +
  报告登记项 `h5`/`dataset`/`png`」，并核 h5 约定（root 三属性、gzip/4、约定 chunk、
  `track_times=False`、`units`）；第 6b 项（原「18 个文档名字面存在」）改为「`<stem>_lf.h5`
  + 9 个 `<stem>_lf_<map>.png` 存在，且**不留任何 per-map npy**」；第 13 项（原「包 JSON + u 场
  npz」）改为「包 h5 约定 + `u_field_file` 指向 `.h5` + 无 `.npz` 遗留」；新增 6c（`<stem>_corrected.h5`
  约定 + `fft2` 与 `.npy`、`corrected` 与 CSV 逐位一致）、13b（`<stem>_lf.h5` 与包 h5 的 `u_x`/`u_y`/`mask`
  逐字节相同）、14b（apply 段 h5 约定且与拟合段逐字节相同）。等值判定用 `tobytes()` 字节比较而非
  `np.array_equal`（后者会把只差 NaN 的数组判为不等）。
- `track_times=False` 的验证方式：本环境 h5py 3.16 的 `get_obj_track_times()` 对读回的数据集一律
  返回 True，因此 self-test 改看对象头的时间消息（`h5py.h5o.get_info(...).ctime/mtime` 恒为 0）。
- **log 措辞同步（唯一的 log 改动）**：`correction.log` 里原本的
  `# LF artifacts: <name> (npy+png), ...` 描述的是已被替换掉的 per-map npy，属于过时的产物
  描述，因此改为 `# LF artifacts: <name> (h5+png), ... in <stem>_lf.h5`；同一行里出现的
  map 名字与顺序一字未动，其余全部 log 行（含两条跨 skill 契约行）逐字节不变。
  `# written:` 行只列仍在写出的产物（csv / fft2 npy / png / report / log），因此没有过时描述、
  保持原样；新 h5 产物写在 `SKILL.md` / `README.md` 与本文件里。
- 验收：self-test **20/20** 通过；同一输入、同一输出目录下与迁移前脚本逐文件比对：CSV / png
  全部 sha256 相同，`.npy` FFT2 字节相同，per-map `.npy` 与 `.npz` 的每个成员都**逐字节**
  等于新 h5 里同名数据集；差异集恰好是：新增 `<stem>_corrected.h5` / `<stem>_lf.h5` /
  `<bundle>.h5`，消失 9 个 per-map `.npy` 与 `<bundle>.npz`，JSON 只有 `lf_artifacts` 登记项
  （`npy` → `h5` + `dataset`）与 `u_field_file`（`.npz` → `.h5`）两处是「描述已移动产物的字段」，
  log 只有上面那一行措辞。详见 `SKILL.md` §3、§4、§5 与 迁移报告。

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
