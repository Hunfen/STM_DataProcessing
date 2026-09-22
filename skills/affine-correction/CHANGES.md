# CHANGES — affine-correction

## 2026-09：数组侧产物改为 HDF5（`<stem>_corrected.h5`），CSV/PNG/JSON/log 逐字节不变

目标：仓库的数据产物 **HDF5 优先**——本 skill 的矫正数组此前只以 CSV（`%.10e`，有精度损失）和
`<stem>_corrected_fft2.npy` 为容器，现在两个数组同时写进一个 h5；inter-skill 的 CSV 契约、
图、报告与 log 一个字节都不动。

- **新增自包含模块 `scripts/h5io.py`**（镜像 `src/stm_data_processing/io/h5_convention.py` 的约定，
  skill **不** import `stm_data_processing`）：`SCHEMA_VERSION = 1`、`COMPRESSION = "gzip"`、
  `COMPRESSION_OPTS = 4`、`CHUNK_TARGET_BYTES = 1 << 20`、`chunk_shape()`（从末维起填满 ≤ 1 MiB，
  预算用完后前导维取 1）、`create_dataset()`（`track_times=False` + gzip/4 + 显式 chunks + 可选
  `units`）、`write_file_metadata()`（root `schema_version`/`generator`/`creation_date`）、
  `write_file()`（一次写多数据集）。
- **`stm_topo_correct.py` / `stm_apply_transform.py`**：在写 CSV 与 `.npy` 之后额外写
  `<stem>_corrected.h5`，数据集 `corrected`（`float64`，**与写进 CSV 的是同一个数组**）与
  `fft2`（`complex128`，**与 `.npy` 是同一个数组**）；root `generator` 分别是
  `stm_topo_correct.py` / `stm_apply_transform.py`，并附 `input`（apply 段另加 `transform_file`）、
  `field_of_view_nm`、`nm_per_px` 溯源属性。两个数组都是输入拓扑的任意单位（无物理单位），
  按仓库约定**不写** `units`。
- **`<stem>_corrected_fft2.npy` 保留**：`skills/phase-analysis/scripts/stm_phase_analysis.py`
  通过 `--fft2` 读它，phase-analysis 明确不在本次迁移范围。
- **未改动的产物**：`<stem>_corrected.csv`、两张 png、`correction.log`、`correction_report.json`、
  `apply_report.json`、`--save-transform` 的 transform JSON（含 stdout 的 `# written:` 行）
  名称与字节全部不变；h5 不进报告/log 的登记列表（那些列表里没有过时的条目要修）。
- **self-test 6 → 8 项**（原 6 项的意图、阈值与子步骤一个不少）：新增第 7 项（矫正段的
  `<stem>_corrected.h5`：root 三属性 + `generator` + gzip/4 + 约定 chunk + `track_times`，
  且 `fft2` 与 `.npy`、`corrected` 与 CSV 逐位一致）与第 8 项（两段式两个 stage 的 h5：拟合段同上，
  同网格 apply 的 `corrected`/`fft2` 与拟合段**逐字节**相同，换尺寸与 `identity_fallback` 段的 h5
  也合规）。等值判定用 `tobytes()` 字节比较而非 `np.array_equal`：后者会把「只差 NaN」的两个数组
  判为不等（apply 段画布边缘正是 NaN）。
- `track_times=False` 的验证方式：本环境 h5py 3.16 的 `get_obj_track_times()` 对从文件读回的
  数据集一律返回 True（不反映创建标志），因此 self-test 改看**对象头的时间消息**——
  `track_times=False` 时对象头不存时间，`h5py.h5o.get_info(...).ctime/mtime` 恒为 0。
- 验收：self-test **8/8** 通过；同一输入、同一输出目录下与迁移前脚本逐文件比对：CSV / png /
  log / JSON 全部 sha256 相同，`.npy` FFT2 字节相同，新增文件恰好只有 `<stem>_corrected.h5`。
  详见 `SKILL.md` §3、§4 与 迁移报告。

## 2026-09：skill 改名 topo-correction → affine-correction（零行为变化）

用户要求把几何矫正 skill 由 `topo-correction` 改名为 `affine-correction`（目录名 + frontmatter
`name:` + 文档/用法示例里的名字与路径），并明确要求**运行时行为逐字节不变**。

- 目录改名：`skills/topo-correction/` → `skills/affine-correction/`（运行时副本
  `~/.agents/skills/topo-correction` → `~/.agents/skills/affine-correction`）。
- `SKILL.md`：frontmatter `name:` 与标题改名；§2 / §3 / §4 的用法示例脚本路径
  `skills/topo-correction/scripts/...` → `skills/affine-correction/scripts/...`；§3 增补一句说明
  transform JSON 的 schema id 是冻结标识。
- `README.md`：标题与全部用法示例脚本路径同步改名。
- **冻结标识与运行时字符串一律不变**（改名只动名字，不动产物与输出）：
  - transform JSON 的 `schema` 值 `topo-correction-transform`（产物标识，apply 段据此校验）；
  - `correction_report.json` / `apply_report.json` 的 `skill` 字段值 `"topo-correction"`；
  - `stdout` 与 `correction.log` 的既有行，含跨 skill 契约行
    `# corrected canvas: <n> x <n> px, field of view <X> nm (...)`；
  - CLI `--help` 文本与 transform JSON 的 `usage` 字段值（二者都是程序输出）。
- 脚本改动仅限模块 docstring 里的 skill 名（`correction_lib.py` / `stm_apply_transform.py` /
  `selftest.py` 各 1 行）：**零逻辑改动**——无默认值、无字段、无文件格式、无 log 行变化。
- 验收：改名前后 `selftest.py` 均 6/6 全绿（原始副本作 baseline），同一输入的全部产物逐字节一致；
  两棵 skill 树 `diff -r` 只出现 docstring 行与 SKILL.md/README.md/CHANGES.md 的命名行。
- 其他 skill（`phase-analysis`）与仓库文档里的活跃交叉引用另出落盘补丁（见 captain 的 landing 流程）。

## 2026-09（v2.2）：两段式变换（导出变换 JSON + 套用到另一张图）

用户要求：拟合出的矫正变换要能**脱离原图**单独保存，再应用到**指定的另一张**拓扑图（可不同尺寸、
不同视场），且**向后兼容**——不加新开关时所有既有输出逐字节不变。

- `stm_topo_correct.py` 新增 `--save-transform FILE`（缺省关闭）：拟合结束后额外写一份独立变换 JSON。
  schema `topo-correction-transform`、`schema_version = 1`，字段：`source_report`（参考
  `correction_report.json` 绝对路径）、`source_input`、`affine_q`（对称正定拉伸 M）、
  `affine_image_reference`、`n_px_reference`、`n_out_reference`、`field_of_view_nm_reference`、
  `pad`、`order`、`method`、`fallback`、`n_labelled`、`anchor_ring`、`a_nm`、`a_ref_nm`、
  `stretch_scale_sqrt_det`、`anchor_verdict`、`usage`（命令行提示字符串）。
  该开关关闭时 `correction.log` / `correction_report.json` / 矫正 csv·npy·png **逐字节不变**：
  实测同一输入、同一 `-o` 下新旧脚本 6 个产物 `cmp` 全部相同、stdout 相同；`SKILL_VERSION` 未变，
  报告与 log 字段未增删改。
- 新脚本 `scripts/stm_apply_transform.py`（CLI：`INPUT.csv --transform TRANSFORM.json -L SIZE_NM
  [-o OUTDIR] [--delimiter X] [--stm-lib DIR]`）：与拟合段相同的读入/分隔符自动识别与预处理
  （`flipud(subtractMeanPlane(...))`）；先校验变换 JSON（schema 字符串、`schema_version`、
  `affine_q` 有限 2×2 且 `det > 0`，不满足则打印明确原因并以非零码退出）；再用**目标图自己的像素数
  n** 重算数组矩阵 `A = P·M⁻¹·P`、画布 `n_out = 2·(⌈max|A⁻¹·角点|⌉ + pad) + 1` 与 offset
  （本地镜像 `bragg_peak.correct._image_transform`，`pad`/`order` 取自 JSON），
  `scipy.ndimage.affine_transform(order=order, mode='constant', cval=NaN, prefilter=order>1)` 重采样；
  输出与拟合段同名的 `<stem>_corrected.csv`（`%.10e` 逗号）、`_corrected_fft2.npy`（`complex128`，
  `compute_fft2(..., subtract_plane=False)`）、`_corrected.png`（gwyddion）、`_corrected_fft.png`
  （inferno 对数）、`correction.log`（含跨 skill 契约行 + 变换来源 / method / fallback /
  anchor_verdict）与 `apply_report.json`（`skill="topo-correction"`、`stage="apply"`、输入与变换路径、来源报告、method/fallback/
  anchor_verdict、视场、`n_px`、`affine_q`、`matrix_used`、`offset`、`n_out`、矫正后视场与 nm/px、
  NaN 比例、`written` 六个产物）。矫正后视场 `= L · n_out / n`。**不重新检测峰、不重跑锚定自检**
  （`anchor_verdict` 是参考图的结论）；`method == "identity_fallback"` 时按包语义原样复制输入
  （`n_out = n`、视场不变、不加 NaN）并在 log 里打 WARNING。
- self-test 5 → 6 项（原 5 项意图与阈值不变）。第 6 项为单条检查、内部子步骤：
  (a) `--save-transform` 导出的 JSON 20 个字段齐全、`affine_q` 与 `correction_report.json` 逐位一致；
  (b) 同一参考图 apply 的 `_corrected.csv` 与拟合段一致（`np.allclose(rtol=1e-10, atol=1e-10,
  equal_nan=True)`——NaN 填充两侧相同，实测逐字节相同）；
  (c) 换尺寸目标（生成器 160 → 画布 183 px）的 `n_out` 按目标 n 独立重算（205，与参考图 435 不同）、
  log 契约行视场 = `L·n_out/n`、`apply_report.json` 写出；
  (d) 本地 `_image_transform` 镜像与包内函数在矩阵 / `n_out` / offset 上逐位一致；
  (e) `identity_fallback` 变换 → 目标原样复制且 log 含 WARNING。
- 文档：`SKILL.md`（§2 用法、§3 变换 JSON 契约与 apply 语义、§4 自检 6 项）与 `README.md`
  （文件表、两段式用法块、自检项数）同步更新。**仅改仓库副本**，运行时副本
  `~/.agents/skills/topo-correction` 由 captain 后续同步。

## 2026-09：从 stm-topo-phase-analysis 拆分为独立 skill

用户要求把几何矫正拆成独立 skill（矫正产物可接任意后处理），与相位分析并列：

- 新目录 `skills/topo-correction/`；原目录 `skills/stm-topo-phase-analysis` 改名
  `skills/phase-analysis`（相位部分，见其 CHANGES.md）。
- `stm_topo_correct.py` 原样移入（除 import 改造：`atlas`/`phasepipe` → 本地
  `correction_lib`）；输出文件名、`correction.log` 行格式、`correction_report.json` 字段
  **逐字节不变**（跨 skill 契约）。`correction_report.json` 的 `skill` 字段值由
  `stm-topo-phase-analysis` 更正为 `topo-correction`。
- 兄弟依赖内聚为 `scripts/correction_lib.py`（自包含拷贝：`group_rings`、`setup_style`、
  `load_colormap`/`_GWYDDION_ANCHORS`、`BAD_COLOR`），不 import phase-analysis 的任何模块。
- 自带一键 self-test（`scripts/selftest.py`，5 项），从原 self-test 的矫正阶段移植
  （合成数据生成器为独立实现，不依赖 phasepipe）。
- 相位 skill 的 self-test 同步移除矫正阶段检查（72 → 68 项）。

## 2026-09（v2.1）：锚定自检新增全局拉伸尺度 tell-tale

原自检只用"矫正后两个最强环的半径比是否 1 : √3"；当原始数据本身就有精确 1 : √3 环对时，
错锚定按全局因子（1/√3 或 √3）缩放整个拟合、比值仍然通过 ⇒ 误报 consistent。
新增 `|det M|^(1/2)` 拉伸尺度判据（容差 5 %），两条 tell-tale 合并判定：
都过 = consistent，任一失败 = inconsistent（比值通过、尺度失败也是 inconsistent），
环对无法形成但尺度在容差内 = unverifiable。实测错锚偏 42 % / 73 %，正确锚定偏 ≤2 %。

## 2026-09（v2）：锚定环显式化（`--anchor-ring`）

包内数据锚定检测把它认定的第一个环标成 (1,0)；r3 环在 1x1 环内侧（半径比 1/√3）时检测可能
锚定在 r3 环上，按 1x1 给参考晶格会把 r3 环拉到 1x1 理想半径、破坏几何。
`--anchor-ring 1x1|r3` 显式声明锚定环，参考晶格 `a_ref = a` 或 `√3·a`。

## 2026-09（v1，原 stm-topo-phase-analysis）

初版矫正流程（topo0009 实测）；v1 入库时改为调用 `stm_data_processing.utils.bragg_peak` 包。
