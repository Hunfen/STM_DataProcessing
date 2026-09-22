# lawler-fujita-correction（Lawler–Fujita 晶格相位矫正）

仓库内置 skill（**v1.0**）。**只做几何矫正数学**：局域 lock-in 相位提取、逐点位移场求解、按位移场
重采样、矫正后自检与可迁移包导出。物理解释由使用者进行——脚本、图、log 与文档都不给物理结论。
矫正产物可接**任意后处理**（例如 `skills/phase-analysis/` 的双环相位分析）。

与 `skills/affine-correction/` 的分工：后者是**全局仿射拉伸**（一个 2×2 对称正定矩阵），
本 skill 是**逐点位移场 u(r)**（每像素两个自由度），对应 Fujita et al., PNAS 2014 SI §4。

## 文件

| 路径 | 作用 |
| --- | --- |
| `scripts/stm_lf_correct.py` | 拟合段 CLI：检测 1x1 环 → 三个等间隔参考波矢（理想半径 + 数据取向）→ q 空间高斯 lock-in → 最小二乘相位解缠 → `u = Q⁻¹(θ̄ − θ)` → 按 u 重采样；输出矫正产物（CSV + h5）+ 相位/幅度/位移/掩码单一 `<stem>_lf.h5` + 报告；`--save-transform` 导出可迁移包（JSON + u 场 h5） |
| `scripts/stm_lf_apply.py` | **第二段**：把包里的稠密位移场（从 bundle 的 `.h5` 读）套用到同一扫描的另一张图（同网格逐位复现；不同像素数按物理位移重采样；目标图无需有晶格；`identity_fallback` 包原样复制 + WARNING），并写出与拟合段同名的 `<stem>_corrected.h5` |
| `scripts/lf_lib.py` | 自包含辅助模块：lock-in（q 空间高斯掩码）、最小二乘（Poisson/DCT）相位解缠、位移场求解、按场重采样、场的网格重采样、绘图样式与 gwyddion 色图、环聚类 |
| `scripts/h5io.py` | 自包含 HDF5 writer：仓库统一 h5 约定（`SCHEMA_VERSION = 1`、`gzip`/4、显式 chunk ≤ 1 MiB、root `schema_version`/`generator`/`creation_date`、`units` 属性），**不 import `stm_data_processing`** |
| `scripts/selftest.py` | **一键自我验证**（**20 项**：拟合产物与契约行、h5 约定与数组逐位一致、已知位移场恢复（原始与去仿射规范）、环质量、第三方向一致性、包迁移（同网格/不同网格）、无晶格回退） |
| `SKILL.md` | 完整方法学：模型与符号约定、lock-in 与可解性条件 `\|∇u\| < 2π/(\|K\|λ)`、参考六方、规范自由度、有效掩码与边界、输出契约、h5 数据集与 `units`、迁移约束、自检与限制 |

## 依赖

- 仓库 `STM_DataProcessing` 源码（`src/`）：`utils.bragg_peak`（数据锚定峰检测/环聚类）、
  `utils.plot_funcs.subtractMeanPlane`、`stm.preview_plot.gwyddion`（色标，缺省 builtin 锚点回退）
- `scipy`（`fft.dctn/idctn`、`ndimage.map_coordinates/distance_transform_edt`）、`numpy`、`matplotlib`
- `h5py`（HDF5 产物；仓库 `pyproject.toml` 已含）
- 执行环境（系统 Python 无 numpy；不要用 `uv run`——它会写仓库 `.venv`）：

```bash
cd /path/to/STM_DataProcessing
export MPLCONFIGDIR=<可写目录> PYTHONDONTWRITEBYTECODE=1

# 矫正 + 导出可迁移包（示例：λ = 30 nm 对应 ≲0.7 % 应变的缓慢畸变）
.venv/bin/python skills/lawler-fujita-correction/scripts/stm_lf_correct.py \
    INPUT.csv -L 50 --a 0.246 --lambda-nm 30 -o OUT --save-transform OUT/bundle.json

# 一键 self-test（20 项）
.venv/bin/python skills/lawler-fujita-correction/scripts/selftest.py \
    --workdir var/lf_selftest --keep
```

## λ 怎么选（重要）

lock-in 只保留满足 `|∇u| < 2π/(|K|λ)` 的畸变结构，`|K| = 4π/(√3 a) ≈ 29.5 nm⁻¹`：

| λ | 能保留的应变上限 | 备注 |
| --- | --- | --- |
| 30 nm（默认） | 0.7 % | 论文默认量级；**要求 λ ≪ L**，50 nm 视场下只有中心区域可信 |
| 5 nm | 4 % | 强畸变（如合成自检）时的选择 |
| ≲1 nm | — | 掩码半径接近邻近 Bragg 峰间距，邻近环泄漏，不可用 |

此外必须满足 `λ · |q_measured − q_ref|/L < 1`（测得峰位落在掩码内）：否则脚本报 WARNING 并在报告里
记 `lockin.band_warning = true`——此时解出的相位不是晶格畸变。

## 输出

`<stem>_corrected.csv` / `<stem>_corrected.h5` / `<stem>_corrected_fft2.npy` /
`<stem>_corrected.png` / `<stem>_corrected_fft.png` / `correction.log` /
`correction_report.json`（apply 段同名，但报告为 `apply_report.json`），外加**一个**
`<stem>_lf.h5`，里面是 9 个数据集：`theta_a/b/c`（`units = "rad"`）、`amplitude_a/b/c`
（无量纲）、`u_x` / `u_y`（`units = "nm"`）、`mask`（无量纲）；
每张图另有 `<stem>_lf_<name>.png` 预览（默认三方向 9 张 png；`--pair-angle 60` 没有 c 方向，
7 个数据集、7 张 png，log 与报告会写明只有两个方向、间距 60°）。

`<stem>_corrected.h5` 是仓库 h5 约定的数组产物：数据集 `corrected`（`float64`，与 CSV 同一数组）
与 `fft2`（`complex128`，与 `.npy` 同一数组），root 有 `schema_version = 1` / `generator` /
`creation_date`，逐数据集显式 chunk + `gzip`/4 + `track_times=False`。
**`<stem>_corrected_fft2.npy` 同时保留**：`skills/phase-analysis/` 用它做 `--fft2` 输入
（phase-analysis 不在本次产物迁移范围）。CSV/PNG/log 名称与内容不变。

下游契约：`correction.log` 必须含 `# corrected canvas: <n_out> x <n_out> px, field of view <X> nm (...)`
一行（`phase-analysis --size-nm-from-log` 优先解析它），`X = L · n_out / n`。详见 `SKILL.md` §3。

## 可迁移包

`--save-transform FILE` 写出 `FILE`（变换 JSON，schema `lawler-fujita-correction-transform` v1）
与同名 `FILE` 换后缀的 `.h5`（数据集 `u_x`、`u_y`（`units = "nm"`）、`valid` 掩码（无量纲，无
`units`）、`field_of_view_nm_reference`（`"nm"`）、`n_px_reference`（像素数，非物理量，无 `units`）、
`nm_per_px`（像素尺度，`"nm/px"`）；后三个在 HDF5 里存成单元素数据集，
因为标量不能分块/过滤）；JSON 的 `u_field_file` 指向该 `.h5`。
`stm_lf_apply.py` 从 h5 读场并处理同一扫描的其它图：同网格逐位复现拟合段；不同像素数按物理位移插值；
目标图不需要有晶格；视场不匹配时打 WARNING；`identity_fallback` 包原样复制目标图。

## 修改记录

见 `CHANGES.md`。
