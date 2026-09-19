# lawler-fujita-correction（Lawler–Fujita 晶格相位矫正）

仓库内置 skill（**v1.0**）。**只做几何矫正数学**：局域 lock-in 相位提取、逐点位移场求解、按位移场
重采样、矫正后自检与可迁移包导出。物理解释由使用者进行——脚本、图、log 与文档都不给物理结论。
矫正产物可接**任意后处理**（例如 `skills/phase-analysis/` 的双环相位分析）。

与 `skills/affine-correction/` 的分工：后者是**全局仿射拉伸**（一个 2×2 对称正定矩阵），
本 skill 是**逐点位移场 u(r)**（每像素两个自由度），对应 Fujita et al., PNAS 2014 SI §4。

## 文件

| 路径 | 作用 |
| --- | --- |
| `scripts/stm_lf_correct.py` | 拟合段 CLI：检测 1x1 环 → 三个等间隔参考波矢（理想半径 + 数据取向）→ q 空间高斯 lock-in → 最小二乘相位解缠 → `u = Q⁻¹(θ̄ − θ)` → 按 u 重采样；输出矫正产物 + 相位/幅度/位移/掩码 9 张图 + 报告；`--save-transform` 导出可迁移包 |
| `scripts/stm_lf_apply.py` | **第二段**：把包里的稠密位移场套用到同一扫描的另一张图（同网格逐位复现；不同像素数按物理位移重采样；目标图无需有晶格；`identity_fallback` 包原样复制 + WARNING） |
| `scripts/lf_lib.py` | 自包含辅助模块：lock-in（q 空间高斯掩码）、最小二乘（Poisson/DCT）相位解缠、位移场求解、按场重采样、场的网格重采样、绘图样式与 gwyddion 色图、环聚类 |
| `scripts/selftest.py` | **一键自我验证**（**17 项**：拟合产物与契约行、已知位移场恢复（原始与去仿射规范）、环质量、第三方向一致性、包迁移（同网格/不同网格）、无晶格回退） |
| `SKILL.md` | 完整方法学：模型与符号约定、lock-in 与可解性条件 `\|∇u\| < 2π/(\|K\|λ)`、参考六方、规范自由度、有效掩码与边界、输出契约、迁移约束、自检与限制 |

## 依赖

- 仓库 `STM_DataProcessing` 源码（`src/`）：`utils.bragg_peak`（数据锚定峰检测/环聚类）、
  `utils.plot_funcs.subtractMeanPlane`、`stm.preview_plot.gwyddion`（色标，缺省 builtin 锚点回退）
- `scipy`（`fft.dctn/idctn`、`ndimage.map_coordinates/distance_transform_edt`）、`numpy`、`matplotlib`
- 执行环境（系统 Python 无 numpy；不要用 `uv run`——它会写仓库 `.venv`）：

```bash
cd /path/to/STM_DataProcessing
export MPLCONFIGDIR=<可写目录> PYTHONDONTWRITEBYTECODE=1

# 矫正 + 导出可迁移包（示例：λ = 30 nm 对应 ≲0.7 % 应变的缓慢畸变）
.venv/bin/python skills/lawler-fujita-correction/scripts/stm_lf_correct.py \
    INPUT.csv -L 50 --a 0.246 --lambda-nm 30 -o OUT --save-transform OUT/bundle.json

# 一键 self-test（16 项）
.venv/bin/python skills/lawler-fujita-correction/scripts/selftest.py \
    --workdir tmp_verify/lf_selftest --keep
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

`<stem>_corrected.csv` / `<stem>_corrected_fft2.npy` / `<stem>_corrected.png` /
`<stem>_corrected_fft.png` / `correction.log` / `correction_report.json`
（apply 段同名，但报告为 `apply_report.json`），外加
`<stem>_lf_theta_{a,b,c}` / `_lf_amplitude_{a,b,c}` / `_lf_u_x` / `_lf_u_y` / `_lf_mask`
各一份 npy + png（默认三方向共 18 个文件；`--pair-angle 60` 没有 c 方向，14 个文件，
log 与报告会写明只有两个方向、间距 60°）。

下游契约：`correction.log` 必须含 `# corrected canvas: <n_out> x <n_out> px, field of view <X> nm (...)`
一行（`phase-analysis --size-nm-from-log` 优先解析它），`X = L · n_out / n`。详见 `SKILL.md` §3。

## 可迁移包

`--save-transform FILE` 写出 `FILE`（变换 JSON，schema `lawler-fujita-correction-transform` v1）
与同名 `FILE` 换后缀的 `.npz`（`u_x`、`u_y` nm + `valid` 掩码）。
`stm_lf_apply.py` 用它处理同一扫描的其它图：同网格逐位复现拟合段；不同像素数按物理位移插值；
目标图不需要有晶格；视场不匹配时打 WARNING；`identity_fallback` 包原样复制目标图。

## 修改记录

见 `CHANGES.md`。
