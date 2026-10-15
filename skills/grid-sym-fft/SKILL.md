---
name: grid-sym-fft
description: 把 Nanonis grid spectroscopy（`.h5`，经 `stm_data_processing.io.grid2h5` 转换）的任意通道按名字抽成角色图（fwd/bwd 平均），用 `grid-lf-pipeline`（Skill A）导出的 Lawler–Fujita 位移场包做「只重采样、不改值域」的矫正，然后对每张 map 做矫正前/后的复数 FFT，并做绕 DC 像素的 Z3（复数域）与 C3（先取幅度再旋转 120/240 平均）三重对称化；第一组 FFT 图后**按角色各生成**多档色标候选（每角色一套 contact sheet）供用户挑选，选定后**同一角色**的 FFT 图共用该角色的窗口（角色之间绝不共用：current 与 didv 的单位不同，log|F| 中位数差 ~26）；rmap/zmap 不在范围内。当用户给出 grid `.h5` + 角色映射（如 current / didv）+ 位移场包，要求列出全部 channel、按名字抽取、套用 LF 位移场、出复数 FFT 与 Z3/C3 对称化、并先挑色标再出全套时使用。
---

# grid-sym-fft（grid → 角色抽取 → LF 位移场矫正 → 复数 FFT → Z3 / 绕 DC 的 C3，v1.0）

**范围**：读取 grid `.h5`，给出全部 channel 及其 fwd/bwd 配对，由用户指定每个要抽的角色（如 `current` / `didv`，角色名自由）→ 每个角色 = 前向通道与其 `[bwd]` 配对取平均，每偏压一张 map → 用 **Skill A（`grid-lf-pipeline`）** 导出的可迁移位移场包做**只重采样、绝不改值域、也绝不翻转**的矫正（map 按名字从 `/data` 取出后已在位移场的帧里，见 §2.2「坐标系约定」） → 出矫正前/后的复数 FFT → 绕 DC 像素做 **Z3**（复数域）与 **C3**（先取幅度）三重对称化。第一组 FFT（b 阶段）之后**按角色各**生成 `--window-candidates` 张色标候选 + 该角色自己的 contact sheet 供用户挑选；选定后**同一角色**的所有 FFT 图共用该角色的窗口（**角色之间绝不共用**：`current` 与 `didv` 单位不同，log|F| 中位数相差 ~26，池化窗口会把一个角色整幅压到色标顶部、另一个压到底部）。**rmap / zmap 明确不在范围内**。

本 skill **不拟合位移场**（那是 Skill A 的职责），也不修改 `skills/lawler-fujita-correction` 下任何文件——它按相对路径调用 `lawler-fujita-correction` 的 `lf_lib.warp_by_field`（与仓库根 `scripts/apply_lf_transform.py` 同一套只重采样逻辑）。脚本、log 与本文档都不给物理结论。

## 1 输入与执行环境

| 项 | 说明 |
| --- | --- |
| 输入 | `<grid>.h5`（`python -m stm_data_processing.io.grid2h5 <grid>.3ds -o <grid>.h5` 得到）；角色映射（`--map ROLE="CHANNEL NAME"`）；Skill A 的位移场包**或** `affine-correction` 的变换包 |
| h5 结构 | `/data (ny, nx, nch, npts)`、`/channels`、`/channel_units`、`/bias`（V，偏压标签只能从这里读）、`/params` + `/param_columns`、`/header` |
| 位移场包 | Skill A 产出的 `<stem>_transform.json`（含 `u_field_file`、`n_px_reference`、`field_of_view_nm_reference`、`pad`、`order`）+ `<stem>_transform.h5`（`u_x` / `u_y` / `valid`，nm / 参考网格）。Skill A 写出的 `u_field_file` 是**绝对路径**（`grid_lf_fit.py` 对 `--save-transform` 做 `.resolve()`）；若遇到相对路径，按 bundle 所在目录解析（`scripts/grid_sym.py:481-482`，即 `load_transform()` 里挨着的两行 `if not u_path.is_absolute():` / `u_path = bundle_json.parent / u_path`；**行号漂了就按这两行原文找**） |
| 变换包（两条通路） | `--transform` 同时接受上面那种 **LF 位移场包** 与 **affine 变换包**（`affine-correction` 的 `transform.json`：schema `topo-correction-transform`，含 `affine_q`（2×2 对称正定拉伸 M）、`pad`、`order`、`n_px_reference`、`field_of_view_nm_reference`，以及溯源字段 `method` / `fallback` / `n_labelled` / `anchor_ring` / `anchor_verdict` / `stretch_scale_sqrt_det`）。判别口径见 §2.4，affine 通路的产物路由见 §3 |
| 视场 L (nm) | 从 h5 header 解析 `Scan>Scanfield`（`;` 分隔，第 3 段宽 ×1e9=nm），否则退回 `Grid settings`；与 Skill A 的 `grid_topo.py` 同源 |
| 解释器 | 仓库 venv：`cd /path/to/STM_DataProcessing && MPLCONFIGDIR=<可写目录> PYTHONDONTWRITEBYTECODE=1 .venv/bin/python <脚本>`（不要用 `uv run`）。**`PYTHONDONTWRITEBYTECODE=1` 是必需的**：本 skill 与 Skill A 都会 import `skills/lawler-fujita-correction/scripts/lf_lib.py`（`grid_sym.py` 把它接进 `sys.path`），不设它就会在 `skills/lawler-fujita-correction/scripts/` 下留下 `__pycache__/`（仓库不该有这些字节码缓存；`.gitignore` 忽略它们，但目录会堆在 checkout 里） |
| 依赖 | `numpy`、`h5py`、`scipy.ndimage`、matplotlib、Pillow（contact sheet），以及本仓库 `skills/lawler-fujita-correction/scripts/lf_lib.py`（**相对路径嵌套调用，不修改它**） |

**按名字解析，禁止任何通道索引字面量**。通道索引一律由 `list.index(name)` 在运行时求出；代码里不出现 `data[:, :, N, :]` 这类数字索引字面量（有 grep 门）。

## 2 两个脚本

### 2.1 `scripts/grid_channels.py`（列出全表 + 确定角色）

```bash
.venv/bin/python skills/grid-sym-fft/scripts/grid_channels.py GRID.h5 \
    --map current="Current (A)" --map didv="DSP 7280 Y (%)"
```

- 启动即读 h5，打印 `/channels` 全表：索引、通道名、单位、以及由名字推出的 fwd/bwd 配对。配对规则：标记插在**尾部单位括号之前**，即 `Current (A)` ↔ `Current [bwd] (A)`（Nanonis 的命名法）；无尾部单位时退化为 `X` ↔ `X [bwd]`。表里只对**该 h5 里确实存在**的孪生通道标 `fwd of ...`。
- **非交互**：每个 `--map ROLE="CHANNEL NAME"` 给一个角色与一个通道名，脚本打印已解析映射后退出 0。
- **交互**：不带 `--map` 时在 stdin 上逐行问「role = channel name」，空行结束。
- 名字匹配不到或匹配到多个 → 列出全部可用名字并非零退出（fail-closed），不得猜；给出 `[bwd]` 名字直接拒绝（fwd 名字是规范入口）。

> grid_channels.py **只打印映射**；真正的抽取在 `grid_sym.py`。

### 2.2 坐标系约定（**改 map 之前先读这一节**）

四个产物（h5、topo CSV、位移场包 u、本 skill 的 map）之间的帧关系是固定的，写死如下（2026-10 用
`/params` 的 X/Y 列做地面真值逐条实测）：

| # | 事实 | 依据 |
| --- | --- | --- |
| 1 | h5 的 `/params` 与 `/data` **共用同一扁平索引**：行号沿物理 Y **递增**（`/params` 的 `Y (m)` 第 0 行最小、末行最大；实测列 0 从 +183.22 nm 单调升到 +212.20 nm） | `grid2h5` 的写盘顺序 |
| 2 | `grid-lf-pipeline/scripts/grid_topo.py:196` 写 topo CSV 时 **flipud 一次**：CSV 帧 = h5 原始帧翻转（CSV 第一行 = 扫描顶行 = 最大 Y） | 代码 |
| 3 | `lawler-fujita-correction/scripts/stm_lf_correct.py:639`（该文件 `main()` 里那句 `topo = np.flipud(subtract_mean_plane(topo))`；**行号漂了就按这个锚点找**，2026-10-14 复核过）读 CSV 后**又 flipud 一次**才拟合 ⇒ **位移场 u 定义在 h5 原始帧**（`n_px_reference = 304` 就是该帧的画布） | 代码 |
| 4 | `stm_lf_correct.py` 的产物**自 2026-10-14（t1）起与它的输入 CSV 同帧**：重采样仍在内部帧做，写回前翻一次再落盘；`stm_lf_apply.py:265`（同样是 `main()` 里那句 `topo = np.flipud(subtract_mean_plane(topo))`）对**它的**输入 flipud（进入同一内部帧），产物同样按输入 CSV 帧写出 ⇒ 该工具**期望的输入是第 2 条那种 CSV 帧，输出也在那一帧** | 代码（2026-10-14 起；行号同日复核） |
| 5 | `grid_sym.py:extract_role_map()` 取 `/data` **不翻** ⇒ map 已经在 u 的帧里 ⇒ **喂给 `warp_by_field` 前不得再翻**（多翻一次 = 位移场被上下镜像施加；2026-10 实测同一物理标记在两条路径上相差 **276 px**，纯镜像应为 `n-1-2r = 263 px`） | 代码 + 标记注入实测 |
| 6 | **affine 通路**：`affine_q` 与 `_image_transform(A, n, pad)` 的 `n_out`/offset **直接作用在 h5 帧的 map 上（0 次 flip）**。affine skill 自己的 apply 段（`stm_apply_transform.py`）读的是 **topo CSV 帧**（§2.2 第 2 条：该帧 = h5 帧翻转一次），所以它在那里 flipud（产物自 2026-10-14 起也按该输入 CSV 帧写出）；把那个输入 flipud 照抄过来就是上下镜像的矫正。实测（同一标记，两条路径都折算到 h5 帧）：affine 通路与 `stm_apply_transform.py` 的产物行差 ≤0.1 px；不折算（忘掉那个 flip）则相差纯镜像量 `n_out-1-2r`（本例 n=304→n_out=331，r=20 时 ~290 px） | 代码 + 标记注入实测（§8 第 7 条） |

一句话：**h5 原始帧 = 位移场帧 = affine 包的数组帧；本 skill 全程不翻转**。CSV 帧（topo CSV / 两个 apply 工具的输入与产物帧）
与它差一次 flip，那是 Skill A / affine skill 内部各自约定的事，与本 skill 无关。

**显示约定**：实空间图与 FFT 图统一用 matplotlib `origin="lower"`（产物图的渲染入口是
`save_array_image()`（`grid_sym.py` 的 `def` 在 680；实空间图经 `save_map_image()`、FFT 图经
`save_fft_image()` 都只调它），里面那个 `ax.imshow(mat, ..., origin="lower", interpolation="nearest", aspect="auto")`
调用的 `origin="lower",` 行在 **720**——**行号漂了就按这个锚点找**），
与 LF 家族一致（`stm_lf_correct.py:1031/1047`、`stm_lf_apply.py:398/414`——每对是「实空间图的
`ax.imshow(corrected, cmap=cmap_bad, origin="lower")` 行 / FFT 图里那个 `origin="lower",` 行」，
**行号漂了就按这两个锚点找**，2026-10-14 复核过）——即**数组第 0 行画在画布底部**，
于是矫正后的 didv 图与矫正后的 topo 图朝向一致（同一物理标记在两张图里的相对行位置一致，实测差 0.56 %）。
用 `origin="upper"`（旧行为）会让本 skill 的图与 LF 产物上下颠倒。

**机械自检（标记法，不看图）**：造一张**零背景**的高斯标记图（σ=2 px，h5 帧行 `r`），把它按 h5 帧
喂给 `apply_transform`（LF 通路）/ `apply_affine`（affine 通路），把它的 `flipud`（同一物理像素在 CSV 帧里的样子）
喂给 `stm_lf_apply.py` / `stm_apply_transform.py`，然后比两份 corrected CSV 里标记的**亚像素质心行**
（两条通路的产物都先 `flipud` 回 h5 帧再比——2026-10-14 起两个 apply 工具的产物是「与输入 CSV 同帧」）——
必须相同（实测 11-15 与 11-08 都是 **0.000 px**，判据 ≤3 px；affine 通路实测 ≤0.1 px）。
零背景很关键：在真实 map 上做同样的事，标记与图自身结构混在一起，centroid 会偏几个像素，
把量测误差误读成帧差。**反例判据**：同一比法若不折算那个 flip 就会偏出纯镜像量 `n-1-2r`
（本仓库 11-15：`r=20, n=304` → LF 通路 263 px 的纯镜像、实测 276.19 px，多出的 13 px 是 u 在两个位置上的差异；
affine 通路 `n=304→n_out=331`，纯镜像 ~290 px），偏不出这个量就说明门在空转。
完整可跑版本见 §8 第 7 条。

### 2.3 画法约定（**所有数据图**，2026-10 用户要求）

用户原话：「生成的图片你不要再加 colorbar 了，我没法直接用，按照我给你的那个文件夹中的画法，直接 pad=0」。
参考目录：`data_processing/20251110/20251110_grid1_30mV500pA_30nm/` 下全部 fft 相关目录
（实测 `didv_corrected_upscaling_fft_C3dc_hamming/*.png` 是 2250×2250、四边 2 px 内非白比例 **1.000**、
逐像素 LUT 匹配 **plasma** 中位距 0.8，magma/inferno/turbo/cividis/viridis 分别 33/40/43/52/55）。

| 项 | 约定 |
| --- | --- |
| 内容 | **只有数据像素**：无 colorbar、无坐标轴/刻度/标题/边框、无留白（`fig.add_axes([0,0,1,1])` + `ax.set_axis_off()` + `savefig(pad_inches=0)`，不加 `bbox_inches`） |
| 几何 | **1:1 像素**：PNG 边长 == 数组边长（`figsize = n/png_dpi`，`png_dpi=300` 与参考图/LF 家族同 dpi；`interpolation="nearest"`） |
| 窗口 | **沿用调用方选定的 vmin/vmax**（K 档候选或 `fft_window.json`），渲染函数绝不自己按 min/max 归一 |
| colormap | **FFT 图（`fft_raw` / `fft_corrected` / `z3` / `c3dc`）= `FFT_CMAP = "plasma"`**；**实空间图（`raw` / `corrected`）= `MAP_CMAP = "viridis"`**。两个常量都在 `grid_sym.py` 顶部一处定义（single source），改一处即全局生效 |
| NaN | 用 LF 家族的 `lf_lib.BAD_COLOR = "#b0b0b0"`（不透明灰）画，不用 matplotlib 默认的透明/白——矫正画布四周本来就是无位移场的 NaN，透明会留下"白边"、也不便于直接叠图 |
| 例外 | `window_candidates/<role>/window_KK.png` 与 `contact_sheet.png` **保留现状**（带 vmin/vmax/压死 % 标注 + colorbar）：它们是选窗工具，去掉标注就没法选；它们同样用 `FFT_CMAP` |

渲染入口：`save_array_image()`（1:1 无装饰）→ `save_map_image()`（实空间，viridis）与
`save_fft_image()`（`log|F|`，plasma）。机械自检见 §8 第 8 条（把 PNG 从数据重算出来逐像素比）。

### 2.4 `scripts/grid_sym.py`（抽取 → 矫正 → 复数 FFT → Z3 / C3）

`--transform` 接受**两种包**，脚本自己判（`scripts/grid_sym.py` 的 `classify_transform()`），判不出来就 fail-closed：

| 判别（按顺序，互斥） | 结论 | 依据 |
| --- | --- | --- |
| `schema == "topo-correction-transform"` **且** `affine_q` 是有限 2×2、`det > 0` | **affine 通路** | schema 优先 |
| 含 `u_field_file` 键 | **LF 通路** | 位移场的登记键 |
| 两者同时成立 / 都不成立 | 非零退出，stderr 说明两个理由 | `fallback` 键**不参与**判别（affine 包在没拟合出东西时也会带 `fallback: true`，那仍是 affine 包） |

`--transform` 传**目录**时同时认 `*_transform.json`（Skill A 的命名）与 `transform.json`（affine skill 的命名）；
目录里出现 ≥2 个候选（例如 LF 与 affine 各一份）非零退出并列出候选文件名。

```bash
# 第一次运行：先出色标候选（a+b + 候选 + contact sheet），exit 0
.venv/bin/python skills/grid-sym-fft/scripts/grid_sym.py GRID.h5 \
    --map current="Current (A)" --map didv="DSP 7280 Y (%)" \
    --transform <SkillA 的 transform.json 或含它的目录> -o OUT \
    --window-candidates 9
# 第二次运行：用户挑候选 K 后出全套（a+b+c+d+e），写 fft_window.json
.venv/bin/python skills/grid-sym-fft/scripts/grid_sym.py GRID.h5 \
    --map current="Current (A)" --map didv="DSP 7280 Y (%)" \
    --transform <...> -o OUT --pick-window K        # 每个角色各取自己的第 K 档
# 或用户给出显式窗口（角色之间单位不同，推荐按角色给）
.venv/bin/python skills/grid-sym-fft/scripts/grid_sym.py GRID.h5 \
    --map ... --transform <...> -o OUT \
    --fft-window-role current -22.5 -17 --fft-window-role didv 4 9
# 或显式指定空间窗（默认 hamming；独立 flag，见 §2.5）
.venv/bin/python skills/grid-sym-fft/scripts/grid_sym.py GRID.h5 \
    --map ... --transform <...> -o OUT --spatial-window hanning   # none | hamming | hanning
# affine 通路：同一个 -o OUT，产物自动前缀到 OUT/affine/ 下（只落 c–e 阶段）
.venv/bin/python skills/grid-sym-fft/scripts/grid_sym.py GRID.h5 \
    --map current="Current (A)" --map didv="DSP 7280 Y (%)" \
    --transform OUT2/affine/transform.json -o OUT        # 色标窗继承 OUT/fft_window.json
# affine 通路 + --upsample N>1：一次 cubic 重采样到 n·N 细网格（画布 = 足迹方形 bbox）
#             + FFT 家族裁回原生 q 窗口（见 §2.6；默认 1 = 现行为）
.venv/bin/python skills/grid-sym-fft/scripts/grid_sym.py GRID.h5 \
    --map current="Current (A)" --map didv="DSP 7280 Y (%)" \
    --transform OUT2/affine/transform.json --upsample 8 -o OUT
# 两个 flag 各管一件事，不共用：
#   --spatial-window {none,hamming,hanning}  数据窗（fft2 之前乘 2D 窗）
#   --fft-window VMIN VMAX                   色标窗，**必须两个浮点数**（套给所有角色；
#                                            角色单位不同，推荐 --fft-window-role）
# 单 token 的 --fft-window（例如 --fft-window 3）会 fail-closed（exit 2）并提示改用
# --spatial-window——一个 flag 两种含义曾在脚本/文档/会话里反复咬过。
```

**两条通路的差别（同一段代码，只有矫正怎么算是不同的）**：

| 项 | LF 通路 | affine 通路 |
| --- | --- | --- |
| 矫正 | `lf_lib.warp_by_field`：`corrected(r) = T(r + u(r))`，画布 `n + 2(⌈max|u|⌉ + pad)` | `A = P·M⁻¹·P` + `_image_transform(A, n, pad)` 的 `n_out`/offset，`scipy.ndimage.affine_transform(order, cval=NaN)`；画布按**目标 n 现算**（本数据 304 → 331，不硬编码）。`--upsample N > 1` 换成**一次 cubic 重采样到 n·N 细网格**、画布 = 足迹方形 bbox（2432²，无 pad 带），FFT 家族再裁回原生 q 窗口（310²）——见 §2.6 |
| 帧 | h5 帧，0 次 flip（§2.2 第 5 条） | h5 帧，**0 次 flip**（§2.2 第 6 条） |
| 阶段 | a+b+c+d+e（第一次还出候选） | **只有 c+d+e**：不写 `raw/`、`fft_raw/`、`window_candidates/`（§3） |
| 产物根 | `<out>/` | **`<out>/affine/`**（调用方仍传 `-o <out>`） |
| 色标窗 | 自己选（`--window-candidates` → `--pick-window` / `--fft-window*`），写 `<out>/fft_window.json` | **不产生候选**：继承 `<out>/fft_window.json`（自身 `<out>/affine/fft_window.json` 优先），并把本次选择写回 `<out>/affine/fft_window.json`；`--window-candidates` / `--pick-window` 在此通路 fail-closed |
| identity_fallback | Skill A 的回退语义（本 skill 的 LF 通路在 `load_transform()` 处拒绝无 u 场的包） | 照 `stm_apply_transform.py` 的语义：**拷贝原图**（`n_out = n`、视场不变、不加 NaN）、log 打 WARNING、provenance 记 `corrected_this_run: false`，z3/c3dc **照跑** |
| 溯源 | `pipeline.log` | `pipeline.log` 里的 provenance 段 + 机器可读 `<out>/affine/affine_provenance.json`（包路径 / schema / method / `affine_q` / pad·order / `n_in→n_out` / `n_labelled` / `anchor_ring` / `anchor_verdict` / `stretch_scale_sqrt_det`，并显式声明 **applied in the h5 frame, no flip** 与 **anchor self-check NOT re-run**——`anchor_verdict` 是拟合段的结论，本 skill 不重跑自检，本数据是 `unverifiable`：别把它读成"已验证"） |

按下列顺序执行，**每一步都写数据 + 图 + log，且每一步都配 FFT**（affine 通路从 c 起）：

**a. 抽取**（仅 LF 通路落盘）：每个角色 = 前向通道与其同名 `[bwd]` 配对（`Current (A)` ↔ `Current [bwd] (A)`）取平均，每偏压一张 map（bias 标签从 `/bias` 读，遍历全部 bias 点；配对规则与两个名字、两个索引都写进 log）。写 CSV + PNG（**实空间图 = viridis**，窗口 = 该图 `nanmedian ± 5σ`，窗口数值写进 log；画法见 §2.3）。CSV 必须等于两方向平均（可用 `/data` 的 fwd/bwd 两片独立复算核对；只取前向时差可达 ~1e-10 量级）。affine 通路同样抽取（log 里打同一行，末尾标 `in memory only`），但不落盘。

**b. 矫正前 FFT**（仅 LF 通路）：`x = map`（NaN 用有效像素 nanmedian 填充），**再加空间窗** `x*W`（`W = outer(w,w)`、`w = np.hamming(n)` 或 `np.hanning(n)`，见 §2.5；`--spatial-window none` 时 `W` 不存在）；`F = np.fft.fftshift(np.fft.fft2(x*W))`，complex128 存 npy；出 `log|F|` 图（**FFT 图 = plasma**，见 §2.3）。**每个偏压写两套**：无窗（既有文件名）+ 加窗（名字带窗名后缀）。这一组图后面跟色标候选（**候选与候选图按当前空间窗的平面算**，见 §2.5）。

**c. 矫正**：LF 通路用 `--transform` 指向的位移场包矫正每个角色每张 map（**只重采样、绝不改值域、也绝不翻转**，见 §2.2）；affine 通路用 `A = P·M⁻¹·P` 与同一 offset 直接作用在 h5 帧的 map 上（**同样 0 次翻转**，§2.2 第 6 条；画布按目标 n 现算）。两条都：写 corrected CSV + PNG（viridis，窗口 = 该图 `nanmedian ± 5σ`）+ 该阶段自己的复数 FFT + `log|F|` 图；**禁止**调用会去平面的 `stm_lf_apply.py` / `stm_apply_transform.py`（它们属于各自的 CSV 帧入口，且 affine 那个还会减平面——本 skill 用的是本地镜像的几何）。

**d. Z3**：对每张矫正后 map 的 `F`，`Z = (F + R120 F + R240 F)/3` 在**复数域**做，旋转**绕 DC 像素**：`R` 为平面内 120/240 度旋转矩阵，`offset = c - R c`，`c = fftshift 后 DC 的像素坐标（浮点，偶数 N 时为 N/2）`，`scipy.ndimage.affine_transform(matrix=R, offset=c-R c, order=3, mode='nearest')`。存 complex128 npy + `log|Z|` 图，log 报 DC 保留（偶数画布应为 1.0000，本 skill 的 affine 画布在 N>1 时是偶数 2432、在 N=1 时是奇数 331，见 §2.6 末注）、三重残差输出/输入之比、能量比。

**e. 绕 DC 的 C3**：先取幅度 `A = |F|`，再同样绕 DC 像素转 120/240 并平均 `C3dc = (A + R120 A + R240 A)/3`（与 Z3 **唯一区别就是先取模**）。存 npy + `log|C3dc|` 图。三次插值振铃会给出少量负值，故 log 用 `log|C3dc|`，并把负值个数与相对幅度记进 log。

**f. 色标（按角色，永不跨角色池化）**：第一组（b 阶段）FFT 图之后，**对每个角色各**生成 `--window-candidates N` 张（**N 必填：该参数没有默认值**）不同 vmin/vmax 的候选 + 该角色自己的一张 contact sheet，落在 `<OUT>/window_candidates/<role>/`。每个角色的候选窗口取自**该角色自己**的 log|F| 分布（**当前空间窗**下该角色全部 bias 平面的并集），因此 current 一套（~−24…−15）、didv 一套（~+2.5…+10.5）互不影响。候选必须同画布尺寸（PIL 断言全部一致）。**候选表与候选图都建立在当前空间窗的平面上**（默认 hamming），所以「你看到的候选 == 最终画布」；同一档窗在**无窗平面**上的数字也同时印出（`plane_none%` 列）供对照。

**窗口与图是两个口径，表里必须分开印（2026-10 修）**：窗口的 vmin/vmax 是该角色**并集分布**的分位数（选窗要用它，因为窗口要套到该角色所有 bias 平面），而候选**图**只能画其中一张 plane——为此取该角色中分位数向量（p0.5/p50/p99.5）最接近其并集分布的那张偏压平面（表里记 `representative_bias_index` / `representative_bias_meV`）。两者会有小幅漂移，所以候选表对每档同时给两套数字，**字段名带口径后缀，禁止同名不同义**：

| 字段（`fft_window_candidates.json` → `roles.<role>.candidates[i]`） | 含义 |
| --- | --- |
| `vmin`, `vmax` | 窗口本身，**并集分位数**（用户选的就是这一对） |
| `clip_below_pct_pooled`, `clip_above_pct_pooled` | 该角色**并集分布**中落在窗口外的像素占比（按构造等于设计档位的百分数） |
| `clip_total_pct_pooled` | 上面两者之和 |
| `clip_below_pct_plane`, `clip_above_pct_plane` | **画布上那张 plane**（代表平面）实测落在窗口外的像素占比——这才是与图对应的数（数据侧，不含坐标区/文字/色条） |
| `clip_total_pct_plane` | 上面两者之和 |
| `clip_budget_pct` | 该档的设计预算（= 选档时用的两个百分数之和，如 index 8 = 2.20 + 2.30 = 4.50 %） |
| `plane_over_budget_pct` | `clip_total_pct_plane − clip_budget_pct`，带符号；超出容差时表格里打 `!` |

候选图的标题同样两行印出 `clip pooled a/b %  plane c/d %`（`scripts/grid_sym.py` 的 `render_candidates()`），避免"标题上的数"与"画布上的图"对不上。

**warning 语义**：代表平面的端点占比本就允许比并集口径略高（并集比任何单张平面宽），容差写在脚本常量 `PLANE_CLIP_TOL_PCT = 0.5`（百分点）。若某档 `plane_over_budget_pct > PLANE_CLIP_TOL_PCT`，脚本在 stdout 与 `pipeline.log` 各打一行 `WARNING candidates <role> window KK: ...`，点名档位、实测端点占比、超预算量、并集口径数值与**未改动**的 vmin/vmax——不静默、也不因此改窗口（窗口仍是并集分位数）。实测：11-08 的 didv 第 5–8 档触发（超 +0.503/+0.561/+0.557/+0.529 pp），11-15 两角色 18 档全部不触发。

候选窗口按 `vmin% + (100 − vmax)% < 5 %` 选（index 0 最宽、index 8 对比最强），即任何一档的端点压死像素都 < 5 %——整幅一个颜色的候选图没有信息，用户看不了。**注意**：这条约束不允许 "vmin = p50" 那类深背景风格（会有 50 % 像素压在底端），要那种观感请用 `--fft-window-role` 显式给窗口。

用户选定后用 `--pick-window K`（读上一轮的候选表；**`K` 是每个角色各自候选表的 `index`，0-based，即该角色目录下 `window_{K:02d}.png` 的编号**，合法范围 `0 <= K < N`），或 `--fft-window-role ROLE VMIN VMAX`（可重复，按角色给显式窗口），或 `--fft-window VMIN VMAX`（= 同一对数值套给所有角色）重跑，写入 `<OUTDIR>/fft_window.json`（格式 `{"window_version": 2, "roles": {<role>: {vmin, vmax}}}`）；此后每个角色自己的所有 FFT 图（fft_raw / fft_corrected / z3 / c3dc）用**该角色**的窗口。候选模式**优先于**盘上的选择：`--window-candidates N` 永远重算候选并覆盖候选表（覆盖时打 NOTE），不受 `fft_window.json` 影响；候选表若没有基准字段（t16 以前的旧表），`--pick-window` 会打一行 NOTE 说明"该表未记录基准、其窗可能在另一套平面上定标"——不再把未知当成已知。已存在 `fft_window.json` 时：**给了任一显式入口就覆盖它**（并打一行 NOTE 说明被覆盖的旧值与生效的入口，见 §5），**没给显式入口才复用**（幂等）；若是旧的单窗口格式（无 `roles` 键）则警告后套给所有角色。尚未选窗时只做 a+b 并退出 0，提示用户选窗后重跑。

### 2.5 空间窗（FFT 前对 map 加窗，默认 hamming；两套并存）

用户旧流程的参考目录里同时存在无窗与加窗变体（`didv_corrected_upscaling_fft_C3dc_hamming/`、
`didv_corrected_upscaling_fft_hamming/`、`corrected_didv_fft_hamming/`、
`didv_corrected_upscaling_fft_Z3/{pure,hanning,hamming}_z3/`），本 skill 因此**每张 map 出两套**。

| 项 | 约定 |
| --- | --- |
| 窗函数 | `--spatial-window {none,hamming,hanning}`（**独立 flag**，不与 `--fft-window` 共用），**默认 hamming**；`none` = 历史行为（只有无窗那套，逐字节不变）。作用位置：`F = fftshift(fft2(x * W))`，即窗加在 **fft2 之前**的空间域（不是频域滤波）。**两条通路都生效**：affine 通路同样按 `--spatial-window` 出两套，只是落在 `<out>/affine/` 下（其 `fft_corrected/z3/c3dc` 的 `taps` 是 affine 画布边长，本例 331） |
| 1D/2D | `W = np.outer(w, w)`，`w = np.hamming(n)`（或 `np.hanning(n)`），`n` = 该阶段的输入画布边长：`fft_raw` 用 map 边长（本例 304），`fft_corrected`/`z3`/`c3dc` 用矫正画布边长（本例 350）。窗永远作用在 **NaN 填充之后**的图上 |
| 命名 | 无窗那套**文件名不变**：`<label>_fft2.npy`、`<label>_fft.png`。加窗那套在**同一目录**下加窗名：`<label>_fft2_hamming.npy`、`<label>_fft_hamming.png`（`--spatial-window hanning` 时后缀是 `_hanning`）。四个族（`fft_raw`/`fft_corrected`/`z3`/`c3dc`）都如此——族由**目录**区分、窗由**文件名后缀**区分，无歧义 |
| Z3/C3dc | 两套各自从**自己那个 F** 出发：无窗 `z3_symmetrize(F)`、加窗 `z3_symmetrize(F_W)`（C3dc 同理，`C3dc = (|F| + R120|F| + R240|F|)/3`）。绝不拿无窗的 F 去算加窗的 Z3 |
| 色标 | 两套共用**该角色已选的同一个色标窗**（K 档或 `fft_window.json`）；加窗平面**不重算**窗口。因为候选按当前空间窗生成，色标窗对**加窗**套是对口径的，**无窗（pure）套反而是离基准的那一套**：每族每偏压打一行 `fft window clip ...`（加窗 vs 无窗的端点压死比例与总差），> 0.5 pp 再打一行 warning 指明哪一套离基准（实测 `current` 首张 bias 两套相差 +9.2 pp） |
| 记录 | `fft_window.json` 新增 `spatial_window` 段：`{"name", "suffix", "taps", "files"}`，`files` 是本次**实际写出的加窗文件清单**（相对 `<OUT>`）；`pipeline.log` 也有一行。`roles`（色标窗）语义不变 |
| 候选基准 | 候选表与候选图**跟随当前空间窗**（默认 hamming；`--spatial-window none` 时即无窗平面），且**按值可验**：`fft_window_candidates.json` 顶层写 `candidate_basis`（= 本次空间窗名，`spatial_window.candidate_basis` 同值），其 `roles.<role>.pooled_log_p50` 必须等于**从该套 `*_fft2[_<win>].npy` 独立算出的 p50**、且**不等于**另一套的 p50（§8 第 10 条）。表里每档给两列：`plane_<win>%`（画布口径，与最终图一致）与 `plane_none%`（同一档窗在无窗平面上的数字，供对照） |
| 记录 | `fft_window.json` 属**配置记录**：`{"name", "suffix", "candidate_basis", "taps", "files"}`；`candidate_basis` 记候选表所用基准（正常等于 `name`；若用另一个 `--spatial-window` 挑的窗则不同并打印 NOTE）。**该文件允许字节变化**——t14 那条"逐字节不变"的要求自此只适用于数据数组（`.csv`/`.npy`）与图，不适用于这个 JSON。它是"上一次选择的记录"，不是权威：显式命令行入口一律覆盖它（§5 第 3 条） |

与用户旧目录名的对应关系（本 skill 一个 `<OUT>` 同时含两套，不再分目录）：

| 用户旧目录 | 本 skill 里的等价位置 |
| --- | --- |
| `didv_corrected_upscaling_fft/`（无窗）/ `.../didv_corrected_upscaling_fft_hamming/` | `fft_corrected/<role>/<label>_fft2.npy` / `<label>_fft2_hamming.npy`（图 `_fft.png` / `_fft_hamming.png`） |
| `.../didv_corrected_upscaling_fft_Z3/{pure,hanning,hamming}_z3/` | `z3/<role>/<label>_fft2.npy`（pure=无窗）、`..._fft2_hanning.npy`、`..._fft2_hamming.npy` |
| `.../didv_corrected_upscaling_fft_C3dc_hamming/` | `c3dc/<role>/<label>_fft2_hamming.npy` |
| `corrected_didv_fft_hamming/` | `fft_corrected/<role>/<label>_fft2_hamming.npy`（同一物） |

（旧目录名里的 `upscaling` = 4× zoom 后再 FFT：那是用户 2025-11-10 的旧路子（先矫正再上采样）；本 skill 的 `--upsample N` 按用户 2026-10 的最终选择改口径为「一次 cubic 重采样到 n·N 细网格 + 画布取足迹方形 bbox + FFT 裁回原生 q 窗口」，**没有**线性预放大那一步，见 §2.6。）
**口径说明**：默认 hamming 下「选窗的平面」与「最终画的平面」是同一套，色标不再对加窗图偏紧；无窗（pure）套用同一个色标窗时可能与基准差一个量级（见上表 warning），那是**故意的**——pure 套保留原样、不改窗。

### 2.6 `--upsample N`：一次 cubic 重采样到 n·N 细网格（只对 affine 通路）

**口径出处（取代了最初的口述，2026-10-15）**：用户最初的口述是「对图像 N 倍**线性**上采样，之后再做仿射变换」，那句来自他对旧 notebook 的记忆；把 notebook 的真实机制（`论文/c6lic6/data_processing/gSiC_Dleft/data_processing.ipynb` 的 `didv_correction`：先 `coords_transformed = (T @ coords_original.T).T`，再交给 `_resample_coords()`，后者只做**一次** `scipy_griddata(..., method="cubic")` 重采样到指定细网格，画布取变换后足迹的方形 bbox、`linspace` 含端点、外插 NaN）摊开后，他在 2026-10-15 本会话的「①/②/③ 各选 A」问答里明确选定 **①A/②A/③A**，**取代**最初的口述。仓库内没有那次问答的机器可读记录，出处是本会话（队长裁定，t3 F1）；因此本 skill **不实现**「先 `zoom(order=1)` 再 warp」的两步链。

**口径**：坐标变换一次算好，在**原图**上做**一次**三阶（cubic）重采样；画布按 notebook 的 **bbox 语义**；FFT 家族再裁回**原生 q 窗口**；全程不做任何值域操作。`--upsample N` 默认 **1**（= §2.4 的现行为）；`N > 1` 时：

**两条链不是同一件事，产物确实不同**：一步 cubic 与「先 `zoom(N, order=1, grid_mode=True, mode="nearest")` 再 cubic warp」走不同的插值路径，同一张 map、同一最终几何下逐点比较**非零**（|Δ| 的 median 量级 ~0.08 %（current）/ ~0.51 %（didv）值域、max ~17 % / 14 %）。量级本身**不作为交付数字**：两步探针必须先做索引换算（第二道 affine 用 `matrix_two = N·A_fine`、`offset_two = N·off_fine + (N−1)/2`，因为 zoom 的 grid_mode 把细索引 `j` 映到源坐标 `(j+0.5)/N − 0.5`），再加 `mode`/`cval`、比较掩码（是否 `isfinite(A) & isfinite(B)`）与是否剔除足迹边缘的不同选择，median/p99/max 会在同一张 map 上成倍变化——早期写进本文档的那套数字正是选错换算得出的，已按 t5（N1）撤掉。**唯一稳定、可机械判定的判据在 §8 第 12 条：把 map 先 zoom 再 warp 的结果必须与出厂 corrected 不同（非零）**，相同就说明实现退回了「先线性上采样再仿射」的两步链。t3 提到的「corrected 略软」属旧两步链；本文档不对两条链的高频含量作强弱结论。

| 项 | 约定 |
| --- | --- |
| 重采样 | **一次** cubic 直接作用在**原图**上：`scipy.ndimage.affine_transform(map, A/scale, offset, output_shape=(n·N, n·N), order=<包的 order>，本例 3, mode="constant", cval=NaN)`。没有预放大、没有中间细网格；NaN 只是「采样点落在原图外」的结果 |
| 画布（bbox 语义） | 画布 = **n·N × n·N**（本例 304×8 = **2432²**）：把**变换后足迹**的**方形 bbox**（bx、by 都取 max span）**铺满** n·N 像素，**不加 pad 边带**。足迹 = 源图 n 像素方块的**像**，其轴对齐半宽 `extent = max|A⁻¹·corners|`（corner = ±(n-1)/2，`A = P·M⁻¹·P`）——与 notebook `_resample_coords` 里「把变换施于**源**网格、取变换后点的 `ptp`」是同一件事。N=1 的画布仍是旧语义 `2(⌈extent⌉+pad)+1`，两种语义以 N=1/N>1 分界 |
| 实算几何 | `extent = (150.2677, 154.9729)`、**span = 309.9459 粗像素**、canvas **2432²**、`scale = (n·N−1)/span = 7.843304` 画布 px/粗像素、`FOV_out = size_nm·span/n = 30.5868 nm`、`Δx_out = FOV_out/(n·N) = 0.0125762 nm/px`（按「像素中心铺满 bbox」实现取 `FOV_out/(n·N−1) = 0.0125814`，差 0.04 %） |
| 与契约估计的差 | 契约给的量级 span ~312.7 / FOV ~30.9 / Δx ~0.01269 用的是 **A**（拟合矩阵的反向：把 `A` 当成源→输出），而 notebook 语义是**源→输出**方向（数组空间 = `A⁻¹`；`303×max(|M₁₁|+|M₁₂|, |M₂₁|+|M₂₂|) = 309.946`，与 `A⁻¹` 同一数值，因为 max span 对轴交换不变）。**实现取 notebook 方向 = 309.9459**，理由：bbox 必须恰好是 NaN 区域的 bbox；用 A 会多出 ~0.9 % 空边、NaN 占比从 5.3 % 升到 6.9 % |
| NaN 占比 | **5.27 %**（几何式 `1 − |det A⁻¹|·(n−1)²/span²`；实测 5.35 %，多出的 0.08 pp 是三阶样条在越界坐标上的边界带）。对照 N=1 的 331² 画布 **16.96 %**：那一版 = 足迹 + `pad=10` 边带 + `⌈extent⌉` 取整，四角与四条 pad 带全是 NaN；N>1 的画布正好等于足迹 bbox，NaN 只剩平行四边形四个角 ⇒ 占比与 N 无关（N=2 与 8 都是 5.3 %）。契约估计的 ~4.6 % 对应「bbox 用像素中心方块、面积用像素边方块」的混用（4.64 %），自洽版本是 5.27 % |
| 比例自洽（N=2 实算） | canvas **608²**、`scale = 607/309.9459 = 1.958406`、`FOV_out = 30.5868 nm`（与 N=8 相同：bbox 语义下 FOV 只由足迹决定）、`Δx_out = 30.5868/608 = 0.0503073 nm/px`、`C = 310`。span / FOV 与 N 无关、只有像素数随 N 变 |
| FFT 家族裁窗（②A） | 在 2432² 细网格上算 FFT（含可选 `--spatial-window`）→ **Z3/C3dc 在细 FFT 上做旋转与平均**（旋转在裁剪前完成，避免裁边效应）→ 结果裁到中央 **C×C**：`C = round(FOV_out/size_nm·n) = round(span) = 310`。`fft_corrected`/`z3`/`c3dc` 存的 npy 与 PNG **都是裁后 310²**，`dq = 2π/FOV_out`；裁剪以 `N//2 − C//2` 起算，DC 仍落在 `C//2` |
| 不变量变化 | **「Z3/C3dc shape == corrected 画布」在 N>1 时不再成立**（310² vs 2432²）——这是显式例外；N=1 时不裁，该不变量仍成立 |
| `--upsample 1` 的回归**范围** | **四个数据族（corrected / fft_corrected / z3 / c3dc 的 csv+npy）+ 全部 PNG + `<out>/affine/fft_window.json` 与已落地 `<DATA>/affine/` 那批逐字节相同**；`<out>/affine/affine_provenance.json` 与已落地那份**键集合、键序、数值都相同**（`upsample`/`nm_per_px`/`fine_canvas_semantics`/`footprint_span_px`/`footprint_extent_px`/`fft_crop_n_px`/`fft_crop_rule` 这 7 个键**只在 N>1 写**）；`pipeline.log` 的 `correct <role> <i>:` 行与既有那句**逐字符相同**（`..., no flip), pad 10, order 3`），N>1 时**另起一行**打 `<N>x fine bbox canvas (n*N, no pad band), NaN … %`——不要笼统写「一切逐字节不变」 |
| 值域 | **不做任何值域操作**：不减平面、不归一化、NaN 不填成均值、不改符号（用户 ③A）。注意 notebook 的 `didv_correction` 里有 `input_data/max(input)` 归一化、`remove_tilt_with_nan_fill` 减平面 + NaN 补均值，本 skill **都不做** |
| 与 `griddata(method="cubic")` 对照 | 同一次坐标变换、同一批目标点上，`affine_transform(order=3)`（B 样条 + prefilter）与 notebook 的 `griddata(method="cubic")`（Clough–Tocher 三角片）**几何完全相同**：线性 ramp 探针 interior median |Δ| = 1.05e-8（值域 1424），而且两者的有限/NaN 掩码**逐像素一致**（0 处不一致）。差异只在核：**本条（含同句 ramp 探针）的 `interior` 口径 = 剔除距源图边界 ≤2 px 的采样带**（实现者 /tmp 探针的定义；不剔除、或改带宽，数值会变——本行测的正是像素级粗糙度那一项，所以必须先声明口径再引用数字）：真实 map interior median |Δ| = **0.041 %**（current）/ **0.22 %**（didv）值域、p99 = 0.43 % / 1.78 %、max = 15.3 % / 13.6 %（单点粗糙处）；把 map 先 σ=1 高斯平滑后 median 掉到 0.006 % / 0.025 % ⇒ 差异来自核作用在**像素级粗糙度**上（didv 的 median|1st diff| = 1.68），不是几何 |
| 色标窗 | 按**裁后** fft_corrected 平面重算（recipe = 父窗那一档候选的百分数对；盘上无可复用档位时退回 index 0 最宽档），写 `<out>/affine/fft_window.json`，打 NOTE。**三种对照口径必须分开读**（都是 hamming 基准的 pooled log\|F\|，父画布 = affine 通路自己的 331² 平面）：
  (i) **真父窗 + ln(N²) vs 出厂重算窗**（idx0 配方）= **−0.6702/+0.5620**（current）、**−0.6576/+0.4550**（didv）⇒「**父窗不可平移复用**」（残差是窗口宽 ~9.7 log 单位的 7 %，足以把大量像素压到一个色标端点）；
  (ii) **同一 percentile index、父画布 331² vs 细画布 310²** = 残差 **0.083–0.299**（idx0：−0.2542/−0.0927（current）、−0.1463/−0.0827（didv）；idx1：−0.2990/−0.1203、−0.2119/−0.1293）⇒ 这一项含 **FOV 变化 32.6645 → 30.5868 nm**（父画布带 pad 带）与 q 网格差；
  (iii) **同 FOV(30.5868 nm)、同配方、只差 N 的 N=2↔N=8** = 位移 **+2.7967/+2.7787**（current）、**+2.7917/+2.7688**（didv）vs ln16 = 2.7726，残差 **≤0.024**（中位数残差 +0.0004/+0.0014）⇒ 这才是「**ln(N²) 刚性平移模型成立**」的严格证据（cross-check：换 idx1 配方时 current 的 p0.1 残差 −0.053，p0.1 对裁剪边缘最敏感）。
  **因果**：裁回原生 q 窗口解决的是 **q 窗口对齐**（同一 FOV、同一 q 网格、把像素数归一），(ii) 里剩下的 0.08–0.30 来自**视场/网格变化**，而 (i) 的 0.46–0.67 里还额外含「**父窗本身取自另一块画布**（LF 332²，带 pad 带）」这一项 |
| 只对 affine 通路 | LF 包 + `N>1` 非零退出；`N` 必须是 ≥1 的整数（0/负 → 非零退出；非整数由 argparse 拦，退出码 2）；`N>1` 与 `--window-candidates`/`--pick-window` 同用 → 非零退出；**affine 包是 `fallback:true` / `method:'identity_fallback'` 时 `N>1` 也非零退出**（没有任何变换 ⇒ 没有足迹可铺，n·N 画布无从填充；同一个包用 N=1 仍照旧跑并落 304² 的拷贝），退出发生在创建任何输出目录之前 |
| 三个读数字的坑 | ① **画布像素数 ≠ 网格细化倍数**：2432² = 5 914 624 px 是 304² 的 **64×**，但 Δx 只从 0.0986842 细到 **0.0125768 nm/px = 7.8465×**（bbox 画布 FOV 30.5868 nm ≠ N=1 画布 32.6645 nm）；`scale = (n·N−1)/span = 7.843304`（画布 px / 粗像素）是**第三个量**。判据：说「细了多少」就用 Δx 比。
  ② **偶数 C 的裁窗留下非 Hermitian 边行边列**：C=310 ⇒ 保留 k ∈ [−155, +154]，k=−155 的镜像伙伴 k=+155 不在窗内（**619 个边像素** = 310+310−1，占 310² 的 0.64 %），存盘平面只在 k ≥ −154 上满足 Hermitian（正确的 mod-n 配对残差 ~3e-17，机器精度量级）。**量度（写清定义才可用）**：对存盘裁后平面取 `np.fft.ifft2(np.fft.ifftshift(F))` 的 `rms(imag)/rms(real)` = **6.29e-5（current, hamming）/ 6.90e-4（didv, hamming）**（无窗套为 1.08e-4 / 7.30e-4——换定义/换窗都会变，所以这个数只在「hamming + 该定义」下引用）。判据：别把存盘的 `fft_corrected` 当成「实信号谱、可直接反变换得实图」；Z3/C3dc 是**先在全细谱上旋转再裁自己的输出**，不受这条影响。
  ③ **order=3 在 NaN 边界会振铃**：corrected 极值会略微超出原图值域。**判据（逐张）**：`(nanmax(corr) − nanmax(raw))/range` 与 `(nanmin(raw) − nanmin(corr))/range`，`range = nanmax(raw) − nanmin(raw)`，>0 记「超」。**全量 20 张实测**（raw = h5 抽出的 fwd/bwd nanmean，corr = 出厂 corrected；N=1 画布 331² / N=8 画布 2432²）：**N=1 有 7/20 张超**（max 侧 2/20，最大 **+6.09 %** 在 current 3_73meV；min 侧 6/20，最大 **+4.61 %**），**N=8 有 20/20 张超**（max 侧 17/20，最大 **+8.68 %**；min 侧 18/20，最大 **+9.83 %**，两者都在 current 3_73meV）⇒ **是否超极值是数据/画布相关的、不是 N>1 独有**。判据：判「值域不重缩放」用 `nanmean`/中位数——20 张的 `|Δnanmean|/|nanmean|` **≤1.2e-4**（0_50meV 那四例是 9.2e-6…5.9e-5），**不要用 max/min** |
| 体积（本例 2 角色 × 10 bias） | corrected CSV 2432² **≈88 MB/张**（20 张 ≈1.8 GB，是目前最大头）、corrected PNG 2432² ≈5 MB/张、`fft_corrected`/`z3` 各 40 个 310² complex128 ≈1.5 MB/文件（各 ≈60 MB）、`c3dc` 40 个 310² float64（≈31 MB）⇒ **合计 ≈2 GB**（比不裁窗小一个量级）。**画布 2432² 与 FFT 家族 310² 是两个画布**，别按同一个边长估体积 |

**机制：细网格是同信息量的插值产物**。n·N 画布是那一条 304² 数据的**插值像**（固定核的一次 cubic），信息量与 304² 相同、没有新数据进来；这正是 ②A 要裁回原生 q 窗口的原因（窗口外只剩插值/加窗的内容）。所以 `--upsample N>1` 的收益**不在 corrected 图本身的"精度"**（把 flag 读成"矫正得更准"是误读），而在 **FFT/Z3/C3dc**：q 采样变密（dq = 2π/FOV_out）与 120°/240° 旋转的插值误差随之变小。把 N>1 的 corrected 与 N=1 逐点比较，得到的是"更细但同信息"的插值/视场差（两者画布尺寸与 FOV 都不同），不能当作精度提升的证据。

**细画布上的 DC 旋转**：`rotate_about_dc()` 的不动点是浮点中心 `c = (N/2, N/2)`；`fftshift` 后的 DC 像素在**偶数**画布上位于 `N/2`，两者重合。N>1 的 affine 画布是 `n·N = 2432²`（偶数，DC index 1216），所以旋转恰好绕 DC、`DC_retained`/`R120 keeps` ≈ 1.0000（实测见报告）。N=1 的画布是 `2k+1 = 331`（奇数，DC index 165 ≠ 165.5），`R120 keeps` 实测 0.086–0.090 —— 那是**既有行为**（与 `--upsample` 无关），本 skill 不改 Z3/C3dc 的数学，只如实说明：§2.4 的 d 里「应为 1.0000」只对偶数画布成立。


## 3 产物布局

数据与图同目录、每个量一个目录（`<win>` = 空间窗名，只在 `--spatial-window` ≠ none 时出现；`<i>` = bias 索引，`<bias>meV` 从 `/bias` 读并 ×1000，如 `0_30meV`、`3_-5meV`）：

```
<OUT>/raw/<role>/<i>_<bias>meV.csv|.png            # PNG: 1:1, viridis, 无 colorbar/轴
<OUT>/fft_raw/<role>/<i>_<bias>meV_fft2.npy|_fft.png        # 无窗；PNG 1:1, plasma, 无 colorbar/轴
<OUT>/fft_raw/<role>/<i>_<bias>meV_fft2_<win>.npy|_fft_<win>.png        # 加窗（§2.5）
<OUT>/corrected/<role>/<i>_<bias>meV.csv|.png        # PNG: 1:1, viridis
<OUT>/fft_corrected/<role>/<i>_<bias>meV_fft2.npy|_fft.png              # 无窗, plasma
<OUT>/fft_corrected/<role>/<i>_<bias>meV_fft2_<win>.npy|_fft_<win>.png  # 加窗
<OUT>/z3/<role>/<i>_<bias>meV_fft2.npy|_fft.png      # 无窗, plasma
<OUT>/z3/<role>/<i>_<bias>meV_fft2_<win>.npy|_fft_<win>.png            # 加窗
<OUT>/c3dc/<role>/<i>_<bias>meV_fft2.npy|_fft.png    # 无窗, plasma
<OUT>/c3dc/<role>/<i>_<bias>meV_fft2_<win>.npy|_fft_<win>.png          # 加窗
<OUT>/window_candidates/<role>/window_KK.png|contact_sheet.png  # 每角色一套候选（第一次运行）
<OUT>/fft_window.json            # {"window_version": 2, "roles": {<role>: {vmin, vmax}},
                                 #  "spatial_window": {name, suffix, taps, files}}   # §2.5
<OUT>/fft_window_candidates.json # 缓存候选表（供 --pick-window 复用）
<OUT>/pipeline.log               # 逐步关键数值：raw 窗口、FFT 窗口、画布增长、
                                 # Z3/C3 的 DC 保留、三重残差输入/输出、能量比、
                                 # c3dc 负值个数与相对幅度
```

**affine 通路**（`--transform` 是 affine 包时）由代码把所有产物前缀成 `<out>/affine/`（调用方仍传 `-o <out>`），
**只落 c–e 阶段**——目录名与 LF 通路**同类同名**，只是根不同：

```
<OUT>/affine/corrected/<role>/<i>_<bias>meV.csv|.png        # 1:1, viridis
<OUT>/affine/fft_corrected/<role>/<i>_<bias>meV_fft2.npy|_fft.png          # 无窗, plasma
<OUT>/affine/fft_corrected/<role>/<i>_<bias>meV_fft2_<win>.npy|_fft_<win>.png
<OUT>/affine/z3/<role>/<i>_<bias>meV_fft2.npy|_fft.png      # 无窗 + 加窗
<OUT>/affine/c3dc/<role>/<i>_<bias>meV_fft2.npy|_fft.png    # 无窗 + 加窗
<OUT>/affine/fft_window.json     # 本次选用的 per-role 色标窗（继承来的也写回，§2.4）
<OUT>/affine/pipeline.log        # 同一套逐步数值 + 一段 affine provenance
<OUT>/affine/affine_provenance.json  # 机器可读 provenance（字段与 log 一致，§2.4）
```

**`--upsample N>1` 时有两套画布（别按同一边长估体积）**：`corrected/` 是 **n·N × n·N 细画布**（本例 2432²），
而 `fft_corrected/`、`z3/`、`c3dc/` 是**裁回原生 q 窗口的 C × C**（本例 310²）——"Z3/C3dc shape == corrected 画布"这条
不变量在 N>1 时**不成立**（N=1 时成立，且逐字节保留）。实测体积（本例 2 角色 × 10 bias）：corrected CSV 2432²
≈88 MB/张（20 张 ≈1.8 GB，最大头）、corrected PNG 2432² ≈5 MB/张、`fft_corrected`/`z3` 各 40 个 310² complex128
≈1.5 MB/文件、`c3dc` 40 个 310² float64 ≈0.8 MB/文件 ⇒ **合计 ≈2 GB**（比"全链停在 2432²"的小一个量级）。

**affine 通路不写**：`<out>/affine/raw/`、`<out>/affine/fft_raw/`、`<out>/affine/window_candidates/`
（既没有 b 阶段，也就没有候选表可建）。同一次 `<out>` 里的 LF 产物（`raw/ fft_raw/ corrected/ fft_corrected/
z3/ c3dc/ window_candidates/ fft_window.json fft_window_candidates.json pipeline.log`）**一个字节都不动**——
affine 通路只往 `<out>/affine/` 里写。

## 4 与 Skill A / lawler-fujita-correction 的分工

| 项 | `grid-lf-pipeline` (Skill A) | `lawler-fujita-correction` | `affine-correction` | 本 skill |
| --- | --- | --- | --- | --- |
| 职责 | 3ds→h5、抽 topo（三步预处理）、LF 拟合并导出位移场包 | 拟合位移场 + 重采样实现 | 拟合全局仿射拉伸 M 并导出变换包（+ 自己的 CSV 帧 apply 段） | 列出 channel、按名字抽角色图、把**包**只重采样地套到每张 map、复数 FFT、Z3 / C3 |
| 依赖方向 | 嵌套调用 `stm_lf_correct.py` | 独立 | 独立 | **按相对路径** `skills/lawler-fujita-correction/scripts/lf_lib.py` 嵌套调用其 `warp_by_field`；**只抄** affine skill 的 `_image_transform` 15 行做本地镜像（`grid_sym.py:image_transform()`，一致性见 §8 第 11 条） |
| 是否修改对方 | 否 | — | — | **否**。不读不写上面三个 skill 下任何文件；affine 的几何是镜像，不是 import |

**包从哪来**：① LF 位移场包 —— 先跑 Skill A `grid_lf_fit.py <stem>_topo.csv -L <nm> -o <out>/lf`，得到
`<out>/lf/<stem>_transform.json + .h5`；② affine 变换包 —— 先跑 `affine-correction` 的
`stm_topo_correct.py <stem>_topo.csv -L <nm> -o <out>/affine --save-transform <out>/affine/transform.json`。
把该目录（或该 json 文件）交给本 skill 的 `--transform`；传目录时 `<out>/lf/` 与 `<out>/affine/` 各自只有一个候选，不会歧义。

**为什么是「只重采样」**：本 skill 的 c 阶段只做重采样，从不减平面/去均值/归一化，也不翻转（0 次 flipud，见 §2.2）；
仓库根 `scripts/apply_lf_transform.py` 与两个 apply 工具（`stm_lf_apply.py` / `stm_apply_transform.py`）面向的是
**CSV 帧**且都会减拟合平面，那两个 flipud 与那次去平面**不适用于本 skill**（本 skill 直接按名字从 `/data` 取名，不经过 topo CSV）。
实测对 +100 pA / 线性斜坡输入输出差仅 ~1e-22。本 skill 矫正后 map 的值域与原图一致（例如 current 通道仍 ~5e-10 A 电平）。

## 5 两次运行交互协议

1. **跑 `--window-candidates N`** → 出 a+b + **每个角色** N 张候选 + 该角色的 contact sheet + 按角色分组的候选表，缓存到 `fft_window_candidates.json`，**exit 0**。**候选模式无条件重算**：目录里已有 `fft_window.json`（或旧候选表）也照跑候选分支（2026-10 修——之前会被「复用已保存窗」静默压过，整条命令变成 no-op、exit 0、什么都不写），覆盖旧候选表时打一行 `# NOTE: … is recomputed and overwritten by --window-candidates N (basis --spatial-window …)`，并另打一行说明 `fft_window.json` 不被本次运行触碰（由下一次 `--pick-window`/`--fft-window*` 覆盖）。
2. **用户从候选表挑编号 K** → **跑 `--pick-window K`**（每个角色各取自己第 K 档），或按角色给显式窗口 `--fft-window-role ROLE VMIN VMAX`（可重复）→ 写 `fft_window.json`，出全套 a+b+c+d+e。
3. **优先级：候选模式 > 显式命令行 > 盘上的 `fft_window.json`**（2026-10 修，之前是文件优先，导致 `--pick-window` 与 `--window-candidates` 双双被静默忽略）：**候选模式 `--window-candidates N` 排在最前**——它是一次「重做候选」的请求，盘上已有的选择不能拦它；而它与三条显式窗入口**互斥**（同时给即 fail-closed，见 §6），因为候选运行出表后就 exit，与「跑全套」是两件事。
   - 给任一条显式入口（`--pick-window K` / `--fft-window VMIN VMAX` / `--fft-window-role ROLE VMIN VMAX`）→ **忽略并覆盖**盘上的文件，写回新选择，并打印/记录一行
     `# NOTE: <outdir>/fft_window.json existed with <旧值>; overridden by <入口> (the file is rewritten with this selection)` 到 stdout 与 `pipeline.log`；
   - 不给显式入口 + 盘上有文件 → 直接复用（幂等），全套照跑（旧值原样保留，不打印 NOTE）；
   - 不给显式入口 + 盘上也没有 → fail-closed（非零退出，见 §6）。
   - 多条显式入口同时给时，**顺序与以前一致**：`--pick-window` 赢，`--fft-window-role` 覆盖 `--fft-window`（NOTE 里会写明哪个入口生效、哪些被忽略）。
4. **角色单位不同（current 与 didv 差 ~26 个 log）时，同一对 vmin/vmax 不可能同时适合两个角色**——这正是 `--pick-window` 按角色取、`--fft-window-role` 按角色给的原因。
5. **affine 通路没有候选这一步**：它只写 c–e，没有 `fft_raw` 平面可作候选基准，所以 `--window-candidates` / `--pick-window` 在该通路 fail-closed（§6）。它的窗口是**继承** `<out>/fft_window.json`（= 同一次 LF 运行选定的 per-role 窗口）并写回 `<out>/affine/fft_window.json`；要换窗口就给 `--fft-window-role ROLE VMIN VMAX`（可重复）或 `--fft-window VMIN VMAX`。

## 6 fail-closed（非零退出且不写半成品）

| 情形 | 行为 |
| --- | --- |
| 角色通道名匹配不到 / 匹配到多个 | 列出全部可用名字，退出非零，不猜 |
| 给的是 `[bwd]` 名字 | 拒绝（fwd 名字是规范入口），退出非零 |
| `--transform` 缺失 / 目录里不唯一（0 或 ≥2 个候选：`*_transform.json` 与 `transform.json`） | 报错（歧义时列出候选文件名），退出非零 |
| `--transform` 的包**既像 affine 又像 LF**（schema `topo-correction-transform` + 合法 `affine_q` **且** 含 `u_field_file`） | 报错，退出非零（不猜该跑哪条通路） |
| `--transform` 的包**两条都不像**（schema 不对 / `affine_q` 非有限 2×2 或 `det ≤ 0`，且无 `u_field_file`） | 报错并打两个理由，退出非零。`fallback` 键**不参与**判别 |
| affine 通路 + `--window-candidates N` | 报错，退出非零，消息指向「继承的窗口 / 显式 `--fft-window-role`」；**在创建任何目录之前**就退出（不留半成品目录） |
| LF 通路（`--transform` 是位移场包）+ `--upsample N>1` | 报错，退出非零（LF 只有 `warp_by_field` 一次重采样，没有 n·N 细网格语义）；**在创建任何目录之前**退出 |
| `--upsample N` 的 `N` 不是 ≥1 的整数（0 / 负数） | 报错，退出非零；非整数（如 `2.5`、`abc`）由 argparse 拦下，退出码 2 |
| affine 通路 + `--upsample N>1` + `--window-candidates`/`--pick-window` | 报错，退出非零（该口径自行按裁后 `fft_corrected` 平面重算窗，且 affine 通路本就没有候选表） |
| affine 包是 `fallback:true` / `method='identity_fallback'` + `--upsample N>1` | 报错，退出非零（「没有变换 ⇒ 没有足迹可铺，n·N 画布无从填充」，提示改用 N=1 或先拟合真变换）；**在创建任何输出目录之前**退出（负对照实测：输出目录不存在）。同一个包 + N=1 仍退出 0、落 304² 拷贝 |
| `--upsample N>1` 算出的原生 q 窗口 C 不小于细画布边长 | 报错，退出非零（拒绝裁一个并不比窗口宽的谱） |
| affine 通路 + `--pick-window K` | 报错，退出非零（该通路没有候选表，`K` 永远无从解析） |
| affine 通路且自身 `<out>/affine/fft_window.json` 与父目录 `<out>/fft_window.json` 都不存在，又没给显式窗口 | 报错，退出非零，提示先跑 LF 通路选窗、或给 `--fft-window-role` |
| h5 缺 `/params` 的 `Z (m)` 列 | 报错，退出非零 |
| 尚未选窗却要求出全套（无 `fft_window.json`，又未给 `--pick-window`/`--fft-window`/`--fft-window-role`） | 提示先跑 `--window-candidates`，退出非零 |
| 盘上有 `fft_window.json` **且** 给了显式入口 | 不报错：覆盖并打印 `# NOTE: ... overridden by <入口>`（§5 第 3 条） |
| `--window-candidates N` 与 `--pick-window`/`--fft-window`/`--fft-window-role` 同时给 | 报错退出非零（候选运行只出表就 exit，与「跑全套」矛盾；先跑候选再 pick） |
| `--pick-window K` 但候选表不存在 / K 越界 | 报错，退出非零 |
| `--pick-window K` 但候选表是**池化时代**的旧格式（无 `roles` 键） | 报错并提示重跑 `--window-candidates N`（旧窗口对每个角色都是错的），退出非零 |
| `--fft-window-role ROLE ...` 的 ROLE 不在本次 `--map` 里 | 报错并列出本次角色，退出非零 |
| 给了 `--fft-window-role` 但没覆盖全部角色 | 报错并列出缺的角色，退出非零 |
| `--fft-window VMIN VMAX` 或 `--fft-window-role ... VMIN VMAX` 且 VMIN ≥ VMAX | 报错，退出非零 |
| `fft_window.json` 缺某个角色的窗口 | 报错，退出非零 |
| h5 视场与位移场包的 `field_of_view_nm_reference` 不一致 | 按 `abs(size_nm - reference_fov) <= 1e-9 * max(1, reference_fov)` 判定；不一致时用 `lf_lib.resample_field` 把 u 场重采样到本图网格（不是报错） |

**落盘纪律**：失败时只清理**本次运行写出**的东西，不动用户原有的文件；Skill A 的 `grid_lf_fit.py` 同理（进入时先记下 `lf/` 已有内容，失败只删本次新增项，其余保留）。

## 7 已知坑

1. **bias 标签只能从 `/bias` 读**（单位 V，需 ×1000 得 meV），不得写死；文件名 `<i>_<bias>meV` 中的 `<i>` 是 bias 索引。
2. **旋转必须绕 DC 像素**：`fftshift` 后偶数 N 的 DC 在索引 `N/2`（不是数组几何中心 `(N-1)/2`）。用 `scipy.ndimage.affine_transform(matrix=R, offset=c-R c, order=3, mode='nearest')`，`c=(N/2, N/2)`（浮点）；这样 DC 保留到 ~1e-16。用 `ndimage.rotate`（绕 `(N-1)/2`）会把中心尖峰糊掉。
3. **c3dc 的 log 必须用 `log|C3dc|`**：三次插值振铃给出少量负值（本例 8–16 像素/图、最大负值幅度约为 DC 峰值的 0.07、负值功率占全谱 ~1.7%），负值个数与相对幅度要写进 log。
4. **log 里的 `DC_retained 1.0000` 是无信息装饰项**：旋转按 `offset = c - R c` 构造，恒有 `output(c) = input(c)`（`mode='nearest'` 下越界采样也不涉及 DC），所以这个比值在构造上恒等于 1，不管对称化是否有效。判断 Z3/C3 是否真的生效请看 `residual_in -> residual_out` 与 `energy_ratio`；不要把这个 1.0000 当证据引用。
5. **未经选窗的 FFT 图窗口是临时的**：candidates 运行给 fft_raw 用**该角色** p50–p99.9 的临时窗；用户选窗后全套重画，每个角色的 FFT 图用该角色选定的窗口（画法不变，只换窗口）。
6. **候选图必须同画布尺寸**：固定 figure 尺寸 + 不 `bbox_inches='tight'`，保证逐像素可比；PIL 断言全部一致后才拼 contact sheet（图注拆短行，防 tight bbox 撑大画布）。
7. **色标窗口绝不能跨角色池化**（2026-10 实测的真 bug，用户看到 6 张 contact sheet 全是纯色）：`current` 的 log|F| 在 −27…−8.7（中位 −19.95），`didv` 在 −1.9…+16.3（中位 +6.27），池化后分位数落在 −21…+10，于是每张候选对 didv 是整幅超 vmax、对 current 是整幅低于 vmin。修前用 PIL 统计候选图坐标区：**端点（饱和）像素 59.9–98.7 %、窗口中间 80 % 区间内 ~0 %**；按角色分开后为 **0.002–1.83 % / 73.6–99.7 %**。判据与脚本：对每张候选图取坐标区（matplotlib 同一 figure 几何的 `ax.get_window_extent()`）统计「颜色等于 **`FFT_CMAP`** 两端点」与「落在窗口中间 80 %」的像素占比，前者必须 < 5 %、后者必须 ≥ 50 %（候选图也走 `FFT_CMAP`；本 skill 的数据图同样用它，见 §2.3）。

8. **数据图无 colorbar、1:1、pad=0**（2026-10 用户要求，见 §2.3）：`raw/ corrected/ fft_raw/ fft_corrected/ z3/ c3dc/` 的 PNG **没有** colorbar/坐标轴/标题/留白，**边长恰好等于数组边长**（每元素 1 像素），窗口仍是调用方选的 vmin/vmax。`window_candidates/` 与 `contact_sheet.png` 是**例外**（选窗工具，保留标注与 colorbar）。判据：把 PNG 从「数组 + 窗口 + cmap」重算出来逐像素比（`save_array_image` 的映射：`LUT(int(clip((v-vmin)/(vmax-vmin))*256))`，`origin="lower"`，NaN→`#b0b0b0`），数据像素必须逐点 ≤1/255 吻合——这条同时排除了 colorbar（任何色条/轴都会占像素、破坏 1:1 映射）。

9. **两套的端点压死比例会差一个量级，而色标窗只对其中一套对口径**（2026-10 实测，基准已改为跟随空间窗）：候选按当前空间窗生成后，色标窗对**加窗**套对口径；**无窗（pure）套**用同一个窗时会明显更紧或更松——本例 `current` 首个 bias 两套相差 +9.2 pp（无窗 0.520 %、加窗 9.749 %）。工具对每族每偏压打一行 `fft window clip ...`（含两套数字与总差），>0.5 pp 再打一行 `WARNING fft window ...` 指明哪一套离基准。处置：**只看自己要在图上用的那一套**——要 pure 图就别改窗，要加窗图就用 `plane_<win>%` 那一列挑窗（§2.5）。

10. **affine 包不是 LF 包，别互相代用**（2026-10 新增）：affine 包只描述**一个全局拉伸 M**（+画布几何），没有 u 场；
    想"把它塞进 LF 包当线性位移场"是伪造溯源（`method` 会被记错、`u_field_file` 会指向不存在的场），
    本 skill 的判别正是为此关掉的（同时像两者 → 非零退出）。affine 通路的 `anchor_verdict == "unverifiable"`
    （本数据）是**拟合段**的结论：affine 通路不重跑锚定自检，provenance 里显式写 "anchor self-check NOT re-run"，
    读结果时不要把"跑完了"当成"验证过了"。

## 8 自检口径

```bash
# 1) 无通道索引字面量
cd /path/to/STM_DataProcessing && ! grep -rEn '\[\s*:\s*,\s*:\s*,\s*[0-9]+\s*,\s*:\s*\]' skills/grid-sym-fft

# 2) grid_channels 非交互模式
MPLCONFIGDIR=<writable> PYTHONDONTWRITEBYTECODE=1 .venv/bin/python skills/grid-sym-fft/scripts/grid_channels.py \
    GRID.h5 --map current="Current (A)" --map didv="DSP 7280 Y (%)"

# 3) 先出候选（a+b + 每角色候选 + contact sheet），exit 0；stdout 存下来供第 3b 步查 warning
mkdir -p /tmp/out          # the redirect target must exist before the shell opens it
MPLCONFIGDIR=<writable> PYTHONDONTWRITEBYTECODE=1 .venv/bin/python skills/grid-sym-fft/scripts/grid_sym.py \
    GRID.h5 --map current="Current (A)" --map didv="DSP 7280 Y (%)" \
    --transform OUT/SkillA/lf -o /tmp/out --window-candidates 9 \
    > /tmp/out/candidates_stdout.log 2>&1
echo "candidates exit=$?"     # must be 0

# 3b) 候选按角色分开落地；表里两套口径（pooled / plane）齐全；超容差的档必须
#     同时有表格里的 ! 与一行 WARNING（不静默），未超容差的档不得有 WARNING
(cd /tmp/out \
    && test "$(ls window_candidates/current/window_*.png | wc -l)" -eq 9 \
    && test "$(ls window_candidates/didv/window_*.png | wc -l)" -eq 9 \
    && test -f window_candidates/current/contact_sheet.png \
    && test -f window_candidates/didv/contact_sheet.png \
    && echo "9+9 figures + 2 sheets ok")
.venv/bin/python - /tmp/out/fft_window_candidates.json /tmp/out/candidates_stdout.log <<'EOF'
import json, sys
t = json.load(open(sys.argv[1]))
log = open(sys.argv[2]).read()
assert set(t["roles"]) == {"current", "didv"}, list(t["roles"])
c = [x["vmin"] for x in t["roles"]["current"]["candidates"]]
d = [x["vmin"] for x in t["roles"]["didv"]["candidates"]]
assert max(c) < min(d), (c, d)          # the two roles must not share a value range
warned = 0
for role, r in t["roles"].items():
    tol = r["plane_clip_tolerance_pct"]
    for x in r["candidates"]:
        pooled = x["clip_below_pct_pooled"] + x["clip_above_pct_pooled"]
        plane = x["clip_below_pct_plane"] + x["clip_above_pct_plane"]
        assert abs(x["clip_total_pct_plane"] - plane) < 1e-9, x
        assert pooled < 5.0, x          # the design budget keeps every candidate viewable
        over = x["plane_over_budget_pct"] > tol
        line = f"WARNING candidates {role} window {x['index']:02d}:"
        assert over == (line in log), (role, x["index"], over)   # never silent, never spurious
        warned += over
print(f"per-role candidate tables ok (both calibers; {warned} over-tolerance warning(s))")
EOF

# 4) 挑候选 K 出全套（每个角色各取自己的第 K 档）
MPLCONFIGDIR=<writable> PYTHONDONTWRITEBYTECODE=1 .venv/bin/python skills/grid-sym-fft/scripts/grid_sym.py \
    GRID.h5 --map current="Current (A)" --map didv="DSP 7280 Y (%)" \
    --transform OUT/SkillA/lf -o /tmp/out --pick-window 3

# 5) 产物计数（每个角色每偏压一个 npy）+ 窗口是按角色存的
(cd /tmp/out \
    && test -f fft_window.json \
    && test "$(ls z3/didv/*_fft2.npy | wc -l)" -eq 5 \
    && test "$(ls c3dc/didv/*_fft2.npy | wc -l)" -eq 5 \
    && test "$(ls fft_corrected/didv/*_fft2.npy | wc -l)" -eq 5 \
    && echo "5 npy per FFT family ok")
.venv/bin/python - /tmp/out/fft_window.json <<'EOF'
import json, sys
w = json.load(open(sys.argv[1]))
assert w["window_version"] == 2 and set(w["roles"]) == {"current", "didv"}, w
c, d = w["roles"]["current"], w["roles"]["didv"]
assert c["vmax"] < d["vmin"], (c, d)    # the roles must not share a window
print("per-role windows ok")
EOF

# 6) 候选图可见性（PIL，绝不靠肉眼）：每张候选图在坐标区内
#    端点（饱和）像素 < 5 %，且落在窗口中间 80 % 区间内的像素 ≥ 50 %
#    （坐标区取 matplotlib 同一 figure 几何的 ax.get_window_extent()）

# 7) 坐标系（标记法，不看图）：同一物理标记在【LF 与 affine 两条通路】的产物里都必须落在
#    **输入帧（h5 帧）的同一行**（≤3 px）：
#      LF    : gs.apply_transform(blob)（h5 帧，0 次 flip）
#              vs stm_lf_apply.py 的 corrected CSV（该工具按 CSV 帧进出 ⇒ 先 flipud 回 h5 帧再比）
#      affine: gs.apply_affine(blob, geometry)（h5 帧，0 次 flip）
#              vs stm_apply_transform.py 的 corrected CSV（同上，flipud 回 h5 帧再比）
#    两条都带**反例判据**：同一比法若漏掉那次折算（即拿未翻转的产物直接比），必须偏出纯镜像量
#    n_out-1-2r；偏不出来说明门在空转。（记账：2026-10-14 t1 之后两个 apply 工具的产物都是
#    「与输入 CSV 同帧」，所以这里比的是 flipud(产物)，不是产物本身。）
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - GRID.h5 OUT/SkillA/lf OUT/affine/transform.json /tmp/out_frame 304 <<'EOF'
import subprocess, sys
from pathlib import Path
import numpy as np
h5 = Path(sys.argv[1])
lf_dir = Path(sys.argv[2])
aff_json = Path(sys.argv[3])
work = Path(sys.argv[4])
n = int(sys.argv[5])
work.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, "skills/grid-sym-fft/scripts")
import grid_sym as gs

ROW, COL = 20, 152
rows, cols = np.mgrid[:n, :n]
blob = np.exp(-(((rows - ROW) ** 2 + (cols - COL) ** 2) / (2 * 2.0 ** 2)))
csv = work / "marker.csv"
np.savetxt(csv, np.flipud(blob), delimiter=",", fmt="%.10e")   # the CSV frame both tools expect


def cen(img):                                # sub-pixel marker row (zeros background)
    v = np.where(np.isfinite(img), img, 0.0)
    w = np.clip(v - 0.05 * v.max(), 0.0, None)
    rr = np.mgrid[: v.shape[0], : v.shape[1]][0]
    return float((rr * w).sum() / w.sum())


def check(name, mine, product, n_out):
    theirs = np.flipud(np.loadtxt(product, delimiter=","))     # back into the h5 frame
    a, b = cen(mine), cen(theirs)
    mirror = n_out - 1 - b                                     # what a missing flip would give
    print(f"{name}: grid_sym {a:.3f}  tool {b:.3f} (both h5 frame)  |delta| = {abs(a - b):.3f} px ; "
          f"without the flip it would land at {mirror:.3f} px ({abs(a - mirror):.3f} px off)")
    assert abs(a - b) <= 3, f"{name}: frames disagree"
    assert abs(a - mirror) > 3, f"{name}: gate is vacuous (mirror amount too small)"


# LF: displacement field, h5 frame in / h5 frame out
lf_bundle = sorted(lf_dir.glob("*_transform.json"))[0]
payload, u_nm, valid, _ = gs.load_transform(gs.resolve_transform(str(lf_bundle)))
size_nm = gs._scan_size_nm(h5)
a_lf, n_out_lf, half, _ = gs.apply_transform(blob, size_nm, payload, u_nm, valid)
subprocess.run([sys.executable, "skills/lawler-fujita-correction/scripts/stm_lf_apply.py",
                str(csv), "--transform", str(lf_bundle), "-L", f"{size_nm:g}",
                "-o", str(work / "lf")], check=True)
check("LF", a_lf, work / "lf" / "marker_corrected.csv", n_out_lf)

# affine: A = P M^-1 P, h5 frame in / h5 frame out, 0 flips
probe = gs.read_json_object(aff_json)
geometry = gs.affine_geometry(probe, aff_json)
a_aff, n_out_aff = gs.apply_affine(blob, geometry)
subprocess.run([sys.executable, "skills/affine-correction/scripts/stm_apply_transform.py",
                str(csv), "--transform", str(aff_json), "-L", f"{size_nm:g}",
                "-o", str(work / "affine")], check=True)
check("affine", a_aff, work / "affine" / "marker_corrected.csv", n_out_aff)
print("frame check ok: both branches write the h5 frame with 0 flips")
EOF

# 8) 画法（1:1 + 无 colorbar + 无轴）：把 PNG 从「数组 + 窗口 + cmap」重算出来逐像素比。
#    数据像素必须全部 ≤1/255 吻合；PNG 边长必须等于数组边长；四边 2 px 内非白比例 ≥0.99。
#    任何 colorbar/坐标轴/留白都会占像素、破坏这个 1:1 映射，所以这一条同时是"无 colorbar"的判据。
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - /tmp/out <<'EOF'
import sys, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from PIL import Image
sys.path.insert(0, "skills/grid-sym-fft/scripts")
import grid_sym as gs
run = Path(sys.argv[1])
win = json.loads((run / "fft_window.json").read_text())["roles"]
rgb = lambda c: np.asarray(plt.get_cmap(c)(np.arange(256) / 256.0)[:, :3]) * 255.0
BAD = np.array(matplotlib.colors.to_rgb(gs.lf_lib.BAD_COLOR)) * 255.0
checked = 0
for family, role in [(f, r) for f in ("raw", "corrected", "fft_raw", "fft_corrected", "z3", "c3dc")
                     for r in ("current", "didv")]:
    for png in sorted((run / family / role).glob("*.png")):
        stem = png.stem
        if family in ("raw", "corrected"):
            mat = np.loadtxt(png.with_suffix(".csv"), delimiter=",")
            med, std = np.nanmedian(mat), np.nanstd(mat)
            vmin, vmax, cmap = med - 5 * std, med + 5 * std, gs.MAP_CMAP
        else:
            mat = np.log(np.abs(np.load(png.with_name(stem.replace("_fft", "") + "_fft2.npy"))))
            vmin, vmax, cmap = win[role]["vmin"], win[role]["vmax"], gs.FFT_CMAP
        bad = ~np.isfinite(mat)
        idx = np.clip(((np.clip((np.nan_to_num(mat) - vmin) / (vmax - vmin), 0, 1)) * 256).astype(int), 0, 255)
        want = rgb(cmap)[idx]; want[bad] = BAD; want = np.flipud(want)
        got = np.asarray(Image.open(png).convert("RGB")).astype(float)
        assert got.shape == want.shape, (png, got.shape, want.shape)
        d = np.abs(got - want).max(axis=2)
        assert np.all(d[~np.flipud(bad)] <= 1.0), (png, float(d[~np.flipud(bad)].max()))
        edge = np.concatenate([got[:2].reshape(-1, 3), got[-2:].reshape(-1, 3),
                               got[:, :2].reshape(-1, 3), got[:, -2:].reshape(-1, 3)])
        assert np.mean(edge.sum(axis=1) < 720.0) >= 0.99, (png, "edge")
        checked += 1
print(f"render check ok on {checked} PNGs (1:1, no colour bar, window preserved)")
EOF
```
# 9) 空间窗（自己写 numpy 复算，不许调实现里的函数）：产出数组必须等于
#    fftshift(fft2(fill(map) * outer(w, w)))，w = np.hamming / np.hanning(n)。
#    用 h5 的 /data 复算 fft_raw 时应当**逐位相同**（0.0）；用 corrected CSV 复算时，
#    残差只能来自 CSV 的 %.10e 量化——要给出解析上界 N^2 * 0.5e-11 * max|map| 并核对。
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - /tmp/out hamming hamming <<'EOF'
import json, sys
from pathlib import Path
import h5py, numpy as np
run, h5, win = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
fill = lambda x: np.where(np.isnan(x), np.nanmedian(x), x)
def mine(x):
    a = fill(np.asarray(x, float))
    w = {"hamming": np.hamming, "hanning": np.hanning}[win](a.shape[0])
    return np.fft.fftshift(np.fft.fft2(a * np.outer(w, w)))
with h5py.File(h5, "r") as fh:
    names = [c.decode() for c in fh["channels"][:]]
    data = fh["data"][:]
for role, fwd, bwd in (("current", "Current (A)", "Current [bwd] (A)"),
                       ("didv", "DSP 7280 Y (%)", "DSP 7280 Y [bwd] (%)")):
    # float64 first: the maps are float32 on disk, and averaging in float32
    # would leave a ~5e-10 relative residue that is not a window error
    m = np.nanmean(np.stack([data[:, :, names.index(fwd), 0].astype(float),
                             data[:, :, names.index(bwd), 0].astype(float)]), axis=0)
    got = np.load(run / "fft_raw" / role / f"0_50meV_fft2_{win}.npy")
    rel = np.abs(mine(m) - got).max() / np.abs(got).max()
    print(f"fft_raw {role}: |mine - {win}| / max|F| = {rel:.3e}")
    assert rel == 0.0, "windowed FFT must reproduce bit for bit from the h5 map"
print("spatial window check ok")
EOF
```
# 10) 候选基准必须按【值】自证（不要拿实现自己读的那批数组复算——那是同源自证）
#     判据：fft_window_candidates.json 顶层 candidate_basis == 本次 --spatial-window；
#     且 roles.<role>.pooled_log_p50 == 从【该套】*_fft2[_<win>].npy 独立算出的 p50（≤1e-6），
#     并 != 从另一套算出的 p50。附：用同一批 vmin/vmax 重渲染两套，产物 PNG 应与基准那套逐字节相同。
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - /tmp/out hamming <<'EOF'
import json, sys
from pathlib import Path
import numpy as np
run, win = Path(sys.argv[1]), sys.argv[2]
table = json.loads((run / "fft_window_candidates.json").read_text())
assert table.get("candidate_basis") == win, (table.get("candidate_basis"), win)
def pooled(w):
    files = sorted((run / "fft_raw" / role).glob(
        "*_fft2.npy" if w == "none" else f"*_fft2_{w}.npy"))
    if not files:
        return np.zeros(0)                      # the other set may not exist
    vals = [np.log(np.abs(np.load(p))) for p in files]
    return np.concatenate([v[np.isfinite(v)] for v in vals])
for role in table["roles"]:
    a = table["roles"][role]["pooled_log_p50"]
    b = float(np.percentile(pooled(win), 50))
    other = "none" if win != "none" else "hamming"
    cross = pooled(other)          # a none-run holds no hamming planes (and vice versa)
    if cross.size:
        c = float(np.percentile(cross, 50))
        print(f"{role}: table {a!r} vs mine({win}) {b!r} |d|={abs(a - b):.3e} ; "
              f"mine({other}) {c!r} |d|={abs(a - c):.3e}")
        assert abs(a - c) > 1e-3, "the basis cannot be distinguished from the other set"
    else:
        print(f"{role}: table {a!r} vs mine({win}) {b!r} |d|={abs(a - b):.3e} ; "
              f"({other} planes absent in this run, cross-check skipped)")
    assert abs(a - b) <= 1e-6, "the table is NOT built on the basis planes"
print("candidate basis check ok (value-based)")
EOF
```

```bash
# 11) affine 几何镜像必须逐位一致：grid_sym.image_transform 与 affine skill 的
#     stm_apply_transform._image_transform 对同一 (M, n, pad) 给出的 matrix / n_out / offset
#     必须完全相同（本 skill 有意不 import 对方的脚本，一致性只能这样钉住）。
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'EOF'
import importlib.util, sys

import numpy as np

spec = importlib.util.spec_from_file_location(
    "stm_apply_transform_affine", "skills/affine-correction/scripts/stm_apply_transform.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
sys.path.insert(0, "skills/grid-sym-fft/scripts")
import grid_sym as gs

for M, n, pad in (
    ([[1.0113155760851729, -0.011608167367382904],
      [-0.011608167367382904, 0.9802575538146885]], 304, 10),
    ([[1.0, 0.0], [0.0, 1.0]], 512, 10),
    ([[1.03, 0.01], [0.01, 0.97]], 331, 7),
):
    M = np.asarray(M, dtype=float)
    mine = gs.image_transform(M, n, pad)
    theirs = mod._image_transform(M, n, pad)
    assert np.array_equal(mine[0], theirs[0]), (M, n, pad, "matrix")
    assert mine[1] == theirs[1], (M, n, pad, "n_out", mine[1], theirs[1])
    assert np.array_equal(mine[2], theirs[2]), (M, n, pad, "offset")
    print(f"det M = {np.linalg.det(M):.6f}, n = {n}, pad = {pad}: matrix bit-equal, "
          f"n_out {mine[1]} == {theirs[1]}, offset bit-equal")
print("affine geometry mirror is bit-for-bit")
EOF
```

**12) `--upsample N>1` 的三条口径**（都可离线复算，不需要跑全链）：

- **画布与几何**：`fine_image_transform(M, n, N)` 给出 `n_out = n·N`、`span = 2·max|A⁻¹·corners|`
  （corners = ±(n−1)/2）、`scale = (n_out−1)/span`、`offset = (n−1)/2 − (A/scale)·((n_out−1)/2)`；几何 NaN 占比
  `1 − |det A⁻¹|·(n−1)²/span²` 应等于 corrected CSV 的实测 NaN 占比（本例 5.27 % 对 5.35 %，差 ≤0.1 pp 是三阶样条的
  边界带；N=1 的旧画布是 16.96 %，因为多了 pad 带）。同一 (M, n, N) 下 `span`/`FOV_out` **与 N 无关**，只有像素数变。
- **一次 cubic、没有预放大**：corrected CSV 必须等于
  `affine_transform(map, A/scale, offset, (n·N, n·N), order=order, mode="constant", cval=NaN)`
  的 `%.10e` 量化（逐位或 ≤1e-11 相对）；把 map 先 `zoom(N, order=1, grid_mode=True, mode='nearest')` 再 warp 的结果必须**不同（非零）**——完全相同就说明实现退回了「先线性上采样再仿射」的两步链。本条只判「不同/相同」，**不引用任何量级**（量级随两步探针的索引换算 `matrix_two = N·A_fine`、`offset_two = N·off_fine + (N−1)/2`、`mode`/`cval`、掩码与是否剔除边缘而变，见 §2.6）。
  与 notebook 的 `griddata(method="cubic")` 对照（同一坐标、同一目标点）：线性 ramp 探针两引擎必须一致到 ~1e-9
  （几何门，本例 interior median 1.05e-8 / 值域 1424）；真实 map 上 interior median |Δ| / 值域 = 0.22 %（didv）与
  0.04 %（current），σ=1 平滑后降到 0.025 % / 0.006 % ⇒ 差异来自核（B 样条 vs Clough–Tocher）作用在像素级粗糙度上，
  不是几何（`interior` 口径见 §2.6：剔除距源图边界 ≤2 px 的采样带；换口径数值会变）；两者的有限掩码（NaN 区）必须一致（本例 0 处不一致）。
- **FFT 裁窗**：`C = round(span)`；`fft_corrected`/`z3`/`c3dc` 的 npy 形状必须是 `(C, C)`
  （N=1 时不裁、等于 corrected 画布），`z3` complex128、`c3dc` 实数；裁剪从 `N_fft//2 − C//2` 起算 ⇒ 裁后 DC 仍在
  `C//2`。`fft_corrected/*_fft2.npy` 的取值必须等于「corrected CSV 复算的 `fftshift(fft2(填 NaN + 空间窗))` 后取的同一中央块」。


**本 skill 未修改 `skills/lawler-fujita-correction` 与 `skills/grid-lf-pipeline` 下任何文件。**