---
name: grid-sym-fft
description: 把 Nanonis grid spectroscopy（`.h5`，经 `stm_data_processing.io.grid2h5` 转换）的任意通道按名字抽成角色图（fwd/bwd 平均），用 `grid-lf-pipeline`（Skill A）导出的 Lawler–Fujita 位移场包做「只重采样、不改值域」的矫正，然后对每张 map 做矫正前/后的复数 FFT，并做绕 DC 像素的 Z3（复数域）与 C3（先取幅度再旋转 120/240 平均）三重对称化；第一组 FFT 图后**按角色各生成**多档色标候选（每角色一套 contact sheet）供用户挑选，选定后**同一角色**的 FFT 图共用该角色的窗口（角色之间绝不共用：current 与 didv 的单位不同，log|F| 中位数差 ~26）；rmap/zmap 不在范围内。当用户给出 grid `.h5` + 角色映射（如 current / didv）+ 位移场包，要求列出全部 channel、按名字抽取、套用 LF 位移场、出复数 FFT 与 Z3/C3 对称化、并先挑色标再出全套时使用。
---

# grid-sym-fft（grid → 角色抽取 → LF 位移场矫正 → 复数 FFT → Z3 / 绕 DC 的 C3，v1.0）

**范围**：读取 grid `.h5`，给出全部 channel 及其 fwd/bwd 配对，由用户指定每个要抽的角色（如 `current` / `didv`，角色名自由）→ 每个角色 = 前向通道与其 `[bwd]` 配对取平均，每偏压一张 map → 用 **Skill A（`grid-lf-pipeline`）** 导出的可迁移位移场包做**只 flipud + 重采样、绝不改值域**的矫正 → 出矫正前/后的复数 FFT → 绕 DC 像素做 **Z3**（复数域）与 **C3**（先取幅度）三重对称化。第一组 FFT（b 阶段）之后**按角色各**生成 `--window-candidates` 张色标候选 + 该角色自己的 contact sheet 供用户挑选；选定后**同一角色**的所有 FFT 图共用该角色的窗口（**角色之间绝不共用**：`current` 与 `didv` 单位不同，log|F| 中位数相差 ~26，池化窗口会把一个角色整幅压到色标顶部、另一个压到底部）。**rmap / zmap 明确不在范围内**。

本 skill **不拟合位移场**（那是 Skill A 的职责），也不修改 `skills/lawler-fujita-correction` 下任何文件——它按相对路径调用 `lawler-fujita-correction` 的 `lf_lib.warp_by_field`（与仓库根 `scripts/apply_lf_transform.py` 同一套只重采样逻辑）。脚本、log 与本文档都不给物理结论。

## 1 输入与执行环境

| 项 | 说明 |
| --- | --- |
| 输入 | `<grid>.h5`（`python -m stm_data_processing.io.grid2h5 <grid>.3ds -o <grid>.h5` 得到）；角色映射（`--map ROLE="CHANNEL NAME"`）；Skill A 的位移场包 |
| h5 结构 | `/data (ny, nx, nch, npts)`、`/channels`、`/channel_units`、`/bias`（V，偏压标签只能从这里读）、`/params` + `/param_columns`、`/header` |
| 位移场包 | Skill A 产出的 `<stem>_transform.json`（含 `u_field_file`、`n_px_reference`、`field_of_view_nm_reference`、`pad`、`order`）+ `<stem>_transform.h5`（`u_x` / `u_y` / `valid`，nm / 参考网格）。Skill A 写出的 `u_field_file` 是**绝对路径**（`grid_lf_fit.py` 对 `--save-transform` 做 `.resolve()`）；若遇到相对路径，按 bundle 所在目录解析（`scripts/grid_sym.py:193-195`） |
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

### 2.2 `scripts/grid_sym.py`（抽取 → 矫正 → 复数 FFT → Z3 / C3）

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
# --fft-window VMIN VMAX 仍可用，语义 = 同一对数值套给**所有**角色
# （只在角色单位恰好一致时才是想要的；否则必然有一个角色被压死）
```

按下列顺序执行，**每一步都写数据 + 图 + log，且每一步都配 FFT**：

**a. 抽取**：每个角色 = 前向通道与其同名 `[bwd]` 配对（`Current (A)` ↔ `Current [bwd] (A)`）取平均，每偏压一张 map（bias 标签从 `/bias` 读，遍历全部 bias 点；配对规则与两个名字、两个索引都写进 log）。写 CSV + PNG（viridis，窗口 = 该图 `nanmedian ± 5σ`，窗口数值写进 log）。CSV 必须等于两方向平均（可用 `/data` 的 fwd/bwd 两片独立复算核对；只取前向时差可达 ~1e-10 量级）。

**b. 矫正前 FFT**：`x = map`（NaN 用有效像素 nanmedian 填充）；`F = np.fft.fftshift(np.fft.fft2(x))`，complex128 存 npy；出 `log|F|` 图。这一组图后面跟色标候选。

**c. 矫正**：用 `--transform` 指向的位移场包矫正每个角色每张 map，**只 flipud + 重采样、绝不改值域**（禁止调用会去平面的 `stm_lf_apply.py`；走 `lawler-fujita-correction` 的 `lf_lib.warp_by_field` 或仓库根 `scripts/apply_lf_transform.py` 这条只重采样的路）。画布按 `2*half` 增长（`half = ceil(max|u|) + pad`）。写 corrected CSV + PNG + 该阶段自己的复数 FFT + `log|F|` 图。

**d. Z3**：对每张矫正后 map 的 `F`，`Z = (F + R120 F + R240 F)/3` 在**复数域**做，旋转**绕 DC 像素**：`R` 为平面内 120/240 度旋转矩阵，`offset = c - R c`，`c = fftshift 后 DC 的像素坐标（浮点，偶数 N 时为 N/2）`，`scipy.ndimage.affine_transform(matrix=R, offset=c-R c, order=3, mode='nearest')`。存 complex128 npy + `log|Z|` 图，log 报 DC 保留（应为 1.0000）、三重残差输出/输入之比、能量比。

**e. 绕 DC 的 C3**：先取幅度 `A = |F|`，再同样绕 DC 像素转 120/240 并平均 `C3dc = (A + R120 A + R240 A)/3`（与 Z3 **唯一区别就是先取模**）。存 npy + `log|C3dc|` 图。三次插值振铃会给出少量负值，故 log 用 `log|C3dc|`，并把负值个数与相对幅度记进 log。

**f. 色标（按角色，永不跨角色池化）**：第一组（b 阶段）FFT 图之后，**对每个角色各**生成 `--window-candidates N` 张（**N 必填：该参数没有默认值**）不同 vmin/vmax 的候选 + 该角色自己的一张 contact sheet，落在 `<OUT>/window_candidates/<role>/`。每个角色的候选窗口取自**该角色自己**的 log|F| 分布（该角色全部 bias 平面的并集），因此 current 一套（~−24…−15）、didv 一套（~+2.5…+10.5）互不影响。候选必须同画布尺寸（PIL 断言全部一致）。

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

用户选定后用 `--pick-window K`（读上一轮的候选表；**`K` 是每个角色各自候选表的 `index`，0-based，即该角色目录下 `window_{K:02d}.png` 的编号**，合法范围 `0 <= K < N`），或 `--fft-window-role ROLE VMIN VMAX`（可重复，按角色给显式窗口），或 `--fft-window VMIN VMAX`（= 同一对数值套给所有角色）重跑，写入 `<OUTDIR>/fft_window.json`（格式 `{"window_version": 2, "roles": {<role>: {vmin, vmax}}}`）；此后每个角色自己的所有 FFT 图（fft_raw / fft_corrected / z3 / c3dc）用**该角色**的窗口。已存在 `fft_window.json` 时直接读取复用（幂等）；若是旧的单窗口格式（无 `roles` 键）则警告后套给所有角色。尚未选窗时只做 a+b 并退出 0，提示用户选窗后重跑。

## 3 产物布局

数据与图同目录、每个量一个目录（`<i>` = bias 索引，`<bias>meV` 从 `/bias` 读并 ×1000，如 `0_30meV`、`3_-5meV`）：

```
<OUT>/raw/<role>/<i>_<bias>meV.csv|.png
<OUT>/fft_raw/<role>/<i>_<bias>meV_fft2.npy|_fft.png
<OUT>/corrected/<role>/<i>_<bias>meV.csv|.png
<OUT>/fft_corrected/<role>/<i>_<bias>meV_fft2.npy|_fft.png
<OUT>/z3/<role>/<i>_<bias>meV_fft2.npy|_fft.png
<OUT>/c3dc/<role>/<i>_<bias>meV_fft2.npy|_fft.png
<OUT>/window_candidates/<role>/window_KK.png|contact_sheet.png  # 每角色一套候选（第一次运行）
<OUT>/fft_window.json            # 选定窗口 {"window_version": 2, "roles": {<role>: {vmin, vmax}}}
<OUT>/fft_window_candidates.json # 缓存候选表（供 --pick-window 复用）
<OUT>/pipeline.log               # 逐步关键数值：raw 窗口、FFT 窗口、画布增长、
                                 # Z3/C3 的 DC 保留、三重残差输入/输出、能量比、
                                 # c3dc 负值个数与相对幅度
```

## 4 与 Skill A / lawler-fujita-correction 的分工

| 项 | `grid-lf-pipeline` (Skill A) | `lawler-fujita-correction` | 本 skill |
| --- | --- | --- | --- |
| 职责 | 3ds→h5、抽 topo（三步预处理）、LF 拟合并导出位移场包 | 拟合位移场 + 重采样实现 | 列出 channel、按名字抽角色图、把位移场包**只重采样**地套到每张 map、复数 FFT、Z3 / C3 |
| 依赖方向 | 嵌套调用 `stm_lf_correct.py` | 独立 | **按相对路径** `skills/lawler-fujita-correction/scripts/lf_lib.py` 嵌套调用其 `warp_by_field` |
| 是否修改对方 | 否 | — | **否**。只运行 `lf_lib` 的只重采样子集，不读不写 `skills/lawler-fujita-correction` 下任何文件，也不写 `skills/grid-lf-pipeline` 下任何文件 |

**位移场包从哪来**：先跑 Skill A `grid_lf_fit.py <stem>_topo.csv -L <nm> -o <out>/lf`，得到 `<out>/lf/<stem>_transform.json + .h5`；把该目录（或该 json 文件）交给本 skill 的 `--transform`。

**为什么是「只重采样」**：`apply_lf_transform.py`（与本 skill 同一套 `warp_by_field` 用法）只做 `flipud`（坐标约定，非值操作）+ `out(r)=T(r+u(r))`，从不减平面/去均值/归一化；`stm_lf_apply.py` 会减拟合平面，`corrected(r)=T(r+u(r))` 只是 Skill A 自己的复现路径，**迁移到别的 map 时禁止用**——否则目标图自己的 DC 电平与倾斜会被抹掉（实测对 +100 pA / 线性斜坡输入输出差仅 ~1e-22）。本 skill 矫正后 map 的值域与原图一致（例如 current 通道仍 ~5e-10 A 电平）。

## 5 两次运行交互协议

1. **跑 `--window-candidates N`** → 出 a+b + **每个角色** N 张候选 + 该角色的 contact sheet + 按角色分组的候选表，缓存到 `fft_window_candidates.json`，**exit 0**。
2. **用户从候选表挑编号 K** → **跑 `--pick-window K`**（每个角色各取自己第 K 档），或按角色给显式窗口 `--fft-window-role ROLE VMIN VMAX`（可重复）→ 写 `fft_window.json`，出全套 a+b+c+d+e。
3. `fft_window.json` 已存在 → 直接复用（幂等），全套照跑。
4. **角色单位不同（current 与 didv 差 ~26 个 log）时，同一对 vmin/vmax 不可能同时适合两个角色**——这正是 `--pick-window` 按角色取、`--fft-window-role` 按角色给的原因。

## 6 fail-closed（非零退出且不写半成品）

| 情形 | 行为 |
| --- | --- |
| 角色通道名匹配不到 / 匹配到多个 | 列出全部可用名字，退出非零，不猜 |
| 给的是 `[bwd]` 名字 | 拒绝（fwd 名字是规范入口），退出非零 |
| `--transform` 缺失 / 目录里不唯一（0 或 ≥2 个 `*_transform.json`） | 报错，退出非零 |
| h5 缺 `/params` 的 `Z (m)` 列 | 报错，退出非零 |
| 尚未选窗却要求出全套（无 `fft_window.json`，又未给 `--pick-window`/`--fft-window`） | 提示先跑 `--window-candidates`，退出非零 |
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
5. **未经选窗的 FFT 图窗口是临时的**：candidates 运行给 fft_raw 用**该角色** p50–p99.9 的临时窗；用户选窗后全套重画，每个角色的 FFT 图用该角色选定的窗口。
6. **候选图必须同画布尺寸**：固定 figure 尺寸 + 不 `bbox_inches='tight'`，保证逐像素可比；PIL 断言全部一致后才拼 contact sheet（图注拆短行，防 tight bbox 撑大画布）。
7. **色标窗口绝不能跨角色池化**（2026-10 实测的真 bug，用户看到 6 张 contact sheet 全是纯色）：`current` 的 log|F| 在 −27…−8.7（中位 −19.95），`didv` 在 −1.9…+16.3（中位 +6.27），池化后分位数落在 −21…+10，于是每张候选对 didv 是整幅超 vmax、对 current 是整幅低于 vmin。修前用 PIL 统计候选图坐标区：**端点（饱和）像素 59.9–98.7 %、窗口中间 80 % 区间内 ~0 %**；按角色分开后为 **0.002–1.83 % / 73.6–99.7 %**。判据与脚本：对每张候选图取坐标区（matplotlib 同一 figure 几何的 `ax.get_window_extent()`）统计「颜色等于 viridis 两端点」与「落在窗口中间 80 %」的像素占比，前者必须 < 5 %、后者必须 ≥ 50 %。

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
```


**本 skill 未修改 `skills/lawler-fujita-correction` 与 `skills/grid-lf-pipeline` 下任何文件。**