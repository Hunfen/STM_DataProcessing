---
name: grid-sym-fft
description: 把 Nanonis grid spectroscopy（`.h5`，经 `stm_data_processing.io.grid2h5` 转换）的任意通道按名字抽成角色图（fwd/bwd 平均），用 `grid-lf-pipeline`（Skill A）导出的 Lawler–Fujita 位移场包做「只重采样、不改值域」的矫正，然后对每张 map 做矫正前/后的复数 FFT，并做绕 DC 像素的 Z3（复数域）与 C3（先取幅度再旋转 120/240 平均）三重对称化；第一组 FFT 图后先生成多档色标候选（contact sheet）供用户挑选，选定后所有 FFT 图共用该窗口；rmap/zmap 不在范围内。当用户给出 grid `.h5` + 角色映射（如 current / didv）+ 位移场包，要求列出全部 channel、按名字抽取、套用 LF 位移场、出复数 FFT 与 Z3/C3 对称化、并先挑色标再出全套时使用。
---

# grid-sym-fft（grid → 角色抽取 → LF 位移场矫正 → 复数 FFT → Z3 / 绕 DC 的 C3，v1.0）

**范围**：读取 grid `.h5`，给出全部 channel 及其 fwd/bwd 配对，由用户指定每个要抽的角色（如 `current` / `didv`，角色名自由）→ 每个角色 = 前向通道与其 `[bwd]` 配对取平均，每偏压一张 map → 用 **Skill A（`grid-lf-pipeline`）** 导出的可迁移位移场包做**只 flipud + 重采样、绝不改值域**的矫正 → 出矫正前/后的复数 FFT → 绕 DC 像素做 **Z3**（复数域）与 **C3**（先取幅度）三重对称化。第一组 FFT（b 阶段）之后生成 `--window-candidates` 张色标候选 + contact sheet 供用户挑选；选定后所有 FFT 图共用该窗口。**rmap / zmap 明确不在范围内**。

本 skill **不拟合位移场**（那是 Skill A 的职责），也不修改 `skills/lawler-fujita-correction` 下任何文件——它按相对路径调用 `lawler-fujita-correction` 的 `lf_lib.warp_by_field`（与仓库根 `scripts/apply_lf_transform.py` 同一套只重采样逻辑）。脚本、log 与本文档都不给物理结论。

## 1 输入与执行环境

| 项 | 说明 |
| --- | --- |
| 输入 | `<grid>.h5`（`python -m stm_data_processing.io.grid2h5 <grid>.3ds -o <grid>.h5` 得到）；角色映射（`--map ROLE="CHANNEL NAME"`）；Skill A 的位移场包 |
| h5 结构 | `/data (ny, nx, nch, npts)`、`/channels`、`/channel_units`、`/bias`（V，偏压标签只能从这里读）、`/params` + `/param_columns`、`/header` |
| 位移场包 | Skill A 产出的 `<stem>_transform.json`（含 `u_field_file`、`n_px_reference`、`field_of_view_nm_reference`、`pad`、`order`）+ `<stem>_transform.h5`（`u_x` / `u_y` / `valid`，nm / 参考网格） |
| 视场 L (nm) | 从 h5 header 解析 `Scan>Scanfield`（`;` 分隔，第 3 段宽 ×1e9=nm），否则退回 `Grid settings`；与 Skill A 的 `grid_topo.py` 同源 |
| 解释器 | 仓库 venv：`cd /path/to/STM_DataProcessing && MPLCONFIGDIR=<可写目录> PYTHONDONTWRITEBYTECODE=1 .venv/bin/python <脚本>`（不要用 `uv run`；`PYTHONDONTWRITEBYTECODE=1` 与 Skill A 一致，避免在 `skills/grid-sym-fft/scripts/` 里留下 `__pycache__/`） |
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
    --transform <...> -o OUT --pick-window K
# 或用户给出显式窗口
.venv/bin/python skills/grid-sym-fft/scripts/grid_sym.py GRID.h5 \
    --map ... --transform <...> -o OUT --fft-window VMIN VMAX
```

按下列顺序执行，**每一步都写数据 + 图 + log，且每一步都配 FFT**：

**a. 抽取**：每个角色 = 前向通道与其同名 `[bwd]` 配对（`Current (A)` ↔ `Current [bwd] (A)`）取平均，每偏压一张 map（bias 标签从 `/bias` 读，遍历全部 bias 点；配对规则与两个名字、两个索引都写进 log）。写 CSV + PNG（viridis，窗口 = 该图 `nanmedian ± 5σ`，窗口数值写进 log）。CSV 必须等于两方向平均（可用 `/data` 的 fwd/bwd 两片独立复算核对；只取前向时差可达 ~1e-10 量级）。

**b. 矫正前 FFT**：`x = map`（NaN 用有效像素 nanmedian 填充）；`F = np.fft.fftshift(np.fft.fft2(x))`，complex128 存 npy；出 `log|F|` 图。这一组图后面跟色标候选。

**c. 矫正**：用 `--transform` 指向的位移场包矫正每个角色每张 map，**只 flipud + 重采样、绝不改值域**（禁止调用会去平面的 `stm_lf_apply.py`；走 `lawler-fujita-correction` 的 `lf_lib.warp_by_field` 或仓库根 `scripts/apply_lf_transform.py` 这条只重采样的路）。画布按 `2*half` 增长（`half = ceil(max|u|) + pad`）。写 corrected CSV + PNG + 该阶段自己的复数 FFT + `log|F|` 图。

**d. Z3**：对每张矫正后 map 的 `F`，`Z = (F + R120 F + R240 F)/3` 在**复数域**做，旋转**绕 DC 像素**：`R` 为平面内 120/240 度旋转矩阵，`offset = c - R c`，`c = fftshift 后 DC 的像素坐标（浮点，偶数 N 时为 N/2）`，`scipy.ndimage.affine_transform(matrix=R, offset=c-R c, order=3, mode='nearest')`。存 complex128 npy + `log|Z|` 图，log 报 DC 保留（应为 1.0000）、三重残差输出/输入之比、能量比。

**e. 绕 DC 的 C3**：先取幅度 `A = |F|`，再同样绕 DC 像素转 120/240 并平均 `C3dc = (A + R120 A + R240 A)/3`（与 Z3 **唯一区别就是先取模**）。存 npy + `log|C3dc|` 图。三次插值振铃会给出少量负值，故 log 用 `log|C3dc|`，并把负值个数与相对幅度记进 log。

**f. 色标**：第一组（b 阶段）FFT 图之后生成 `--window-candidates N` 张（**N 必填：该参数没有默认值**）不同 vmin/vmax 的候选 + 一张 contact sheet。候选必须同画布尺寸（PIL 断言全部一致）、每张标注 vmin/vmax 与两端压死百分比，并打印候选表（编号/vmin/vmax/两端裁剪比例）。用户选定后用 `--pick-window K`（读上一轮的候选表）或 `--fft-window VMIN VMAX` 重跑，写入 `<OUTDIR>/fft_window.json`；此后**所有** FFT 图（fft_raw / fft_corrected / z3 / c3dc）共用该窗口。已存在 `fft_window.json` 时直接读取复用（幂等）。尚未选窗时只做 a+b 并退出 0，提示用户选窗后重跑。

## 3 产物布局

数据与图同目录、每个量一个目录（`<i>` = bias 索引，`<bias>meV` 从 `/bias` 读并 ×1000，如 `0_30meV`、`3_-5meV`）：

```
<OUT>/raw/<role>/<i>_<bias>meV.csv|.png
<OUT>/fft_raw/<role>/<i>_<bias>meV_fft2.npy|_fft.png
<OUT>/corrected/<role>/<i>_<bias>meV.csv|.png
<OUT>/fft_corrected/<role>/<i>_<bias>meV_fft2.npy|_fft.png
<OUT>/z3/<role>/<i>_<bias>meV_fft2.npy|_fft.png
<OUT>/c3dc/<role>/<i>_<bias>meV_fft2.npy|_fft.png
<OUT>/window_candidates/window_KK.png|contact_sheet.png     # 候选色标（第一次运行）
<OUT>/fft_window.json            # 选定窗口 {vmin, vmax}
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

1. **跑 `--window-candidates N`** → 出 a+b + N 张候选 + contact sheet + 候选表，缓存到 `fft_window_candidates.json`，**exit 0**。
2. **用户从候选表挑编号 K** → **跑 `--pick-window K`**（或直接 `--fft-window VMIN VMAX`）→ 写 `fft_window.json`，出全套 a+b+c+d+e。
3. `fft_window.json` 已存在 → 直接复用（幂等），全套照跑。

## 6 fail-closed（非零退出且不写半成品）

| 情形 | 行为 |
| --- | --- |
| 角色通道名匹配不到 / 匹配到多个 | 列出全部可用名字，退出非零，不猜 |
| 给的是 `[bwd]` 名字 | 拒绝（fwd 名字是规范入口），退出非零 |
| `--transform` 缺失 / 目录里不唯一（0 或 ≥2 个 `*_transform.json`） | 报错，退出非零 |
| h5 缺 `/params` 的 `Z (m)` 列 | 报错，退出非零 |
| 尚未选窗却要求出全套（无 `fft_window.json`，又未给 `--pick-window`/`--fft-window`） | 提示先跑 `--window-candidates`，退出非零 |
| `--pick-window K` 但候选表不存在 / K 越界 | 报错，退出非零 |
| `--fft-window VMIN VMAX` 且 VMIN ≥ VMAX | 报错，退出非零 |
| h5 视场与位移场包的 `field_of_view_nm_reference` 不一致 | 按 `abs(size_nm - reference_fov) <= 1e-9 * max(1, reference_fov)` 判定；不一致时用 `lf_lib.resample_field` 把 u 场重采样到本图网格（不是报错） |

**落盘纪律**：失败时只清理**本次运行写出**的东西，不动用户原有的文件；Skill A 的 `grid_lf_fit.py` 同理（进入时先记下 `lf/` 已有内容，失败只删本次新增项，其余保留）。

## 7 已知坑

1. **bias 标签只能从 `/bias` 读**（单位 V，需 ×1000 得 meV），不得写死；文件名 `<i>_<bias>meV` 中的 `<i>` 是 bias 索引。
2. **旋转必须绕 DC 像素**：`fftshift` 后偶数 N 的 DC 在索引 `N/2`（不是数组几何中心 `(N-1)/2`）。用 `scipy.ndimage.affine_transform(matrix=R, offset=c-R c, order=3, mode='nearest')`，`c=(N/2, N/2)`（浮点）；这样 DC 保留到 ~1e-16。用 `ndimage.rotate`（绕 `(N-1)/2`）会把中心尖峰糊掉。
3. **c3dc 的 log 必须用 `log|C3dc|`**：三次插值振铃给出少量负值（本例 8–16 像素/图、最大负值幅度约为 DC 峰值的 0.07、负值功率占全谱 ~1.7%），负值个数与相对幅度要写进 log。
4. **log 里的 `DC_retained 1.0000` 是无信息装饰项**：旋转按 `offset = c - R c` 构造，恒有 `output(c) = input(c)`（`mode='nearest'` 下越界采样也不涉及 DC），所以这个比值在构造上恒等于 1，不管对称化是否有效。判断 Z3/C3 是否真的生效请看 `residual_in -> residual_out` 与 `energy_ratio`；不要把这个 1.0000 当证据引用。
5. **未经选窗的 FFT 图窗口是临时的**：candidates 运行给 fft_raw 用 p50–p99.9 的临时窗；用户选窗后全套重画，所有 FFT 图用选定窗口。
6. **候选图必须同画布尺寸**：固定 figure 尺寸 + 不 `bbox_inches='tight'`，保证逐像素可比；PIL 断言全部一致后才拼 contact sheet（图注拆短行，防 tight bbox 撑大画布）。

## 8 自检口径

```bash
# 1) 无通道索引字面量
cd /path/to/STM_DataProcessing && ! grep -rEn '\[\s*:\s*,\s*:\s*,\s*[0-9]+\s*,\s*:\s*\]' skills/grid-sym-fft

# 2) grid_channels 非交互模式
MPLCONFIGDIR=<writable> PYTHONDONTWRITEBYTECODE=1 .venv/bin/python skills/grid-sym-fft/scripts/grid_channels.py \
    GRID.h5 --map current="Current (A)" --map didv="DSP 7280 Y (%)"

# 3) 先出候选（a+b + 候选 + contact sheet），exit 0
MPLCONFIGDIR=<writable> PYTHONDONTWRITEBYTECODE=1 .venv/bin/python skills/grid-sym-fft/scripts/grid_sym.py \
    GRID.h5 --map current="Current (A)" --map didv="DSP 7280 Y (%)" \
    --transform OUT/SkillA/lf -o /tmp/out --window-candidates 9

# 4) 挑候选 K 出全套
MPLCONFIGDIR=<writable> PYTHONDONTWRITEBYTECODE=1 .venv/bin/python skills/grid-sym-fft/scripts/grid_sym.py \
    GRID.h5 --map current="Current (A)" --map didv="DSP 7280 Y (%)" \
    --transform OUT/SkillA/lf -o /tmp/out --pick-window 3

# 5) 产物计数（每个角色每偏压一个 npy）
cd /tmp/out && test -f fft_window.json \
    && test "$(ls z3/didv/*_fft2.npy | wc -l)" -eq 5 \
    && test "$(ls c3dc/didv/*_fft2.npy | wc -l)" -eq 5 \
    && test "$(ls fft_corrected/didv/*_fft2.npy | wc -l)" -eq 5
```

**本 skill 未修改 `skills/lawler-fujita-correction` 与 `skills/grid-lf-pipeline` 下任何文件。**