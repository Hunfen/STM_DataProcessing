# STM 绘图工具函数模块接口文档

## 模块概述

- **模块路径**：`src/stm_data_processing/utils/plot_funcs.py`
- **职责**：基于 `nanonispy`（Nanonis 数据读取）+ `matplotlib` + `OpenCV` 提供 STM/S 实验数据的图像预处理与单图绘制函数，涵盖形貌图、map（dI/dV）、QPI、电流图、STS 单谱、linecut 等，供 `nanonis_ppt_generator` 等下游模块调用。
- **兼容性补丁**：模块顶部设置 `np.float = float`、`np.int = int`，用于兼容 NumPy 2.x 中被移除的别名。
- **绘图输出**：各绘图函数以 `matplotlib` 保存 `.tif`（依赖 Pillow 后端），`bbox_inches="tight", transparent=True, pad_inches=0`，并 `plt.close()` 释放图形。

---

## 核心函数

### 工具函数

#### `get_divider(file_path)`

根据路径中的缩放标记返回偏压缩放因子。

```python
def get_divider(file_path: Path) -> int
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `file_path` | `Path` | 数据文件路径 |

**返回**：`int`，缩放因子。源码按 `"d1"` → 1、`"d10"` → 10、`"d100"` → 100 顺序做子串匹配，默认返回 1。

> ⚠️ 注意：由于 `"d1"` 是 `"d10"` / `"d100"` 的子串，且 `"d1"` 分支最先判断，因此 `"d10"`、`"d100"` 命中的路径当前也会命中 `"d1"` 分支返回 1（`"d10"`/`"d100"` 分支实际不可达）。

#### `angle_def(angle)`

将角度映射到 `[0, 360)` 区间。

```python
def angle_def(angle):
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `angle` | 数值 | 输入角度 |

**返回**：非负角度；负数则 `360 + angle`。

#### `img_rotate(data, angle, range_x, range_y)`

以图像中心旋转数据，并返回旋转后的物理尺寸（考虑扫描范围变化）。

```python
def img_rotate(data, angle, range_x, range_y):
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `data` | `np.ndarray` | 2D 图像数据 |
| `angle` | `float` | 旋转角度 |
| `range_x`, `range_y` | `float` | 原始扫描范围（x/y） |

**返回**：`(rotated_image, range_x_new, range_y_new)`，其中旋转后的边界尺寸为 `range_x_new = range_y·sinθ + range_x·cosθ`，`range_y_new = range_y·cosθ + range_x·sinθ`（θ 取旋转矩阵元素绝对值）。

#### `img_rotate_for_box(data, degree=90, zoom_pan=1)`

仅旋转图像（尺寸不变），用于概览缩略图。

```python
def img_rotate_for_box(data, degree=90, zoom_pan=1):
```

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `data` | `np.ndarray` | - | 2D 图像数据 |
| `degree` | `float` | `90` | 旋转角度 |
| `zoom_pan` | `float` | `1` | 缩放因子 |

**返回**：旋转后的图像，尺寸与输入相同（`cv2.warpAffine` 输出 `(cols, rows)`）。

#### `subtractMeanPlane(matrix)`

最小二乘拟合最佳平面并扣除，用于去除形貌图的倾斜衬底。

```python
def subtractMeanPlane(matrix):
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `matrix` | `np.ndarray` | 2D 形貌数据 |

**返回**：`matrix - plane`（同形状），`plane = coeffs[0]·x + coeffs[1]·y + coeffs[2]`，系数由 `np.linalg.lstsq` 解得。

---

### 单图绘图函数

#### `plot_sxm_topo(topopath, output_path)`

绘制简单形貌图（不旋转、无标注）并保存。

```python
def plot_sxm_topo(topopath: Path, output_path: Path) -> None
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `topopath` | `Path` | `.sxm` 文件路径 |
| `output_path` | `Path` | 输出图片路径 |

**返回**：`None`。数据取 `nap.read.Scan` 的 `signals["Z"]["forward"]`，`cmap="Blues_r"`；`scan_dir == "up"` 时 `origin="lower"`，否则不设置 `origin`；`extent` 用 `scan_range * 1e9` 转换为纳米。

#### `plot_map_bias(mappath, n, output_dir)`

绘制指定偏压索引的 dI/dV map 图，保存到 `output_dir / "temp_map_{n}.tif"`。

```python
def plot_map_bias(mappath: Path, n: int, output_dir: Path) -> Path
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `mappath` | `Path` | `.3ds` 网格文件路径 |
| `n` | `int` | 偏压索引 |
| `output_dir` | `Path` | 输出目录 |

**返回**：保存的图片路径（`Path`）。数据取 `LI Demod 1 Y (A)`（缺省回退 `LI Demod 1 Y [AVG] (A)`）的 `[:, :, n]` 切片；偏压列表优先从 header 的分段偏压信息计算，失败时用 `signals["sweep_signal"] * 1000 / divider`；图上以 `f"{bias[n]:.2f} mV"` 标注。

#### `plot_qpi_bias(mappath, n, output_dir)`

绘制指定偏压的 QPI（FFT 功率谱对数）图，保存到 `output_dir / "temp_QPI_{n}.tif"`。

```python
def plot_qpi_bias(mappath: Path, n: int, output_dir: Path) -> Path
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `mappath` | `Path` | `.3ds` 网格文件路径 |
| `n` | `int` | 偏压索引 |
| `output_dir` | `Path` | 输出目录 |

**返回**：保存的图片路径（`Path`）。数据处理：`fft2 → fftshift → log(1 + |·|)`；显示范围 `vmin = min(qpi2)`，`vmax = mean + 1.5·std`；倒空间范围 `range_qx = 2π / (scan_range[0]·1e9 / data.shape[1])`（qy 同理）。

#### `plot_map_current_bias(mappath, n, output_dir, smooth=False)`

绘制指定偏压的电流图，保存到 `output_dir / "temp_mapI_{n}.tif"`。

```python
def plot_map_current_bias(
    mappath: Path, n: int, output_dir: Path, smooth: bool = False
) -> Path
```

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `mappath` | `Path` | - | `.3ds` 网格文件路径 |
| `n` | `int` | - | 偏压索引 |
| `output_dir` | `Path` | - | 输出目录 |
| `smooth` | `bool` | `False` | 是否对图像做高斯滤波（`gaussian_filter(sigma=1)`） |

**返回**：保存的图片路径（`Path`）。数据取 `signals["Current (A)"][:, :, n]`。

#### `plot_sts(stspath, topopath, output_dir, smooth=False)`

绘制单条谱线与对应形貌图（带打谱位置标记）。

```python
def plot_sts(stspath: Path, topopath: Path, output_dir: Path, smooth: bool = False):
```

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `stspath` | `Path` | - | `.dat` 谱文件路径 |
| `topopath` | `Path` | - | `.sxm` 形貌文件路径，可为 `None` |
| `output_dir` | `Path` | - | 输出目录 |
| `smooth` | `bool` | `False` | 是否对谱做高斯滤波（`gaussian_filter1d(sigma=1)`） |

**返回**：`(sts_path, topo_marked_path)`，前者为 `output_dir / "temp_sts.tif"`；`topopath` 为 `None` 时后者为 `None`，否则为 `output_dir / "temp_ststopo.tif"`。谱数据 `bias = "Bias calc (V)"·1000/divider`，`didv` 取 `LI Demod 1 Y [AVG] (A)`（缺省回退无 AVG），形貌经 `img_rotate` 旋转（`scan_dir == "up"` 取 `+angle`，否则 `-angle`）并在 `(X, Y)`（由 header `"X (m)"`/`"Y (m)"` ×1e9）处画红点。

#### `plot_linecut(lcpath, topopath, output_dir, smooth=False)`

为 linecut 绘制三张图：瀑布图（含堆叠谱 + 高度轮廓）、overlap 图、形貌标记图。

```python
def plot_linecut(lcpath: Path, topopath: Path, output_dir: Path, smooth: bool = False):
```

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `lcpath` | `Path` | - | `.3ds` linecut 文件路径 |
| `topopath` | `Path` | - | `.sxm` 形貌文件路径，可为 `None` |
| `output_dir` | `Path` | - | 输出目录 |
| `smooth` | `bool` | `False` | 是否沿偏压轴高斯滤波（`gaussian_filter1d(sigma=1, axis=1)`） |

**返回**：`(lc_path, ol_path, topo_marked_path)`，依次为 `temp_lc.tif`、`temp_ol.tif`、`temp_lctopo.tif`；`topopath` 为 `None` 时第三项为 `None`。linecut 长度 `L = size_xy[0]·1e9`，数据取 `LI Demod 1 Y [AVG] (A)[0, :, :]`（缺省回退无 AVG），高度取 `signals["topo"][0][:]·1e12`；形貌标记用 linecut 中心 `pos_xy`、尺寸 `size_xy[0]`、角度 `angle·π/180` 计算起终点 `(X,Y)` 与 `(XX,YY)`。

#### `update_frame_from_dir(frame_dir, n, ax)`

从目录读取 `temp_map_{n}.tif` 并显示到指定坐标轴（用于动画帧）。

```python
def update_frame_from_dir(frame_dir: Path, n, ax):
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `frame_dir` | `Path` | 存放 `temp_map_{n}.tif` 的目录 |
| `n` | `int` | 帧索引 |
| `ax` | `matplotlib.axes.Axes` | 目标坐标轴 |

**返回**：`None`。先 `ax.cla()`、`ax.set_axis_off()`，再 `plt.imread` 并 `imshow`，`extent=[0, cols, 0, rows]`。

---

## 数学公式

### 图像旋转（`img_rotate`）

以图像中心 `(cx, cy) = ((cols-1)/2, (rows-1)/2)` 旋转，旋转矩阵由 `cv2.getRotationMatrix2D` 生成。旋转后外接框尺寸：

```
cols_new = rows·sinθ + cols·cosθ
rows_new = rows·cosθ + cols·sinθ
range_x_new = range_y·sinθ + range_x·cosθ
range_y_new = range_y·cosθ + range_x·sinθ
```

其中 `cosθ = |M[0,0]|`、`sinθ = |M[0,1]|`。

### 平面扣除（`subtractMeanPlane`）

对 `(x, y, 1)` 设计矩阵做最小二乘：

```
plane = a·x + b·y + c，其中 [a, b, c] = lstsq(A, b_flat)
A = [x.ravel(), y.ravel(), ones]
```

### QPI 功率谱（`plot_qpi_bias`）

```
qpi2 = log(1 + |fftshift(fft2(data))|)
vmin = min(qpi2),  vmax = mean(qpi2) + 1.5·std(qpi2)
range_qx = 2π / (scan_range_x·1e9 / N_cols)
```

---

## 使用示例

```python
from pathlib import Path
from stm_data_processing.utils.plot_funcs import (
    plot_sxm_topo,
    plot_map_bias,
    plot_qpi_bias,
    plot_sts,
    subtractMeanPlane,
)

# 形貌图
plot_sxm_topo(Path("scan.sxm"), Path("topo.tif"))

# 第 0 个偏压的 dI/dV map 与 QPI
out = Path("out"); out.mkdir(exist_ok=True)
map_img = plot_map_bias(Path("grid.3ds"), 0, out)      # out/temp_map_0.tif
qpi_img = plot_qpi_bias(Path("grid.3ds"), 0, out)      # out/temp_QPI_0.tif

# STS 单谱 + 形貌标记
sts_img, topo_marked = plot_sts(Path("spec.dat"), Path("scan.sxm"), out)

# 平面扣除（作为预处理）
import nanonispy as nap
import numpy as np
topo = nap.read.Scan("scan.sxm").signals["Z"]["forward"]
topo_flat = subtractMeanPlane(topo)
```

---

## 依赖项

| 依赖 | 必需 | 说明 |
|------|------|------|
| `numpy` | 是 | 数组计算 |
| `cv2`（opencv-python） | 是 | 图像旋转/仿射变换 |
| `matplotlib` | 是 | 绘图与 `.tif` 输出 |
| `nanonispy` | 是 | Nanonis 数据读取 |
| `scipy.ndimage` | 是 | 高斯滤波（`gaussian_filter` / `gaussian_filter1d`） |
| `pathlib` | 是 | 路径处理 |
| Pillow（经 matplotlib 隐式依赖） | 是 | `.tif` / GIF 保存 |

---

## 错误处理

| 异常 | 触发条件 |
|------|----------|
| `FileNotFoundError` | 数据文件（`.sxm`/`.3ds`/`.dat`）不存在 |
| `Exception`（被捕获） | `plot_map_bias`/`plot_qpi_bias`/`plot_map_current_bias` 读取分段偏压 header 失败时回退 `sweep_signal`；读取 `LI Demod 1 Y (A)` 失败时回退 `[AVG]` 信号 |
| `KeyError` | 数据缺少预期信号且无回退路径时（如 `Current (A)`） |

> 绘图函数依赖 Nanonis 文件的 header 字段（`scan_range`、`scan_dir`、`size_xy` 等），字段缺失时 `KeyError` 会向上抛出。

---

## 接口对齐检查清单

生成调用此模块的代码时，请确保：

- [ ] **`get_divider` 依赖路径子串匹配，注意 `"d1"` 分支优先级导致 `"d10"/"d100"` 实际返回 1**
- [ ] **`img_rotate` 返回三元组 `(rotated_image, range_x_new, range_y_new)`，`img_rotate_for_box` 只返回图像**
- [ ] **绘图函数输出路径固定为 `output_dir / "temp_*.tif"`，并返回 `Path`**
- [ ] **`plot_sts` / `plot_linecut` 的 `topopath` 可为 `None`，此时标记图返回 `None`**
- [ ] **`smooth` 参数：map 用 `gaussian_filter(sigma=1)`，谱/linecut 用 `gaussian_filter1d(sigma=1)`**
- [ ] **`plot_map_bias` 的 `n` 为偏压索引，须小于偏压点数**
- [ ] **函数内部会 `plt.close(fig)`，调用方无需手动关闭图形**
- [ ] **模块设置 `np.float`/`np.int` 兼容别名，依赖 NumPy 版本时注意**

---

## 版本信息

- 模块路径：`src/stm_data_processing/utils/plot_funcs.py`
- 绘图后端：`matplotlib`（由 `nanonis_ppt_generator` 统一 `matplotlib.use("Agg")`，本模块不设置）
- 日志级别：无（静默绘图）
- **依赖 Nanonis 数据格式（`nanonispy` 读取），图像统一输出 `.tif`**
