# grid2h5：Nanonis .3ds 转 HDF5

## 模块概述

`grid2h5.py` 把一个 Nanonis 栅格谱文件（`.3ds`）转换成一个 HDF5 产物，数据与元数据全部来自
`io.nanonis_loader.NanonisFileLoader` 的公开输出。

- **模块路径**：`src/stm_data_processing/io/grid2h5.py`
- **职责**：读取一个 `.3ds`，把 `/data` 数据立方、通道表、参数表与完整文件头写进一个 `.h5`。
- **写盘约定**：所有数据集经 `io.h5_convention.create_dataset()`（gzip/4 + 显式分块），文件属性经
  `write_file_metadata()`（`schema_version` / `generator` / `creation_date`）。

## 用法

```bash
.venv/bin/python -m stm_data_processing.io.grid2h5 SRC.3ds [-o OUT.h5]
```

```python
from stm_data_processing.io.grid2h5 import grid_to_h5

grid_to_h5("Grid Spectroscopy001.3ds")  # 默认写到 <source>.h5
grid_to_h5("Grid Spectroscopy001.3ds", "g1.h5")  # 指定输出
```

不传 `-o` 时输出为 `<source>.h5`。重复转换同一目标会保留其原有 `creation_date`。

## 产物结构

| 路径 | 形状 / 类型 | 说明 |
|------|-------------|------|
| `/data` | `(ny, nx, nch, npts)` float32 | 谱数据立方，属性 `source_dtype`、`axis_order` |
| `/channels` | `(nch,)` 字符串 | 通道名，顺序同文件头 |
| `/channel_units` | `(nch,)` 字符串 | 从通道名尾部括号解析出的单位（无则空串） |
| `/bias` | `(npts,)` float64 | 扫掠轴，属性 `bias_source`、`units` |
| `/params` | `(npix, nparam)` float64 | 逐像素实验参数 |
| `/param_columns` | `(nparam,)` 字符串 | 参数列名（`Fixed parameters` + `Experiment parameters`） |
| `/header` | 组树 | `loader.header` 的自适应镜像：标量键→属性，模块字典→子组 |

`nx` 取文件头 `Grid dim` 的第一个数（快轴），`ny` 取第二个数，`npts` 取 `Points`，`nch` 取通道个数；
扁平像素序号与栅格的对应关系是 `n = iy * nx + ix`。

## 关键规则

**扫掠轴（`/bias`）**：优先使用文件头 `Sweep Signal` 指定的**实测通道**——只要该通道在所有像素上取值一致，
`/bias` 就是它的逐点值，`bias_source = 'measured_channel'`。多段（MLS）扫偏压的真实轴是
`[0.03, 0.005, 0, -0.005, -0.03] V` 这类非等间距序列，用 `linspace(Sweep Start, Sweep End)` 会在中间点写错
（等间距假设给出 `[0.03, 0.015, 0, -0.015, -0.03]`）。实测通道缺失或逐像素不一致时才回落到
`linspace`，并在 `bias_source = 'linspace'` 中如实记录。两条分支都写 `units`。

**头部自适应**：`/header` 逐键镜像 `loader.header`，不为任何键写死分支，因此新版本 Nanonis 新增的头部键
会自动落盘。单属性模块保留其叶子键名；键名中的 `/` 会被转义。

**截断文件**：录制未完成（载荷短于文件头声明的长度）的 `.3ds` 仍然转换——loader 会把数据补齐为 NaN，
产物保持声明形状，NaN 位置即未录到的像素。

## 已知边界

- 名字里出现的偏压值是**扫掠起点**，不是固定偏压。
- 转换只做格式搬运，不做任何物理处理（不平移、不归一、不掩膜）。
- 一个源文件对应一个产物；不批量、不递归。

## 相关

- 数据来源与字段语义：`io/nanonis_loader.py`
- HDF5 写出约定：`io/h5_convention.py`、`docs/hdf5_convention.md`
- 回归：`tests/regression/check_grid2h5.py`（合成夹具，CI 选中）、
  `tests/regression/check_grid2h5_real.py`（真实文件）
