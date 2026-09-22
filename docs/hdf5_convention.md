# HDF5 约定（`stm_data_processing` 的唯一 h5 规范）

本仓库所有 `.h5` 产物都必须由 `src/stm_data_processing/io/h5_convention.py` 写出。
该模块是**唯一**调用 `h5py` `create_dataset` 的地方：

```
$ grep -rn 'create_dataset' src/stm_data_processing/ --include='*.py'
src/stm_data_processing/io/h5_convention.py:159:    dataset = parent.create_dataset(name, data=array, **kwargs)
```

其余命中都是各 writer 对本模块 `create_dataset` / `write_file_metadata` 的调用。
新增结果类型时**不要**直接调用 `h5py` 的 `create_dataset`。

**技能（`skills/`）的例外**：技能是自包含的分发单元，不允许 import `stm_data_processing`，
所以 `skills/affine-correction`、`skills/lawler-fujita-correction`、`skills/local-q-map`
各自在 `scripts/h5io.py` 里**逐值镜像**同一套规则（同样的 `gzip`/4、显式分块
`chunk_shape()`、`track_times=False`、按量给 `units`，以及根属性
`schema_version` / `generator` / `creation_date`），并把它作为该技能内**唯一**的
`create_dataset` 入口。上面那句「唯一」只约束 `src/stm_data_processing` 内部。改动本模块的
常量或 `chunk_shape()` 时必须同步这三个镜像——同样输入下它们必须给出相同的分块。

## 1. 文件与数据集命名

* 一个文件 = 一个结果对象（一次计算的一次产物），不把多类结果塞进同一文件。
* 主数据集用描述性的复数/量名，与 `load_*` 返回的键一致：
  * `qpi_layers` —— QPI 强度，`(n_energies, nq, nq)` 或 `(nq, nq)`（`save_qpi_to_h5`）
  * `susceptibility` —— fftshift 后的响应函数，`(nq, nq)`（`save_susceptibility_to_h5`）
  * `energies` / `k1_grid` / `k2_grid` —— 二维能带与分数坐标网格（`EK2DIO.save_ek2d`）
* 辅助数组固定在文件根下用小写名字：`bvecs`（倒格基矢，`(2,2)` 或 `(3,3)`）、
  `V`（散射势）、`mask`（实空间掩码）。
* 网格**不存储**：`q_*`/`k_*` 网格由 `nq`/`nk` 属性重建
  （`load_susceptibility_from_h5` 用 `np.fft.fftshift(np.fft.fftfreq(nq))`，
  `load_qpi_from_h5` 用 `linspace(-0.5, 0.5, nq, endpoint=False)`），
  因此文件格式与网格分辨率解耦。

## 2. 必填属性

文件级（由 `write_file_metadata()` 写入，见 `h5_convention.py`）：

| 属性 | 含义 | 取值来源 |
| --- | --- | --- |
| `schema_version` | 本约定的版本号，当前为 `SCHEMA_VERSION = 1` | 常量 |
| `generator` | 产生该文件的工具名 | `DEFAULT_GENERATOR = "stm_data_processing"`，或 writer 传入（`save_ek2d` 传 `"EK2DCalculator"`） |
| `creation_date` | 该产物首次创建时间，ISO-8601 带时区偏移（如 `2026-09-22T02:08:25+08:00`） | 新建时取当前时间；**重写已有产物时沿用原值**（见 §5） |
| `units_<量>` | 该文件里某个量的单位，可选 | `save_ek2d` 写 `units_energy="eV"`、`units_k_frac="reciprocal lattice units"` |

数据集级：

| 属性 | 含义 |
| --- | --- |
| `units` | 该数组的物理单位；**无量纲量不写**（如 `mask`、归一化后的 `qpi_layers`） |

现有单位取值：`susceptibility` → `1/eV`；`V`、`energies` → `eV`；
`bvecs` → `1/angstrom`；`k1_grid`/`k2_grid` → `reciprocal lattice units`。

其余业务属性（`module_type`、`eta`、`nq`、`bands`、`energy_range`、`temperature`、
`chemical_potential`、`omega_limit`、`resolution`、`nk`、`num_wann`、`total_points`、
`folder`、`seedname`、`orbital_select`、`projection` 等）继续由各 writer 通过
`write_file_metadata(extra=...)` 或 `**metadata_kwargs` 写入。

## 3. 分块（chunking）规则

**显式分块，不用 h5py 的隐式 auto-chunking**，规则见 `h5_convention.chunk_shape()`：

* 每个 chunk 不超过 `CHUNK_TARGET_BYTES = 1 MiB`（未压缩）；
* 从**最后一维（变化最快、最连续）开始**填满字节预算，预算用完后其余维度每块取 1 个元素；
* 数据集小于预算时 `chunks == shape`（仍是显式分块，压缩因此始终生效）；
* 标量或空数据集不指定 `chunks`。

理由：读者总是沿尾部维度切片（q/k 网格的行、能带的最后一维），尾部完整的 chunk 让这些
读取连续；1 MiB 预算则限制单块读取/改写需要的内存。实测（`var/t8/t8_evidence.py`）：

```
qpi_layers (4,256,256) float64   -> chunks=(2, 256, 256)     1 MiB
susceptibility (256,256) float64 -> chunks=(256, 256)      512 KiB
energies (10,64,64) float64      -> chunks=(10, 64, 64)    320 KiB
```

## 4. 压缩规则

**全仓统一 `gzip` + level 4**（`COMPRESSION` / `COMPRESSION_OPTS`）。

改动前的状态是分裂的：`ek2d_io` 用 gzip/4，`qpi_io` 与 `susceptibility_io` 用 gzip/6；
两者都不指定 `chunks`。统一为 level 4 的实测依据（同一台机器、同一批代表性数组）：

```
qpi_layers (4,256,256) float64 随机   gzip/4 1981.4 KiB / 33 ms | gzip/6 1981.4 KiB / 31 ms
susceptibility (256,256) float64 随机 gzip/4  495.9 KiB /  8 ms | gzip/6  495.9 KiB /  8 ms
susceptibility (1024,1024) float64    gzip/4 7887.2 KiB / 173 ms | gzip/6 7887.2 KiB / 175 ms
smooth chi0-like (256,256) float64    gzip/4  479.0 KiB /  9 ms | gzip/6  479.0 KiB /  9 ms
smooth chi0-like (1024,1024) float32  gzip/4 3452.5 KiB / 98 ms | gzip/6 3452.6 KiB / 98 ms
```

即：在本包实际写出的数组形状上，level 4 与 level 6 的体积差 **≤0.1 %**、耗时相同，
所以统一到 4 不带来体积回归。为证明该旋钮确实生效，用可压缩模式复核：

```
zeros (1024,1024)        L1 57.4 KiB | L4 29.7 KiB | L6 29.7 KiB | L9 29.7 KiB
tiled ramp (1024,1024)   L1 138.9 KiB | L4 35.8 KiB | L6 38.6 KiB | L9 30.6 KiB
rank-1 smooth (1024,1024) L1 7401.9 KiB | L4 7385.3 KiB | L6 7386.3 KiB | L9 7386.3 KiB
```

`shuffle=True` 在 rank-1 数据上能把 7385 KiB 降到 6138 KiB，但两个旧 writer 都没用它，
本约定**暂不引入**（会额外改变磁盘字节，属于独立决策）；需要时在
`h5_convention.create_dataset()` 一处加上即可，所有产物自动继承。

`create_dataset()` 总是传 `track_times=False`，让同样内容的两次写出产生逐字节相同的文件。

## 5. 版本与向后兼容

* `schema_version` 是**单调递增整数**。写入方改变**布局**（数据集名字、单位、必填属性集合、
  分块/压缩默认值）时必须 +1；仅新增可选属性不算布局变化。
* 读取方必须**向后兼容**：`load_*` 一律用 `dict(f.attrs)` / `f.attrs.get(...)` 读取，
  新属性只会作为 `metadata[...]` 的额外键出现，绝不能因为缺少 `schema_version`
  （旧文件没有该属性）而拒绝加载。旧文件（无 `schema_version`、无 `units`、
  隐式分块、gzip/6）用 `var/t8/t8_evidence.py` 第 4 节手工构造并逐一验证可读。
* **重写**已有产物时保留原 `creation_date`（`read_creation_date()`）：
  并行驱动 `write_result_h5` 先在临时文件里组装再原子替换，它把旧文件的
  `creation_date` 通过 `**metadata_kwargs` 传给 `save_susceptibility_to_h5`，
  后者把它交给 `write_file_metadata`（**只写一次** —— 同一属性写两次会改变 HDF5
  属性布局和文件字节）。这样"重算一片完全相同的分片不改变产物"这一契约才成立。

## 6. 新增结果类型的清单

1. 在 `io/` 下新建/扩展一个 `*_io.py`，写 `save_<结果>_to_h5(...)` 与 `load_<结果>_from_h5(...)`。
2. 写入侧：`with h5py.File(path, "w") as f:` → 每个数组用
   `h5_convention.create_dataset(f, name, array, units=<单位或 None>)`，
   最后调用 `h5_convention.write_file_metadata(f, generator=..., units={...}, extra={...})`
   写业务属性；**不要**自己写 `schema_version`/`generator`/`creation_date`。
3. 读取侧：只按名字取数据集，属性用 `.get(..., default)`；不要假设新属性存在。
4. 若布局有变化：`SCHEMA_VERSION += 1`，并在本节与上文表格中记录。
5. 回归：在 `tests/regression/` 里加一个 `check_*.py`（本仓约定）验证往返一致性
   （`np.array_equal`）与属性存在性，必要时在 `var/` 下手工构造旧格式文件验证可读。
6. 不要把参数数组、npz 分片等中间产物塞进正式 `.h5`（并行驱动的 npz 分片是另一套格式，
   由 `dft/wannier90/lindhard_re_chi_parallel.py` 自己管理）。

## 7. 相关实现位置

| 内容 | 位置 |
| --- | --- |
| 约定本体：chunk/压缩/属性 | `src/stm_data_processing/io/h5_convention.py` |
| QPI | `src/stm_data_processing/io/qpi_io.py` |
| 磁化率（含并行驱动产物） | `src/stm_data_processing/io/susceptibility_io.py` |
| 二维能带 | `src/stm_data_processing/io/ek2d_io.py` |
| 并行驱动的原子重写与 creation_date 传递 | `src/stm_data_processing/dft/wannier90/lindhard_re_chi_parallel.py:write_result_h5` |
| 证据脚本（约定 dump、往返、体积、旧文件、重写字节一致） | `var/t8/t8_evidence.py`（一次性产物，不入库） |
