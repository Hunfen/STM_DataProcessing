# OpenMX Unfolding 模块接口文档

## 模块概述

`unfolding.py` 提供 OpenMX 能带反折叠（band unfolding）权重数据的读取与谱函数计算功能，用于把超胞能带反折叠回原胞并投影到指定元素/原子，支持 CPU/GPU 双后端（GPU 依赖 CuPy，可选）。

**输入格式**: OpenMX 的 `.unfold_orbup` 权重文件（`np.loadtxt` 读取的数值矩阵），列依次为：k 路径（Bohr⁻¹）、能量（eV，已相对 E_F）、以及各轨道权重列；同时需要一个已解析原子数据的 `OpenMX` 实例用于生成轨道列名。

**输出格式**: `read_unfold_orbup` 返回带命名的 `pd.DataFrame`；`compute_spectral_function` 返回 `(k, e, a)` 三个 `np.ndarray`。

**后端选择**: 模块导入时探测 CuPy，可用则置 `CUPY_AVAILABLE = True`；`compute_spectral_function` 的 `use_gpu=True` 时在 CuPy 可用的情况下走 GPU 路径，否则回退 CPU，返回结果始终转换为 `np.ndarray`。

---

## 模块常量

| 常量 | 值 | 说明 |
|------|-----|------|
| `CUPY_AVAILABLE` | `bool` | 导入时探测 CuPy 是否可用 |
| `bohr_to_angstrom` | `0.52917721092` | Bohr→Å 换算（k 路径转换用） |
| `kb_ev_per_k` | `8.617333262145e-5` | 玻尔兹曼常数（eV/K） |
| `x0` | `ln(3 + 2√2) ≈ 1.7627` | `-∂f/∂E` 半高全宽系数 |

---

## 核心函数

### `read_unfold_orbup`

```python
def read_unfold_orbup(file_path: str, openmx_parser: OpenMX) -> pd.DataFrame
```

读取并处理 OpenMX 反折叠权重文件。

| 参数 | 类型 | 说明 |
|------|------|------|
| `file_path` | `str` | `.unfold_orbup` 文件路径 |
| `openmx_parser` | `OpenMX` | 已解析原子数据的 `OpenMX` 实例（需先调用 `read_atomic_positions` 与 `read_atomic_species_from_out`） |

**返回**: `pd.DataFrame`，列说明：

| 列 | 类型 | 说明 |
|----|------|------|
| `kpath` | `float` | k 路径值（已由 Bohr⁻¹ 转换为 Å⁻¹） |
| `energy` | `float` | 能量（eV，相对 E_F，原样保留） |
| `0-C-0s`、`0-C-0px`、… | `float` | 各轨道权重列，命名 `{atom_idx}-{element}-{orbital}` |

**处理步骤**:

1. `np.loadtxt` 读取全部数值；
2. 第 0 列 k 路径除以 `bohr_to_angstrom` 转为 Å⁻¹；
3. 第 1 列能量原样保留（已为 eV 且相对 E_F）；
4. 第 2 列起为轨道权重，用 `_generate_orbital_column_names` 生成的列名命名；
5. 校验权重列数与期望列数一致，否则抛 `ValueError`。

**轨道列命名约定**（`atom_idx` 为 0 基，遵循 Python 索引）:

- s 轨道：`0s, 1s, ...`
- p 轨道：`0px, 0py, 0pz, 1px, ...`
- d 轨道：`0d3z^2-r^2, 0dx^2-y^2, 0dxy, 0dxz, 0dyz, ...`

---

### `compute_spectral_function`

```python
def compute_spectral_function(
    df: pd.DataFrame,
    *,
    element: str | None = None,
    atom_index: int | None = None,
    nk: int = 512,
    ne: int = 512,
    delta_k_input_nm: float = 100,
    delta_e_input_k: float = 100,
    use_gpu: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]
```

由反折叠权重计算谱函数 A(k, E)。

| 参数 | 类型 | 说明 |
|------|------|------|
| `df` | `pd.DataFrame` | `read_unfold_orbup` 的输出，含 `kpath`、`energy` 与轨道权重列 |
| `element` | `str \| None` | 投影到的元素符号（如 `'C'`）。指定后仅计入该元素的轨道 |
| `atom_index` | `int \| None` | 指定原子索引（与列名中的 `atom_idx` 一致，0 基）。指定后仅计入该原子的轨道 |
| `nk` | `int` | k 轴网格点数，默认 512 |
| `ne` | `int` | 能量轴网格点数，默认 512 |
| `delta_k_input_nm` | `float` | 实空间相干长度 L（nm），默认 100 |
| `delta_e_input_k` | `float` | 温度 T（K），决定能量展宽，须满足 `0.1 ≤ T ≤ 300`，默认 100 |
| `use_gpu` | `bool` | 是否在 CuPy 可用时使用 GPU，默认 `True` |

**返回**: `(k, e, a)`，三者均为 `np.ndarray`，形状 `(nk, ne)`：

| 返回值 | 说明 |
|--------|------|
| `k` | k 路径 `meshgrid` 数组，形状 `(nk, ne)` |
| `e` | 能量 `meshgrid` 数组，形状 `(nk, ne)` |
| `a` | 谱函数 A(k, E)，形状 `(nk, ne)` |

**行为**:

- k、E 范围自动取 `df['kpath']` 与 `df['energy']` 的 `[min, max]`；
- `element` 与 `atom_index` 均为 `None` 时，对所有轨道权重求和；
- 权重列按 `_get_target_weights_from_df` 匹配（列名按 `-` 拆分为 `[atom_idx, element, orbital]`）；
- 温度越界时抛 `ValueError`。

**⚠️ 注意**: `atom_index` 为 0 基索引，与列名 `{atom_idx}-{element}-{orbital}` 中的 `atom_idx` 一致（源码 `_generate_orbital_column_names` 使用 `enumerate(elements)` 生成）。`compute_spectral_function` 的 docstring 中“1-based”描述与实际实现不符，以源码为准。

---

### `lorentzian_2d`

```python
def lorentzian_2d(k, e, k0, e0, delta_k, delta_e)
```

二维洛伦兹展宽核，对 NumPy 与 CuPy 数组通用。

| 参数 | 说明 |
|------|------|
| `k, e` | 网格坐标数组 |
| `k0, e0` | 中心点 |
| `delta_k, delta_e` | 两轴展宽半宽 |

**返回**: 与输入同形状的展宽值数组。

---

### 内部辅助函数

| 函数 | 说明 |
|------|------|
| `_generate_orbital_column_names(openmx_parser)` | 依据原子坐标与种类生成轨道权重列名，返回 `list[str]` |
| `_get_target_weights_from_df(df, element=None, atom_index=None)` | 依据列名匹配筛选权重列并沿行求和，返回 `np.ndarray` |

`_generate_orbital_column_names` 要求 `openmx_parser.atomic_positions_data` 非空（否则抛 `RuntimeError`），且 `openmx_parser.atomic_species` 非空（否则抛 `RuntimeError`）；某元素不在种类数据中时抛 `ValueError`。

---

## 数学公式

二维洛伦兹展宽核：

```
L(k, e; k0, e0, δk, δe) = 1 / ( ((k − k0)/δk)² + ((e − e0)/δe)² + 1 )
```

谱函数为对所有 (k, E) 采样点加权求和：

```
A(k, E) = Σ_i w_i · L(k, E; k_i, E_i, δk, δE)
```

动量展宽半宽由实空间相干长度导出：

```
L(Å)  = 10 × L(nm)
δk    = 2π / L(Å)      （单位 1/Å）
```

能量展宽半宽由温度导出（采用洛伦兹线型但半宽取 `-∂f/∂E` 的 HWHM）：

```
x0   = ln(3 + 2√2) ≈ 1.7627
δE   = x0 · k_B · T    ，k_B = 8.617333262145e-5 eV/K
```

k 路径单位换算：

```
k(1/Å) = k(Bohr⁻¹) / 0.52917721092
```

---

## 使用示例

### 读取反折叠权重并计算谱函数

```python
from stm_data_processing.dft.openmx.parser import OpenMX
from stm_data_processing.dft.openmx.unfolding import read_unfold_orbup, compute_spectral_function

# 1. 解析原子数据（列名生成所需）
mx = OpenMX(folder="./work", systemname="C6LiC6")
mx.read_atomic_species_from_out()
mx.read_atomic_positions()

# 2. 读取权重
df = read_unfold_orbup("./work/C6LiC6.unfold_orbup", mx)
print(df.columns.tolist())   # ['kpath', 'energy', '0-C-0s', ...]

# 3. 计算谱函数
k, e, a = compute_spectral_function(df)
print(k.shape, e.shape, a.shape)   # (512, 512) (512, 512) (512, 512)
```

### 投影到指定元素/原子

```python
from stm_data_processing.dft.openmx.unfolding import compute_spectral_function

# 仅投影到 C 元素
k, e, a_C = compute_spectral_function(df, element="C")

# 仅投影到第 0 号原子
k, e, a_atom0 = compute_spectral_function(df, atom_index=0)
```

### 调整展宽与网格

```python
k, e, a = compute_spectral_function(
    df,
    nk=1024,
    ne=1024,
    delta_k_input_nm=50,   # 相干长度 50 nm
    delta_e_input_k=10,    # 温度 10 K
    use_gpu=False,         # 强制 CPU
)
```

### 绘制谱函数

```python
import matplotlib.pyplot as plt
from stm_data_processing.dft.openmx.unfolding import compute_spectral_function

k, e, a = compute_spectral_function(df, element="C")

plt.pcolormesh(k[:, 0], e[0, :], a.T, shading="auto", cmap="hot")
plt.xlabel("k (1/Å)")
plt.ylabel("E − E_F (eV)")
plt.colorbar(label="A(k, E)")
plt.show()
```

---

## 依赖项

| 依赖 | 必需 | 说明 |
|------|------|------|
| `numpy` | 是 | 数组运算、`loadtxt`、`meshgrid` |
| `pandas` | 是 | DataFrame 存储权重 |
| `cupy` | 否 | GPU 加速（可选，缺失时自动回退 CPU） |
| `.parser.OpenMX` | 是 | 生成轨道列名所需 |

---

## 错误处理

| 异常 | 触发条件 |
|------|----------|
| `ValueError` | `compute_spectral_function` 中 `delta_e_input_k`（温度）超出 `[0.1, 300]` K；`_generate_orbital_column_names` 中元素不在种类数据中；`read_unfold_orbup` 中权重列数与期望列数不一致；`_get_target_weights_from_df` 中无列匹配且指定了 `element`/`atom_index` |
| `RuntimeError` | `_generate_orbital_column_names` 中 `atomic_positions_data` 或 `atomic_species` 未就绪（需先调用 `read_atomic_positions` / `read_atomic_species_from_out`） |

---

## 接口对齐检查清单

生成调用此模块的代码时，请确保：

- [ ] **调用 `read_unfold_orbup` 前，`OpenMX` 已调用 `read_atomic_species_from_out` 与 `read_atomic_positions`**
- [ ] **`read_unfold_orbup` 返回 DataFrame，列为 `kpath`、`energy` 加轨道权重列**
- [ ] **`kpath` 单位已转为 Å⁻¹；`energy` 已相对 E_F（原样保留）**
- [ ] **`compute_spectral_function` 返回 `(k, e, a)`，三者形状均为 `(nk, ne)`**
- [ ] **`atom_index` 为 0 基索引，与列名 `{atom_idx}-...` 一致**
- [ ] **温度参数 `delta_e_input_k` 必须在 `[0.1, 300]` K 内**
- [ ] **`use_gpu=True` 且 CuPy 不可用时自动回退 CPU，返回仍为 `np.ndarray`**

---

## 版本信息

- 模块路径：`src/stm_data_processing/dft/openmx/unfolding.py`
- 后端检测：导入时探测 CuPy（`CUPY_AVAILABLE`）
- 日志级别：无（静默计算）
- 单位常量：`bohr_to_angstrom = 0.52917721092`、`kb_ev_per_k = 8.617333262145e-5`、`x0 ≈ 1.7627`
- **无环境变量配置**
- **输入格式：OpenMX `.unfold_orbup` 数值文件 + 已解析的 `OpenMX` 实例**
- **输出格式：`pd.DataFrame`（权重）/ `(k, e, a)` 三个 `np.ndarray`（谱函数）**
