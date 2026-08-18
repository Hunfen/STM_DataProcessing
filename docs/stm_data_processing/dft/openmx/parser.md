# OpenMX Parser 模块接口文档

## 模块概述

`parser.py` 提供 OpenMX 输出文件（`.out` / `.dat`）的解析功能，核心为 `OpenMX` 类，用于读取并提取倒格矢、正格矢、原子种类定义、原子坐标、自旋权重与费米能级等结构信息。

**输入格式**: OpenMX 的 `.out`（输出日志）或 `.dat`（输入数据）文本文件，内部按 OpenMX 关键字块（如 `<Atoms.UnitVectors`、`<Atoms.SpeciesAndCoordinates`、`<Definition.of.Atomic.Species`）解析。

**输出格式**: 各读取方法返回 `dict` 或 `np.ndarray`，并以实例属性缓存解析结果（`avecs`、`bvecs`、`atomic_species`、`n_species`、`fermi_level`、`n_atoms`、`atomic_positions_data`）。

**日志约定**: 本模块使用标准库 `logging`（`logger = logging.getLogger(__name__)`）输出日志，不使用 `print`。

**单位约定**: 倒格矢单位为 1/Å；费米能级由 Hartree 转换为 eV（`hartree_to_ev = 27.211386245988`）；坐标默认按分数坐标（FRAC）处理。

---

## 模块常量

### `VALID_ELEMENTS`

```python
VALID_ELEMENTS: set[str]
```

有效化学元素符号集合（H → Og，共 118 个），用于原子种类定义行的合法性校验。元素符号不在该集合内的种类行会被跳过（并记录 `WARNING` 日志）。

---

## 核心类：`OpenMX`

### 类定义

```python
class OpenMX:
    """Parser for OpenMX band structure data and output files."""
```

### 构造函数

```python
def __init__(self, folder: str | None = None, systemname: str | None = None)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `folder` | `str \| None` | 存放 OpenMX 文件的目录。若为 `None`，需手动加载文件 |
| `systemname` | `str \| None` | OpenMX 系统名。若为 `None`，需手动加载文件 |

**说明**: 若同时提供 `folder` 与 `systemname`，构造时立即调用 `load_from_systemname(folder, systemname)` 设置文件路径。

### 实例属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `bvecs` | `np.ndarray \| None` | 倒格矢 `(3,3)`，单位 1/Å |
| `avecs` | `np.ndarray \| None` | 正格矢（实空间晶格矢量）`(3,3)`，单位 Å |
| `folder` | `str \| None` | OpenMX 文件目录 |
| `systemname` | `str \| None` | OpenMX 系统名 |
| `out_file` | `Path \| None` | `.out` 文件路径（`load_from_systemname` 后为 `Path`） |
| `dat_file` | `Path \| None` | `.dat` 文件路径（`load_from_systemname` 后为 `Path`） |
| `atomic_species` | `list[dict] \| None` | 解析后的有效原子种类列表（`read_openmx_file` 后填充） |
| `n_species` | `int \| None` | 有效原子种类数 |
| `n_atoms` | `float \| None` | 原子总数；无法解析坐标时为 `np.nan` |
| `atomic_positions_data` | `dict \| None` | 最近一次 `read_atomic_positions` 成功解析的坐标数据缓存 |
| `fermi_level` | `float` | 费米能级（化学势），单位 eV，初始值 `0.0` |

---

## 公开实例方法

### `load_from_systemname`

```python
def load_from_systemname(self, folder: str, systemname: str) -> None
```

根据目录与系统名设置 `.out`/`.dat` 文件路径。

| 参数 | 类型 | 说明 |
|------|------|------|
| `folder` | `str` | OpenMX 文件目录 |
| `systemname` | `str` | OpenMX 系统名 |

**行为**: 构造 `out_file = folder/systemname.out` 与 `dat_file = folder/systemname.dat`，检查文件是否存在；不存在时仅记录 `WARNING` 日志，**不抛出异常**（因为并非所有操作都需要文件）。同时更新 `self.folder` 与 `self.systemname`。

---

### `read_openmx_file`

```python
def read_openmx_file(self, fname: str | None = None) -> dict
```

读取 OpenMX `.out`/`.dat` 文件并提取各类属性，是其余读取方法的基础。

| 参数 | 类型 | 说明 |
|------|------|------|
| `fname` | `str \| None` | 文件路径。若为 `None`，依次尝试：已设置的 `out_file` → `dat_file` → 由 `folder`/`systemname` 构造的路径 |

**返回**: `dict`，键说明：

| 键 | 类型 | 说明 |
|----|------|------|
| `lines` | `list[str]` | 文件全部行（供内部解析） |
| `avecs` | `np.ndarray \| None` | 正格矢 `(3,3)`（若找到 `Atoms.UnitVectors` 块） |
| `bvecs` | `np.ndarray \| None` | 倒格矢 `(3,3)`（若找到 Reciprocal vector b1/b2/b3） |
| `species_list` | `list[dict]` | 有效原子种类列表 |
| `n_species` | `int` | 有效原子种类数 |
| `raw_lines` | `list[str]` | 种类定义块原始行 |
| `fermi_level` | `float \| None` | 费米能级（eV），未找到则为 `None` |

**副作用**: 解析成功后更新 `self.avecs`、`self.bvecs`、`self.fermi_level`、`self.atomic_species`、`self.n_species`。

---

### `read_bvecs_from_out`

```python
def read_bvecs_from_out(self, fname: str | None = None) -> np.ndarray
```

读取倒格矢，内部调用 `read_openmx_file` 以解析全部属性。

| 参数 | 类型 | 说明 |
|------|------|------|
| `fname` | `str \| None` | `.out`/`.dat` 文件路径；若为 `None` 且初始化时设置了 `out_file`/`dat_file`，则使用可用路径 |

**返回**: `np.ndarray`，形状 `(3,3)`，倒格矢 b1/b2/b3，单位 1/Å。

**行为**: 若解析结果中 `bvecs` 为 `None`，抛出 `RuntimeError`；否则更新 `self.bvecs` 并返回。

---

### `read_atomic_species_from_out`

```python
def read_atomic_species_from_out(self, fname: str | None = None) -> dict
```

读取原子种类定义，内部调用 `read_openmx_file`。

| 参数 | 类型 | 说明 |
|------|------|------|
| `fname` | `str \| None` | `.out`/`.dat` 文件路径；若为 `None` 使用可用路径 |

**返回**: `dict`，键说明：

| 键 | 类型 | 说明 |
|----|------|------|
| `species_list` | `list[dict]` | 有效种类字典列表 |
| `n_species` | `int` | 有效种类数 |
| `raw_lines` | `list[str]` | 种类定义块原始行 |

每个种类字典（`species_list` 的元素）包含：

| 键 | 类型 | 说明 |
|----|------|------|
| `label` | `str` | 原始标签（第一列） |
| `element` | `str` | 校验后的元素符号 |
| `orbitals` | `dict` | 轨道数 `{'s': int, 'p': int, 'd': int}` |
| `pseudopotential` | `str` | 赝势名（第三列） |
| `basis_info` | `str` | 基组信息串（第二列，如 `C6.0-s3p2d2`） |

---

### `read_atomic_positions`

```python
def read_atomic_positions(self, fname: str | None = None) -> dict
```

按优先级读取原子坐标：

1. `.out` 文件中“最终结构”分数坐标（最高优先级）
2. `.out` 文件中的 `Atoms.SpeciesAndCoordinates` 块
3. `.dat` 文件中的 `Atoms.SpeciesAndCoordinates` 块（最低优先级）

自旋权重仅来自来源 2、3。

| 参数 | 类型 | 说明 |
|------|------|------|
| `fname` | `str \| None` | `.out`/`.dat` 文件路径；若为 `None` 使用可用路径 |

**返回**: `dict`，键说明：

| 键 | 类型 | 形状 | 说明 |
|----|------|------|------|
| `positions_frac` | `np.ndarray` | `(n_atoms, 3)` | 分数坐标 |
| `positions_cart` | `np.ndarray \| None` | `(n_atoms, 3)` | 笛卡尔坐标（仅当 `self.avecs` 可用时计算） |
| `elements` | `list[str]` | `(n_atoms,)` | 元素符号列表 |
| `spin_weights` | `np.ndarray` | `(n_atoms, 2)` | `[up_spin, down_spin]` 权重；不可用时为 `np.nan` |
| `source` | `str` | `'final_structure'` / `'species_coordinates'` / `'dat_file'` |

**副作用**: 成功后更新 `self.atomic_positions_data` 与 `self.n_atoms`；失败则将 `self.n_atoms` 置为 `np.nan`、`self.atomic_positions_data` 置为 `None` 并抛出 `RuntimeError`。

---

## 内部辅助方法（静态）

以下静态方法供内部解析使用，通常无需直接调用：

| 方法 | 说明 |
|------|------|
| `_parse_fermi_level(lines)` | 从行中解析“Chemical Potential (Hartree)”最后一次出现的值并转换为 eV，返回 `float \| None` |
| `_parse_unit_cell_vectors(lines)` | 解析 `<Atoms.UnitVectors ... </Atoms.UnitVectors>` 块，返回正格矢 `(3,3)` 或 `None` |
| `_parse_reciprocal_vectors(lines)` | 解析 `Reciprocal vector b1/b2/b3` 行，返回倒格矢 `(3,3)` 或 `None` |
| `_parse_atomic_species(lines)` | 解析 `<Definition.of.Atomic.Species ... </Definition.of.Atomic.Species>` 块，返回 `{'species_list', 'n_species', 'raw_lines'}` |
| `_parse_vector_line(line)` | 从 `#  Reciprocal vector b1 (1/Ang):  ...` 行提取前 3 个浮点分量 |
| `_parse_orbital_basis_static(basis_info, element)` | 用正则解析基组串（如 `C6.0-s3p2d2`）得到 `{'s', 'p', 'd'}` 轨道数，失败返回 `None` |
| `_parse_final_structure_positions(lines)` | 解析“Fractional coordinates of the final structure”后续坐标，返回 `{'positions_frac', 'elements'}` |
| `_parse_species_and_coordinates(lines)` | 解析 `Atoms.SpeciesAndCoordinates` 块，返回 `{'positions_frac'/'positions_ang', 'elements', 'spin_weights'}` |

### 实例辅助方法

| 方法 | 说明 |
|------|------|
| `_create_positions_dict(positions_frac, elements, spin_weights, source)` | 组装标准化坐标字典；`spin_weights` 为 `None` 时填充 `(n_atoms, 2)` 的 `np.nan`；`self.avecs` 可用时计算 `positions_cart` |

---

## 使用示例

### 通过目录与系统名初始化并读取

```python
from stm_data_processing.dft.openmx.parser import OpenMX

mx = OpenMX(folder="./work", systemname="C6LiC6")

# 倒格矢 (3,3)，单位 1/Å
bvecs = mx.read_bvecs_from_out()
print(bvecs.shape)   # (3, 3)

# 原子种类定义
species = mx.read_atomic_species_from_out()
print(species["n_species"])
print(species["species_list"][0]["orbitals"])   # {'s': 3, 'p': 2, 'd': 2}

# 原子坐标（含自旋权重）
pos = mx.read_atomic_positions()
print(pos["positions_frac"].shape)   # (n_atoms, 3)
print(pos["elements"])               # ['C', 'Li', ...]
print(pos["source"])                 # 'final_structure'
```

### 不传路径、直接指定文件

```python
from stm_data_processing.dft.openmx.parser import OpenMX

mx = OpenMX()   # 不自动加载
result = mx.read_openmx_file("./work/C6LiC6.out")
print(result.keys())            # dict_keys(['lines', 'avecs', 'bvecs', 'species_list', 'n_species', 'raw_lines', 'fermi_level'])
print(mx.fermi_level)           # eV
print(mx.n_species)
```

### 访问费米能级与晶格矢量

```python
from stm_data_processing.dft.openmx.parser import OpenMX

mx = OpenMX(folder="./work", systemname="C6LiC6")
mx.read_openmx_file()

print("fermi_level (eV):", mx.fermi_level)
print("avecs (Å):\n", mx.avecs)
print("bvecs (1/Å):\n", mx.bvecs)
```

---

## 依赖项

| 依赖 | 必需 | 说明 |
|------|------|------|
| `numpy` | 是 | 数组构造与矩阵运算 |
| `pathlib` | 是 | 文件路径处理（标准库） |
| `re` | 是 | 基组串正则解析（标准库） |
| `logging` | 是 | 日志输出（标准库） |

---

## 错误处理

| 异常 | 触发条件 |
|------|----------|
| `ValueError` | `read_openmx_file` 中无有效文件路径且无法构造 `.out`/`.dat` |
| `FileNotFoundError` | 指定文件不存在 |
| `RuntimeError` | `read_bvecs_from_out` 中未找到倒格矢；`_parse_atomic_species` 中原子种类块缺失或 `Species.Number` 行非法/缺失；`read_atomic_positions` 中所有来源均找不到原子坐标 |

**非致命警告**（仅记录日志，不抛出）:

- `Species.Number` 与解析到的种类行数不一致
- 种类行长度不足 / 标签非有效元素 / 基组串无法解析（跳过该行）
- `load_from_systemname` 中 `.out`/`.dat` 文件不存在
- 无法解析化学势数值（跳过该行）

---

## 接口对齐检查清单

生成调用此模块的代码时，请确保：

- [ ] **`OpenMX(folder, systemname)` 会自动设置 `.out`/`.dat` 路径（不读文件内容）**
- [ ] **`read_openmx_file` 是其余读取方法的基础，会先解析全部属性**
- [ ] **`read_bvecs_from_out` 返回 `(3,3)` 倒格矢（1/Å），未找到则抛 `RuntimeError`**
- [ ] **`read_atomic_species_from_out` 返回 `{'species_list', 'n_species', 'raw_lines'}`**
- [ ] **`read_atomic_positions` 返回 `{'positions_frac', 'positions_cart', 'elements', 'spin_weights', 'source'}`**
- [ ] **`read_atomic_positions` 优先级：最终结构 > `.out` 的 SpeciesAndCoordinates > `.dat`**
- [ ] **自旋权重形状 `(n_atoms, 2)`，不可用时为 `np.nan`**
- [ ] **`fermi_level` 单位 eV；`avecs` 单位 Å；`bvecs` 单位 1/Å**
- [ ] **本模块使用 `logging` 输出日志，不使用 `print`**

---

## 版本信息

- 模块路径：`src/stm_data_processing/dft/openmx/parser.py`
- 日志级别：`INFO`（文件路径/种类解析统计）、`WARNING`（文件缺失、种类行跳过）、`ERROR`（基组解析异常）
- 单位换算常量：`hartree_to_ev = 27.211386245988`
- **无环境变量配置**
- **输入格式：OpenMX `.out` / `.dat` 文本文件**
- **输出格式：`dict` / `np.ndarray`，并通过实例属性缓存解析结果**
