# 高精度晶格运算模块接口文档

## 模块概述

模块路径：`src/stm_data_processing/utils/lattice.py`

`lattice.py` 提供晶格容器类 `LATTICE`，用于表示三维晶格的正空间（实空间）与倒空间（倒格矢）向量，并支持高精度互求、超胞/子胞变换、绕 z 轴旋转以及一致性校验等操作。内部使用 `mpmath` 做任意精度计算，输入输出保持 `numpy.float64` 兼容。

**精度约定**：默认计算精度为 **50 位十进制有效数字**，由 `_PrecisionConfig` 管理，可通过 `LATTICE.set_precision()` 调整。

**存储约定（row-wise）**：所有晶格向量（实空间与倒空间）均按 **行向量** 存储为 `(3, 3)` 数组：`avecs[i] = a_{i+1}`、`bvecs[i] = b_{i+1}`（第 i 行是第 i+1 个向量）。

**变换约定**：对于整数变换矩阵 `M`，以晶格向量为行的 `(3, 3)` 矩阵 `A` 满足：

```
A_super = M.T @ A_prim          # supercell(M): A_new = M.T      @ A_old
A_sub   = inv(M).T @ A_old      # subcell(M):   A_new = inv(M).T @ A_old
```

该约定对实空间（`avecs`）与倒空间（`bvecs`）向量一致适用。

---

## 核心类

### 精度配置类 `_PrecisionConfig`

```python
class _PrecisionConfig:
    """Precision configuration manager"""
```

| 成员 | 类型 | 说明 |
|------|------|------|
| `_dps` | `int` | 类属性，默认 50 位十进制精度 |
| `get_dps()` | `classmethod` | 返回当前精度 `_dps` |
| `set_dps(dps)` | `classmethod` | 设置精度并同步 `mp.mp.dps` |

### 主类 `LATTICE`

```python
class LATTICE:
    """Lattice container with a consistent crystallographic convention."""
```

类属性 `_precision_dps = 50`。

#### 构造函数

```python
def __init__(self, avecs=None, bvecs=None, degree=0.0)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `avecs` | `array_like \| None` | `None` | 实空间晶格向量，`(3, 3)` 行向量 |
| `bvecs` | `array_like \| None` | `None` | 倒空间晶格向量，`(3, 3)` 行向量 |
| `degree` | `float` | `0.0` | 绕 z 轴旋转角（度，逆时针） |

构造规则：

- `avecs` 与 `bvecs` **至少提供一个**，否则抛出 `ValueError`。
- 仅给 `avecs` 时，用高精度正倒格互求得到 `bvecs`；仅给 `bvecs` 时同理得到 `avecs`。
- 同时给出时，会用 `_real_to_reciprocal(avecs)` 校验两者是否满足 `a_i · b_j = 2π δ_ij`，不满足抛出 `ValueError`。
- 若 `degree != 0`，构造完成后对实空间向量做绕 z 轴原地旋转，并重算倒格矢。

#### 实例属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `avecs` | `np.ndarray` | 实空间晶格向量，`(3, 3)` 行向量 |
| `bvecs` | `np.ndarray` | 倒空间晶格向量，`(3, 3)` 行向量 |
| `degree` | `float` | 旋转角（度） |
| `radian` | `float` | 旋转角（弧度，`np.deg2rad(degree)`） |
| `ops` | `LatticeOperations` | 关联的几何操作对象 |

#### 只读属性（property）

| 属性 | 类型 | 说明 |
|------|------|------|
| `a` | `np.ndarray (3,)` | 各实空间基矢长度 `‖aᵢ‖` |
| `b` | `np.ndarray (3,)` | 各倒格矢长度 `‖bᵢ‖` |
| `a1` / `a2` / `a3` | `np.ndarray (3,)` | 实空间基矢（`avecs` 第 0/1/2 行） |
| `b1` / `b2` / `b3` | `np.ndarray (3,)` | 倒格矢（`bvecs` 第 0/1/2 行） |
| `volume` | `float` | 晶胞体积 `a1 · (a2 × a3)` |
| `reciprocal_volume` | `float` | 倒空间体积 `b1 · (b2 × b3)` |

---

### 类方法

#### `set_precision(dps)`

设置全局计算精度（十进制有效数字）。

```python
@classmethod
def set_precision(cls, dps)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `dps` | `int` | 十进制有效数字，推荐 50–100；超高精度 100–200 |

该方法是类级全局设置：更新 `_precision_dps` 并同步 `mp.mp.dps`。

#### `get_precision()`

返回当前精度设置（十进制有效数字）。

```python
@classmethod
def get_precision(cls) -> int
```

---

### 实例方法

| 方法 | 签名 | 说明 |
|------|------|------|
| `supercell` | `supercell(transformation_matrix)` | 由当前晶胞生成超胞，返回新 `LATTICE` |
| `subcell` | `subcell(transformation_matrix)` | 由超胞恢复原胞/子胞，返回新 `LATTICE` |
| `rotate` | `rotate(degree)` | 返回绕 z 轴旋转后的新 `LATTICE` |
| `verify_consistency` | `verify_consistency(atol=1e-12)` | 校验 `a_i · b_j = 2π δ_ij`，返回 `bool` |
| `get_transform_error` | `get_transform_error(original_lattice)` | 计算与原始晶格的变换误差，返回 `float` |

#### `supercell(transformation_matrix)`

```python
def supercell(self, transformation_matrix)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `transformation_matrix` | `array_like` | 2x2 或 3x3 整数变换矩阵 `M` |

**返回**：新的 `LATTICE`（超胞）。计算规则为 `A_new = M.T @ A_old`（行向量约定，mpmath 高精度）。2x2 矩阵会被自动嵌入 xy 平面提升为 3x3（z 方向不变）。

#### `subcell(transformation_matrix)`

```python
def subcell(self, transformation_matrix)
```

**返回**：新的 `LATTICE`（子胞）。计算规则为 `A_new = inv(M).T @ A_old`（mpmath 高精度求逆）。

#### `rotate(degree)`

```python
def rotate(self, degree)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `degree` | `float` | 旋转角（度，绕 z 轴逆时针） |

**返回**：新的 `LATTICE`，行向量按 `v' = v @ R.T` 旋转（mpmath 高精度）。

#### `verify_consistency(atol=1e-12)`

```python
def verify_consistency(self, atol=1e-12) -> bool
```

计算 `avecs @ bvecs.T`，与 `2π·I` 比较（`np.allclose(atol, rtol)`）。通过返回 `True`，否则抛出 `ValueError`（含最大误差）。

#### `get_transform_error(original_lattice)`

```python
def get_transform_error(self, original_lattice) -> float
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `original_lattice` | `LATTICE` | 原始晶格对象 |

**返回**：最大相对误差（`max(|Δa| / (|a_orig| + 1e-15))`），用于验证往返变换精度。

---

### 内部方法（不建议直接调用）

| 方法 | 说明 |
|------|------|
| `_init_vectors(avecs, bvecs)` | 初始化向量并校验正倒格一致性 |
| `_validate_matrix` / `_validate_transformation_matrix` | 静态方法，校验矩阵非奇异 |
| `_promote_transform(T)` | 静态方法，2x2 变换矩阵提升为 3x3 |
| `_np_to_mp` / `_mp_to_np` | 静态方法，numpy ↔ mpmath 矩阵转换 |
| `_real_to_reciprocal` / `_reciprocal_to_real` | 静态方法，正倒格互求（高精度） |
| `_apply_rotation_inplace(degree)` | 绕 z 轴原地旋转并重算倒格矢 |

---

## 数学公式

### 正倒格矢关系

正倒格矢满足：

```
a_i · b_j = 2π δ_ij
```

### 倒格矢计算公式（实空间 → 倒空间）

```
Ω = a1 · (a2 × a3)

b1 = 2π (a2 × a3) / Ω
b2 = 2π (a3 × a1) / Ω
b3 = 2π (a1 × a2) / Ω
```

### 实空间向量反推公式（倒空间 → 实空间）

```
Ω_rec = b1 · (b2 × b3)

a1 = 2π (b2 × b3) / Ω_rec
a2 = 2π (b3 × b1) / Ω_rec
a3 = 2π (b1 × b2) / Ω_rec
```

### 体积关系

```
Ω · Ω_rec = (2π)³
```

### 变换矩阵

```
A_super = M.T @ A_prim            （超胞）
A_sub   = inv(M).T @ A_old        （子胞）
```

### 旋转矩阵（绕 z 轴逆时针，行向量约定 v' = v @ R.T）

```
R = [[ cos θ, -sin θ, 0 ],
     [ sin θ,  cos θ, 0 ],
     [ 0,       0,     1 ]]
```

---

## 使用示例

### 由实空间基矢构造晶格

```python
import numpy as np
from stm_data_processing.utils.lattice import LATTICE

# 六方晶格（石墨烯类，a=2.46 Å）
a = 2.46
avecs = np.array([
    [a, 0.0, 0.0],
    [-a / 2, a * np.sqrt(3) / 2, 0.0],
    [0.0, 0.0, 10.0],
])
lat = LATTICE(avecs=avecs)

print(lat.bvecs)      # 倒格矢 (3, 3)
print(lat.volume)     # 晶胞体积
lat.verify_consistency()  # True
```

### 由倒格矢构造晶格

```python
lat2 = LATTICE(bvecs=lat.bvecs)
print(np.allclose(lat2.avecs, lat.avecs))
```

### 超胞 / 子胞变换

```python
# 2x2 变换矩阵（嵌入 xy 平面）
M = np.array([[2, 0], [0, 2]])
super_lat = lat.supercell(M)   # A_new = M.T @ A_old
sub_lat = super_lat.subcell(M) # 恢复原胞

# 往返变换误差
err = sub_lat.get_transform_error(lat)
print(f"round-trip error: {err:.2e}")
```

### 旋转与精度设置

```python
rot_lat = lat.rotate(30.0)     # 绕 z 轴逆时针旋转 30°
LATTICE.set_precision(100)     # 提升精度到 100 位
print(LATTICE.get_precision()) # 100
```

### 字符串表示

```python
print(lat)         # 多行文本，列出 a1..a3, b1..b3, 体积, 旋转角
print(repr(lat))   # LATTICE(precision=50dps, volume=..., degree=...)
```

---

## 依赖项

| 依赖 | 必需 | 说明 |
|------|------|------|
| `mpmath` | 是 | 任意精度正倒格互求、求逆、旋转 |
| `numpy` | 是 | 数组存储与 `float64` 输入输出 |
| `stm_data_processing.utils.lattice_operations.LatticeOperations` | 是 | 构造时挂载到 `LATTICE.ops` |

---

## 错误处理

| 异常 | 触发条件 |
|------|----------|
| `ValueError` | `avecs` 与 `bvecs` 均为 `None` |
| `ValueError` | 输入矩阵不是 3x3、近奇异（行列式 < 1e-12） |
| `ValueError` | 实空间/倒空间向量线性相关（体积 < 1e-40） |
| `ValueError` | 同时给出的 `avecs` 与 `bvecs` 不满足 `a_i · b_j = 2π δ_ij` |
| `ValueError` | `transformation_matrix` 不是 2x2/3x3 或奇异 |
| `ValueError` | `verify_consistency` 校验失败（附带最大误差） |
| `ValueError` | 实空间向量未初始化时调用 `rotate` |

---

## 接口对齐检查清单

生成调用此模块的代码时，请确保：

- [ ] **`avecs` / `bvecs` 为 `(3, 3)` 数组，且按行向量存储（第 i 行为第 i 个向量）**
- [ ] **`avecs` 与 `bvecs` 至少提供一个，且必须满足 `a_i · b_j = 2π δ_ij`**
- [ ] **超胞变换满足 `A_new = M.T @ A_old`，子胞满足 `A_new = inv(M).T @ A_old`**
- [ ] **2x2 变换矩阵自动嵌入 xy 平面，z 方向保持不变**
- [ ] **旋转为绕 z 轴逆时针（度），行向量按 `v' = v @ R.T` 变换**
- [ ] **默认精度 50 位十进制，`set_precision` 为全局类级设置**
- [ ] **输入输出均为 `numpy.float64`，内部计算使用 mpmath 高精度**
- [ ] 构造时 `degree` 参数会原地旋转实空间向量并重算倒格矢

---

## 版本信息

- 模块路径：`src/stm_data_processing/utils/lattice.py`
- 默认精度：50 位十进制（`_PrecisionConfig._dps = 50`、`LATTICE._precision_dps = 50`）
- 存储约定：实空间与倒空间向量均为 `(3, 3)` 行向量
- 变换约定：`supercell(M)` → `M.T @ A`，`subcell(M)` → `inv(M).T @ A`
- 依赖 `lattice_operations.LatticeOperations` 提供几何操作
