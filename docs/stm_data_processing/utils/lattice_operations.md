# 晶格几何操作模块接口文档

## 模块概述

模块路径：`src/stm_data_processing/utils/lattice_operations.py`

`lattice_operations.py` 提供基于 `LATTICE` 实例的几何操作：C3 对称倒格矢生成、超胞 Bragg 点折入 1x1 折叠窗、圆内 Bragg 点枚举、六角晶格第一布里渊区（1st BZ）顶点计算，以及平面多边形掩膜生成等工具。

**数据约定**：

- 2D 向量统一使用 **列向格式** `(2, N)` 返回（每列是一个二维向量），除非单独说明。
- 倒格矢的 xy 分量取自 `lattice.b1[:2]`、`lattice.b2[:2]`（只使用前两维）。

---

## 核心类：`LatticeOperations`

```python
class LatticeOperations:
    """Class for geometric operations on LATTICE instances."""
```

### 构造函数

```python
def __init__(self, lattice)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `lattice` | `object` | 提供晶格接口的对象（通常是 `LATTICE` 实例），需含 `bvecs`、`b1`、`b2` 等属性 |

实例属性 `self.lattice` 保存传入的晶格对象。

> `LATTICE` 构造时会自动创建 `self.ops = LatticeOperations(self)`，通常无需手动实例化。

---

### 实例方法

| 方法 | 签名 | 返回 | 说明 |
|------|------|------|------|
| `extend_vecs_c3` | `extend_vecs_c3(include_neg=True, tol=1e-10, sort=False)` | `(2, n)` | C3 对称生成倒格矢 |
| `get_bragg_points_supercell_in_1x1_fftshift` | `(transformation_matrix, include_origin=False, return_uv=False, return_mn=False, tol=1e-10)` | `(2, N)` 及可选 `uv`/`mn` | 超胞 Bragg 点折入 1x1 折叠窗 |
| `get_bragg_points_in_circle` | `get_bragg_points_in_circle(q_max, include_origin=True)` | `(2, N)` | 圆内 Bragg 点 |
| `get_1stbz_vertices` | `get_1stbz_vertices(tol=1e-10)` | `(2, 6)` | 六角晶格第一布里渊区顶点 |

内部辅助方法：`_sort_vectors_clockwise`、`_sort_polygon_clockwise`、`_wrap_centered`。

#### `extend_vecs_c3(include_neg=True, tol=1e-10, sort=False)`

对原胞倒格矢 `b1`、`b2` 施加 C3（120°/240° 旋转）对称，生成全部倒格矢。

```python
def extend_vecs_c3(self, include_neg=True, tol=1e-10, sort=False)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `include_neg` | `bool` | `True` | 是否包含 `-G` 向量 |
| `tol` | `float` | `1e-10` | 去重容差 |
| `sort` | `bool` | `False` | 为 `True` 时按顺时针排序 |

**返回**：`ndarray (2, n)`，C3 对称生成的全部唯一倒格矢（列向格式，每列为二维向量）。`include_neg=False` 且六角晶格时通常得到 6 个向量。

#### `get_bragg_points_supercell_in_1x1_fftshift(...)`

生成**全部**超胞 Bragg 点并折入 1x1 的 FFTshift 窗口 `[-0.5, 0.5)`。

```python
def get_bragg_points_supercell_in_1x1_fftshift(
    self,
    transformation_matrix,
    include_origin=False,
    return_uv=False,
    return_mn=False,
    tol=1e-10,
)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `transformation_matrix` | `array-like (2,2)` | — | 实空间超胞变换矩阵 `T`（行向基） |
| `include_origin` | `bool` | `False` | 是否包含 Γ (0,0) |
| `return_uv` | `bool` | `False` | 同时返回原胞倒格基下的折叠约化坐标 `(u, v)` |
| `return_mn` | `bool` | `False` | 同时返回超胞倒格基下的整数系数 `(m, n)` |
| `tol` | `float` | `1e-10` | 边界与去重容差 |

**返回**：

- 默认：`points`，`(2, N)`，Bragg 点的笛卡尔坐标 `(qx, qy)`（列向）。
- `return_uv=True` 时追加 `uv`：`(2, N)`，原胞倒格基下的折叠约化坐标。
- `return_mn=True` 时追加 `mn`：`(2, N)`，超胞倒格基下的整数 `(m, n)`（dtype=int）。
- 多个返回时返回 `tuple`（顺序：`points, [uv], [mn]`）。

#### `get_bragg_points_in_circle(q_max, include_origin=True)`

枚举半径 `q_max` 圆内的所有 Bragg 点。

```python
def get_bragg_points_in_circle(self, q_max, include_origin=True)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `q_max` | `float` | — | 包含 Bragg 点的最大模长（半径） |
| `include_origin` | `bool` | `True` | 是否包含原点（Γ 点，h=0, k=0） |

**返回**：`(2, N)`，按模长升序排列的 Bragg 点笛卡尔坐标（列向）。无点时返回 `(2, 0)` 空数组。

#### `get_1stbz_vertices(tol=1e-10)`

计算六角晶格第一布里渊区的六个顶点。

```python
def get_1stbz_vertices(self, tol=1e-10)
```

**返回**：`(2, 6)`，六个顶点（列向），按顺时针排序。若 `extend_vecs_c3` 未返回 6 个向量则抛出 `ValueError`。

---

### 内部辅助方法

| 方法 | 说明 |
|------|------|
| `_sort_vectors_clockwise(vectors)` | `(n, 2)` 行向向量按顺时针排序 |
| `_sort_polygon_clockwise(polygon)` | `(2, N)` 多边形顶点按质心角顺时针排序 |
| `_wrap_centered(x)` | 将 `x` 映射到 `[-0.5, 0.5)`（`x - floor(x + 0.5)`） |

---

## 模块级函数

### `create_polygon_mask(qx_grid, qy_grid, polygon, eps=1e-12)`

生成多边形内部（含边界）点的布尔掩膜。

```python
def create_polygon_mask(qx_grid, qy_grid, polygon, eps=1e-12)
```

| 参数 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `qx_grid` | `array-like` | `(nx, ny)` | x 坐标网格 |
| `qy_grid` | `array-like` | `(nx, ny)` | y 坐标网格 |
| `polygon` | `array-like` | `(2, M)` | 多边形顶点（列向） |
| `eps` | `float` | — | 边界检查容差，默认 `1e-12` |

**返回**：`ndarray (nx, ny)` 布尔掩膜（多边形内/边界为 `True`）。

实现：先按质心角顺时针排序顶点，再用边界点判定 + 射线法（ray casting）判断内部点。

### 内部函数

| 函数 | 说明 |
|------|------|
| `_sort_polygon_clockwise_external(polygon)` | 独立版本的顺时针多边形顶点排序 |
| `_point_on_segment_external(px, py, x1, y1, x2, y2, eps)` | 判断点是否在线段上（含容差） |

---

## 数学公式

### C3 旋转矩阵（用于 `extend_vecs_c3`）

```python
rot(θ) = [[ cos θ, -sin θ ],
          [ sin θ,  cos θ ]]

r120 = rot(2π/3)    # 120°
r240 = rot(4π/3)    # 240°
```

生成的向量集合为对种子 `{b1, b2}`（及可选 `{-b1, -b2}`）分别施加 `I, r120, r240` 后去重。

### 超胞倒格基变换

原胞倒格基（行向）记为 `B`（由 `b1_xy`、`b2_xy` 堆叠），超胞倒格基为：

```
B' = B @ (T^{-1})^T
```

候选倒格矢 `g = (m, n) @ B'` 折入原胞倒格基下的约化坐标 `uv = B^{-1} @ g`，再经 `_wrap_centered` 折叠到 `[-0.5, 0.5)`。

### 中心折叠

```
_wrap_centered(x) = x - floor(x + 0.5)   → x ∈ [-0.5, 0.5)
```

### 点是否在多边形内（`create_polygon_mask`）

- 边界判定：点 `P` 在线段 `(x1,y1)-(x2,y2)` 上，当且仅当叉积 `(py-y1)(x2-x1)-(px-x1)(y2-y1)` 为零（容差内）且投影落在区间内。
- 内部判定：射线法，统计从点出发的射线与多边形各边相交次数，奇数次在多边形内。

---

## 使用示例

### 六角晶格第一布里渊区顶点

```python
import numpy as np
from stm_data_processing.utils.lattice import LATTICE

a = 2.46
avecs = np.array([
    [a, 0.0, 0.0],
    [-a / 2, a * np.sqrt(3) / 2, 0.0],
    [0.0, 0.0, 10.0],
])
lat = LATTICE(avecs=avecs)

# 第一布里渊区六个顶点（列向 (2, 6)）
bz_vertices = lat.ops.get_1stbz_vertices()
print(bz_vertices.shape)  # (2, 6)
```

### C3 对称倒格矢

```python
gs = lat.ops.extend_vecs_c3(include_neg=False, sort=True)
print(gs.shape)  # (2, 6)
```

### 圆内 Bragg 点

```python
pts = lat.ops.get_bragg_points_in_circle(q_max=1.0, include_origin=False)
print(pts.shape)  # (2, N)
```

### 超胞 Bragg 点折入 1x1 折叠窗

```python
T = np.array([[2, 0], [0, 2]])
points, uv, mn = lat.ops.get_bragg_points_supercell_in_1x1_fftshift(
    T, include_origin=False, return_uv=True, return_mn=True
)
```

### 多边形掩膜

```python
from stm_data_processing.utils.lattice_operations import create_polygon_mask

x = np.linspace(-1, 1, 200)
qx, qy = np.meshgrid(x, x)
hexagon = bz_vertices  # (2, 6)
mask = create_polygon_mask(qx, qy, hexagon)
print(mask.shape)  # (200, 200)
```

---

## 依赖项

| 依赖 | 必需 | 说明 |
|------|------|------|
| `numpy` | 是 | 核心数组运算 |

---

## 错误处理

| 异常 | 触发条件 |
|------|----------|
| `ValueError` | `bvecs` 未初始化（`lattice.bvecs is None`） |
| `ValueError` | `transformation_matrix` 不是 2x2 或奇异 |
| `ValueError` | 原胞倒格基 `(b1, b2)` 在 xy 平面奇异 |
| `ValueError` | `q_max < 0` |
| `ValueError` | 倒格矢模长近零（`get_bragg_points_in_circle`） |
| `ValueError` | `get_1stbz_vertices` 未得到 6 个倒格矢 |
| `ValueError` | `polygon` 形状非 `(2, N)` 或顶点数 < 3 |
| `ValueError` | `qx_grid` 与 `qy_grid` 形状不一致 |
| `RuntimeError` | 解 BZ 顶点线性方程失败 |

---

## 接口对齐检查清单

生成调用此模块的代码时，请确保：

- [ ] **2D 向量统一使用列向格式 `(2, N)`（每列一个向量）**
- [ ] **倒格矢 xy 分量取自 `lattice.b1[:2]`、`lattice.b2[:2]`**
- [ ] **超胞变换矩阵 `T` 为 2x2（实空间、行向基），超胞倒格基 `B' = B @ (T^{-1})^T`**
- [ ] **`get_bragg_points_supercell_in_1x1_fftshift` 默认不含原点，折叠窗为 `[-0.5, 0.5)`**
- [ ] **`get_bragg_points_in_circle` 默认包含原点，结果按模长升序**
- [ ] **`create_polygon_mask` 的多边形为 `(2, M)` 列向，自动按质心角顺时针排序**
- [ ] **`get_1stbz_vertices` 仅适用于六角晶格，返回 `(2, 6)`**

---

## 版本信息

- 模块路径：`src/stm_data_processing/utils/lattice_operations.py`
- 数据约定：2D 向量列向 `(2, N)`；倒格矢 xy 分量截取前两维
- 由 `LATTICE.ops` 自动挂载，也可独立实例化 `LatticeOperations(lattice)`
