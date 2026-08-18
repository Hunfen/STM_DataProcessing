# OpenMX Diff Gcube 模块接口文档

## 模块概述

`diff_gcube.py` 提供 Gaussian cube 文件（`.cube`）的读取、写入与差分功能，用于计算两个 cube 文件的逐格点差值。

**差分定义**: `output = input1 − input2`（即 `input1` 为被减数、`input2` 为减数）。

**输入格式**: 标准 Gaussian cube 文本文件（注释行、原子数/原点、网格信息、原子坐标、体数据）。

**输出格式**: 与输入同几何（网格数、原点、网格矢量、原子）的 cube 文件，体数据为逐点差值。

**CLI 入口**: 模块含 `main()`，可直接以命令行方式运行（`python diff_gcube.py input1.cube input2.cube output.cube`）。

---

## 核心类：`CubeFile`

### 类定义

```python
@dataclass
class CubeFile:
    """Represents a Gaussian cube file."""
```

### 数据字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `comment1` | `str` | 第 1 行注释 |
| `comment2` | `str` | 第 2 行注释 |
| `atom_num` | `int` | 原子数 |
| `origin` | `list[float]` | 原点坐标 `[x, y, z]` |
| `ngrid` | `list[int]` | 三轴网格点数 `[n1, n2, n3]` |
| `gtv` | `list[list[float]]` | 网格矢量 `(3, 3)` |
| `atoms` | `list[list[float]]` | 原子行 `[原子序数, x, y, z]`（写入时补 0 电荷列） |
| `data` | `list[list[list[float]]]` | 体数据，形状 `(n1, n2, n3)` |

---

### 类方法 `read`

```python
@classmethod
def read(cls, filepath: str | Path) -> Self
```

读取一个 Gaussian cube 文件。

| 参数 | 类型 | 说明 |
|------|------|------|
| `filepath` | `str \| Path` | cube 文件路径 |

**返回**: 填充完毕的 `CubeFile` 实例。

**解析约定**:

- 第 1/2 行为注释；
- 第 3 行：原子数 + 原点 3 分量；
- 第 4–6 行：各轴网格点数 + 网格矢量 3 分量；
- 第 7 行起 `atom_num` 行原子坐标（取前 5 个数值）；
- 其后所有数值按顺序填入 `(n1, n2, n3)` 的体数据。

---

### 方法 `write`

```python
def write(self, filepath: str | Path) -> None
```

将 cube 数据写回文件。

| 参数 | 类型 | 说明 |
|------|------|------|
| `filepath` | `str \| Path` | 输出 cube 文件路径 |

**写入约定**: 网格数据每行 6 个数值（科学计数法 `%13.3E`）；原子行写入 `{Z} 0.0 x y z`（第 2 列固定为 0 电荷）。

---

### 方法 `validate_compatibility`

```python
def validate_compatibility(self, other: Self) -> None
```

校验两个 cube 文件是否可用于相减。

| 参数 | 类型 | 说明 |
|------|------|------|
| `other` | `CubeFile` | 待比较的另一个 cube 文件 |

**行为**: 依次检查 `ngrid`、`origin`、`gtv` 是否一致，收集所有不兼容原因并合并进一条 `ValueError` 消息；全部一致则不抛异常。

---

## 核心函数

### `diff_cube_files`

```python
def diff_cube_files(input1: str | Path, input2: str | Path, output: str | Path) -> None
```

计算两个 cube 文件的差值并写出结果。

| 参数 | 类型 | 说明 |
|------|------|------|
| `input1` | `str \| Path` | 被减数 cube 文件 |
| `input2` | `str \| Path` | 减数 cube 文件 |
| `output` | `str \| Path` | 输出差分 cube 文件 |

**返回**: `None`（结果写入 `output` 文件）。

**处理步骤**:

1. `CubeFile.read` 读取两个文件；
2. `validate_compatibility` 校验兼容性；
3. 逐格点计算 `cube1.data − cube2.data`；
4. 以 `input1` 的几何（注释、原子数、原点、网格、原子）构造输出，写入 `output`。

---

### `main`

```python
def main() -> None
```

CLI 入口。要求命令行参数恰为 3 个（`input1 input2 output`），否则打印用法并以 `sys.exit(1)` 退出。

**用法**:

```bash
python diff_gcube.py input1.cube input2.cube output.cube
```

---

## 数学公式

差分定义（逐格点）：

```
data_out[n1][n2][n3] = data_in1[n1][n2][n3] − data_in2[n1][n2][n3]
```

对任意格点 `(n1, n2, n3)`。几何信息（`origin`、`ngrid`、`gtv`、`atoms`）完全继承自 `input1`。

---

## 使用示例

### Python API 计算差分

```python
from stm_data_processing.dft.openmx.diff_gcube import diff_cube_files

diff_cube_files("input1.cube", "input2.cube", "output.cube")
```

### 直接读写 cube 文件

```python
from stm_data_processing.dft.openmx.diff_gcube import CubeFile

cube = CubeFile.read("input1.cube")
print(cube.ngrid)      # [n1, n2, n3]
print(cube.origin)     # [x, y, z]
print(len(cube.data))  # n1

cube.write("copy.cube")
```

### 校验兼容性

```python
from stm_data_processing.dft.openmx.diff_gcube import CubeFile

c1 = CubeFile.read("input1.cube")
c2 = CubeFile.read("input2.cube")
c1.validate_compatibility(c2)   # 不兼容时抛 ValueError
```

### 命令行

```bash
python -m stm_data_processing.dft.openmx.diff_gcube input1.cube input2.cube output.cube
```

---

## 依赖项

| 依赖 | 必需 | 说明 |
|------|------|------|
| `dataclasses` | 是 | `@dataclass` 定义 `CubeFile`（标准库） |
| `pathlib` | 是 | 文件路径处理（标准库） |
| `typing` | 是 | `Self` 类型注解（标准库） |
| `logging` | 是 | 日志输出（标准库） |

---

## 错误处理

| 异常 | 触发条件 |
|------|----------|
| `ValueError` | `validate_compatibility` 发现 `ngrid`、`origin`、`gtv` 任一不一致（所有原因合并进一条消息） |
| `FileNotFoundError` | 输入的 cube 文件不存在（`Path.open` 抛出） |
| `IndexError` | cube 文件行数不足（格式非法） |

---

## 接口对齐检查清单

生成调用此模块的代码时，请确保：

- [ ] **差分语义为 `input1 − input2 = output`（非 `|input1 − input2|`）**
- [ ] **两个输入文件的 `ngrid`、`origin`、`gtv` 完全一致，否则抛 `ValueError`**
- [ ] **输出几何继承自 `input1`（注释、原子、原点、网格矢量）**
- [ ] **`CubeFile.data` 为嵌套列表 `(n1, n2, n3)`，非 NumPy 数组**
- [ ] **`write` 以每行 6 个数值、`%13.3E` 格式写出网格数据**
- [ ] **`main()` 为 CLI 入口，恰好接受 3 个位置参数**

---

## 版本信息

- 模块路径：`src/stm_data_processing/dft/openmx/diff_gcube.py`
- 日志级别：`INFO`（成功写出差分文件）
- 命令行入口：`main()`（`__main__` 下执行）
- **无环境变量配置**
- **输入格式：两个 Gaussian cube 文本文件**
- **输出格式：`input1 − input2` 的 Gaussian cube 文件**
