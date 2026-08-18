# 后端配置模块接口文档

## 模块概述

`config.py` 是 `stm_data_processing` 包的后端配置中心，负责 CPU（NumPy）/ GPU（CuPy）双后端的检测、切换与查询。所有计算模块（如 `mlwf_hamiltonian`、`qpi_jdos` 等）通过本模块获取当前激活的后端。

**核心机制**:
- 模块导入时自动探测 CuPy 是否可导入且存在可用 CUDA 设备，据此初始化全局常量 `BACKEND`（`"gpu"` 或 `"cpu"`）。
- 提供 `get_xp()` 统一返回当前后端对应的数组模块（`cupy` 或 `numpy`），是**推荐**的获取后端模块方式。
- 提供 `set_backend()` 手动切换后端，但**必须在导入任何计算模块之前调用**。

---

## 全局常量与状态

### 公开常量：`BACKEND`

```python
BACKEND: Literal["cpu", "gpu"] = _detect_backend()
```

| 常量 | 类型 | 说明 |
|------|------|------|
| `BACKEND` | `Literal["cpu", "gpu"]` | 当前激活的计算后端，模块导入时由 `_detect_backend()` 确定 |

**注意**:
- 初始值在模块导入时确定：CuPy 可用且有 CUDA 设备 → `"gpu"`，否则 → `"cpu"`。
- 可通过 `set_backend()` 在运行时修改（全局变量）。

### 模块级 `logger`

```python
logger = logging.getLogger(__name__)
```

模块导入时记录一条 `INFO` 日志：`[Config] backend={BACKEND} (cupy_usable={_cupy_usable()})`。

---

## 内部函数

| 函数 | 说明 |
|------|------|
| `_cupy_usable() -> bool` | 返回 CuPy 是否**可导入**且 CUDA 设备数 > 0；任何异常均返回 `False` |
| `_detect_backend() -> Literal["cpu", "gpu"]` | 根据 `_cupy_usable()` 决定后端：可用 → `"gpu"`，否则 → `"cpu"` |

---

## 核心函数

### `set_backend(backend)`

手动设置计算后端。

```python
def set_backend(backend: Literal["cpu", "gpu", "auto"]) -> None
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `backend` | `Literal["cpu", "gpu", "auto"]` | `'cpu'` 强制 CPU；`'gpu'` 强制 GPU（不可用时回退 CPU）；`'auto'` 自动探测 |

**⚠️ 重要约束**: 必须在导入任何计算模块（如 `mlwf_hamiltonian`、`qpi_jdos`）**之前**调用，因为这些模块在导入时读取 `BACKEND` 并据此绑定数组后端。

**GPU 回退逻辑**:
- 请求 `"gpu"` 但 `_cupy_usable()` 为 `False` 时，记录一条 `WARNING` 日志 `[Config] GPU requested but CuPy is not usable. Falling back to CPU.`，并将 `BACKEND` 设为 `"cpu"`。
- 请求 `"auto"` 时直接调用 `_detect_backend()`，记录 `[Config] backend set to auto -> {BACKEND}`。
- 非法值（非 `'cpu'`/`'gpu'`/`'auto'`）抛出 `ValueError`。

```python
from stm_data_processing.config import set_backend
set_backend("cpu")  # 强制 CPU
```

### `get_backend()`

获取当前后端设置。

```python
def get_backend() -> Literal["cpu", "gpu"]
```

| 返回 | 说明 |
|------|------|
| `Literal["cpu", "gpu"]` | 当前 `BACKEND` 的值 |

### `get_xp()`

获取当前后端对应的数组模块（**推荐**方式）。

```python
def get_xp()
```

| 返回 | 说明 |
|------|------|
| `module` | 若 `BACKEND == "gpu"` 且 CuPy 可用 → `cupy`；否则 → `numpy` |

```python
from stm_data_processing.config import get_xp
xp = get_xp()
array = xp.array([1, 2, 3])  # CPU/GPU 通用
```

### `get_cupy()`

获取 CuPy 模块（**已弃用**，新代码请使用 `get_xp()`）。

```python
def get_cupy() -> Optional
```

| 返回 | 说明 |
|------|------|
| `cupy module` 或 `None` | 若 `BACKEND == "gpu"` 且 CuPy 可用 → `cupy`，否则 → `None` |

### `is_gpu_available()`

检查 GPU 后端当前是否可用。

```python
def is_gpu_available() -> bool
```

| 返回 | 说明 |
|------|------|
| `bool` | `BACKEND == "gpu"` 且 `_cupy_usable()` 为 `True` 时返回 `True` |

### `get_backend_status()`

获取详细的后端状态信息。

```python
def get_backend_status() -> dict
```

**返回字典包含以下键**:

| 键 | 类型 | 说明 |
|----|------|------|
| `backend` | `str` | 当前后端（`'cpu'` 或 `'gpu'`） |
| `cupy_importable` | `bool` | CuPy 是否可导入（模块导入时探测结果 `_CUPY_IMPORT_OK`） |
| `cuda_devices` | `int` | 可用 CUDA 设备数量（探测失败时为 `0`） |
| `cupy_version` | `str` 或 `None` | CuPy 版本字符串（不可用时为 `None`） |

---

## 核心类：`BackendArray`

后端无关的数组操作辅助类，为 CPU/GPU 提供统一接口。

### 类定义

```python
class BackendArray:
    """Helper class for backend-aware array operations."""
```

### 构造函数

```python
def __init__(self, backend: Literal["cpu", "gpu", "auto"] = "auto")
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `backend` | `Literal["cpu", "gpu", "auto"]` | `"auto"` | 后端选择；`"auto"` 时使用全局 `BACKEND` |

**内部属性**:

| 属性 | 类型 | 说明 |
|------|------|------|
| `backend` | `str` | 实例后端（`"auto"` 时解析为全局 `BACKEND`） |
| `xp` | `module` | 后端数组模块：`backend == "gpu"` 时经 `get_xp()` 获取（可能仍为 numpy，见下），否则为 `numpy` |

**注意**: `xp` 的赋值逻辑为 `get_xp() if self.backend == "gpu" else _numpy_module`。若显式传入 `backend="gpu"` 但 CuPy 不可用，`get_xp()` 会回退返回 `numpy`，即 `self.backend` 标记为 `"gpu"` 而 `self.xp` 实际为 `numpy`。

### 实例方法

#### `array(data, **kwargs)`

在当前后端创建数组。

```python
def array(self, data, **kwargs)
```

返回 `self.xp.array(data, **kwargs)`。

#### `asarray(data, **kwargs)`

将数据转换为当前后端的数组。

```python
def asarray(self, data, **kwargs)
```

返回 `self.xp.asarray(data, **kwargs)`。

#### `to_cpu(arr)`

将数组转换为 NumPy（已在 CPU 上则为无操作，原样返回）。

```python
def to_cpu(self, arr)
```

若 `arr` 为 CuPy `ndarray`，返回 `cupy.asnumpy(arr)`；否则直接返回 `arr`。

#### `to_gpu(arr)`

将数组转换为 CuPy（已在 GPU 上则为无操作）。

```python
def to_gpu(self, arr)
```

若 CuPy 不可用（`_cupy_module is None`），抛出 `RuntimeError("CuPy not available")`；否则返回 `cupy.asarray(arr)`。

#### `__getattr__(name)`

将未知属性访问委托给后端模块。

```python
def __getattr__(self, name)
```

返回 `getattr(self.xp, name)`，从而可直接调用如 `ba.sum`、`ba.zeros` 等后端函数。

---

## 使用示例

### 获取后端模块并创建数组

```python
from stm_data_processing.config import get_xp, get_backend

xp = get_xp()
arr = xp.array([1, 2, 3])   # CPU: numpy, GPU: cupy
print(get_backend())        # 'cpu' 或 'gpu'
```

### 在导入计算模块前强制后端

```python
from stm_data_processing.config import set_backend

set_backend("cpu")  # 必须先于计算模块导入
from stm_data_processing.dft.wannier90.mlwf_hamiltonian import MLWFHamiltonian
# 后续 MLWFHamiltonian 将使用 CPU 后端
```

### 查询后端状态

```python
from stm_data_processing.config import get_backend_status, is_gpu_available

status = get_backend_status()
print(status["backend"])          # 'cpu' 或 'gpu'
print(status["cupy_importable"])  # bool
print(status["cuda_devices"])     # int
print(status["cupy_version"])     # str 或 None

print(is_gpu_available())         # bool
```

### 使用 BackendArray

```python
from stm_data_processing.config import BackendArray

ba = BackendArray()                # 跟随全局 BACKEND
arr = ba.array([1, 2, 3])
result = ba.sum(arr)               # 通过 __getattr__ 委托到 xp.sum
cpu_arr = ba.to_cpu(result)        # 转回 NumPy

ba_gpu = BackendArray(backend="gpu")  # 显式指定后端
```

### 非法后端值处理

```python
from stm_data_processing.config import set_backend

try:
    set_backend("tpu")  # 非法值
except ValueError as e:
    print(e)  # backend must be 'cpu', 'gpu', or 'auto', got 'tpu'
```

---

## 依赖项

| 依赖 | 必需 | 说明 |
|------|------|------|
| `numpy` | 是 | CPU 后端（始终导入作为回退） |
| `cupy` | 否 | GPU 后端（可选，导入失败时自动回退 CPU） |
| `logging` | 是 | 状态日志输出 |
| `typing` | 是 | 类型注解（`Literal`, `Optional`） |

---

## 错误处理

| 异常 | 触发条件 |
|------|----------|
| `ValueError` | `set_backend()` 传入非 `'cpu'`/`'gpu'`/`'auto'` 的值 |
| `RuntimeError` | `BackendArray.to_gpu()` 在 CuPy 不可用时被调用 |

**非异常回退**:
- `set_backend("gpu")` 在 CuPy 不可用时**不抛异常**，而是记录 WARNING 并将 `BACKEND` 回退为 `"cpu"`。
- `get_xp()` 在 `BACKEND == "gpu"` 但 CuPy 不可用时回退返回 `numpy`。
- `get_cupy()` 在 GPU 不可用时返回 `None`（不抛异常）。

---

## 接口对齐检查清单

生成调用此模块的代码时，请确保：

- [ ] **⚠️ 在导入任何计算模块之前调用 `set_backend()`**
- [ ] **通过 `get_xp()` 获取后端数组模块（推荐），而非直接 import cupy/numpy**
- [ ] **`set_backend()` 接受 `'cpu'`、`'gpu'`、`'auto'` 三值，其他值抛 `ValueError`**
- [ ] **`BACKEND` 为全局可变状态，`set_backend()` 会改变其值；计算模块导入后后端即固化**
- [ ] **GPU 请求在不可用时会回退 CPU 并记录 WARNING，而非抛异常**
- [ ] **`get_backend_status()` 返回固定键：`backend`、`cupy_importable`、`cuda_devices`、`cupy_version`**
- [ ] **`get_cupy()` 已弃用，新代码使用 `get_xp()`**
- [ ] **`BackendArray(backend="auto")` 跟随全局 `BACKEND`；显式传值则覆盖**
- [ ] **`BackendArray.to_gpu()` 在 CuPy 缺失时抛 `RuntimeError`**

---

## 版本信息

- 模块路径：`src/stm_data_processing/config.py`
- 后端检测：模块导入时自动完成（`_detect_backend()`）
- 全局状态：`BACKEND`（`Literal["cpu", "gpu"]`）
- 日志级别：`INFO`（状态变更）、`WARNING`（GPU 回退）
- 推荐取后端方式：`get_xp()`
- 辅助类：`BackendArray`（后端无关数组操作）
