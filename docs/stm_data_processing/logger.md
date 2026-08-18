# 日志模块接口文档

## 模块概述

`logger.py` 是 `stm_data_processing` 包的日志封装，围绕标准库 `logging` 提供零样板（`basicConfig` / handler / formatter）的简洁接口。

**核心设计**:
- 所有 handler（控制台 + 文件）都挂在 **root logger** 上，因此包内任意位置（以及用户脚本中）通过 `logging.getLogger(__name__)` 创建的 logger 都能自动继承相同的格式与文件输出。
- 提供一行式工厂 `get_logger()`、级别控制 `set_level()`、文件输出 `enable_file()`、格式定制 `set_format()` 与一次性配置 `setup()`。
- `get_logger()` 默认使用调用方模块的 `__name__`，使每个模块自带 `%(name)s` 标签。

---

## 模块常量

| 常量 | 值 | 说明 |
|------|-----|------|
| `DEFAULT_FORMAT` | `"%(asctime)s \| %(levelname)-8s \| %(name)s \| %(message)s"` | 默认日志格式 |
| `DEFAULT_DATE_FORMAT` | `"%Y-%m-%d %H:%M:%S"` | 默认时间格式 |
| `DEFAULT_LEVEL` | `"INFO"` | 默认日志级别 |
| `_PACKAGE_LOGGER` | `"stm_data_processing"` | 包级 logger 名称 |

**级别映射 `_LEVELS`**（支持名称 → 数值）:

| 名称 | 数值 |
|------|------|
| `"critical"` | `logging.CRITICAL` |
| `"error"` | `logging.ERROR` |
| `"warning"` | `logging.WARNING` |
| `"info"` | `logging.INFO` |
| `"debug"` | `logging.DEBUG` |
| `"notset"` | `logging.NOTSET` |

---

## 内部状态

| 状态 | 类型 | 说明 |
|------|------|------|
| `_state` | `dict` | 全局配置：`level`、`format`、`datefmt` |
| `_stream_handler` | `StreamHandler \| None` | 控制台 handler（`stderr`） |
| `_file_handler` | `FileHandler \| None` | 文件 handler |
| `_managed` | `dict[str, Logger]` | 经 `get_logger()` 创建的 logger，供 `set_level()` 实时更新 |
| `_fixed` | `set[str]` | 显式指定过 level 的 logger 名，`set_level()` 对其免疫 |

---

## 内部函数

| 函数 | 说明 |
|------|------|
| `_resolve_level(level)` | 将级别名称（`"info"`）或数值（`logging.INFO`）转换为 `int`；未知名称抛 `ValueError` |
| `_make_formatter()` | 依据 `_state` 构造 `logging.Formatter` |
| `_ensure_configured()` | 在 root logger 无 handler 时挂载默认控制台 handler，并设置包级 logger 级别 |

---

## 核心函数

### `get_logger(name=None, level=None)`

返回一个完整配置的 logger。

```python
def get_logger(name: str | None = None, level: str | int | None = None) -> logging.Logger
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | `str \| None` | `None` | logger 名称；默认取调用方模块的 `__name__`（无法解析时回退为 `"stm_data_processing"`） |
| `level` | `str \| int \| None` | `None` | 该 logger 的级别；默认跟随全局配置级别 |

**行为**:
- `level` 为 `None` 时：logger 使用全局级别 `_state["level"]`，并**从 `_fixed` 中移除**该名称（最新调用生效）。
- `level` 非 `None` 时：logger 使用显式级别，并**加入 `_fixed`**（此后免疫 `set_level()`）。
- 无论哪种情况，logger 都会被登记到 `_managed`。

```python
from stm_data_processing.logger import get_logger
logger = get_logger()       # name = 当前模块 __name__
logger.info("hello")
```

### `set_level(level)`

设置 `get_logger()` 与包级 logger 的默认级别。

```python
def set_level(level: str | int) -> None
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `level` | `str \| int` | 新默认级别（名称或数值） |

**行为**:
- 更新 `_state["level"]`。
- 设置包级 logger `stm_data_processing` 的级别。
- 遍历 `_managed` 中**不在 `_fixed`** 内的 logger，实时更新其级别。

```python
from stm_data_processing.logger import set_level
set_level("debug")
```

### `set_format(fmt=None, datefmt=None)`

自定义 handler 的日志格式（及可选时间格式）。

```python
def set_format(fmt: str | None = None, datefmt: str | None = None) -> None
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `fmt` | `str \| None` | logging 格式字符串，如 `"%(levelname)s %(message)s"` |
| `datefmt` | `str \| None` | 时间格式字符串，如 `"%H:%M:%S"` |

**行为**: 非 `None` 时更新 `_state`，并即时应用到已存在的控制台与文件 handler。

### `enable_file(path, level=None)`

将日志输出路由到文件（在控制台输出之外追加）。

```python
def enable_file(path: str | Path, level: str | int | None = None) -> logging.FileHandler
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `path` | `str \| Path` | 日志文件路径（父目录自动创建） |
| `level` | `str \| int \| None` | 文件 handler 的级别阈值（可选） |

**返回**: 新挂载的 `logging.FileHandler`。

**行为**:
- 再次调用并传入新路径会**替换**先前的日志文件（移除并关闭旧 handler）。
- 以 `encoding="utf-8"` 打开文件。

```python
from stm_data_processing.logger import enable_file
enable_file("logs/run.log")
```

### `disable_file()`

卸载并关闭 `enable_file()` 添加的文件 handler。

```python
def disable_file() -> None
```

### `setup(level=DEFAULT_LEVEL, file=None, fmt=None, datefmt=None)`

一次性配置，并返回调用方模块的 logger。

```python
def setup(
    level: str | int = DEFAULT_LEVEL,
    file: str | Path | None = None,
    fmt: str | None = None,
    datefmt: str | None = None,
) -> logging.Logger
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `level` | `str \| int` | `DEFAULT_LEVEL`（`"INFO"`） | 默认日志级别 |
| `file` | `str \| Path \| None` | `None` | 同时写入的日志文件（可选） |
| `fmt` | `str \| None` | `None` | 格式字符串 |
| `datefmt` | `str \| None` | `None` | 时间格式字符串 |

**返回**: 以调用方模块命名、已完成配置的 `logging.Logger`。

**执行顺序**: `set_format`（若提供了 fmt/datefmt）→ `set_level(level)` → `enable_file`（若提供了 file）→ `get_logger(调用方 __name__)`。

```python
from stm_data_processing.logger import setup
logger = setup(level="debug", file="run.log")
logger.info("ready")
```

---

## 与包内 `logging.getLogger(__name__)` 的兼容关系

本模块的设计保证与包内既有的 `logging.getLogger(__name__)` 调用完全兼容：

| 机制 | 说明 |
|------|------|
| **handler 挂在 root** | 控制台/文件 handler 都挂在 root logger，包内任意 logger（包括 `logging.getLogger(__name__)` 直接创建的）无需自行配置 handler，即可自动继承格式与文件输出 |
| **包级 logger 默认级别控制** | `_ensure_configured()` 与 `set_level()` 会设置 `stm_data_processing` 这一包级 logger 的级别，使得包内所有子 logger（`stm_data_processing.*`）默认可见，同时**不抬高**第三方库的日志噪声 |
| **managed logger 的 `set_level` 实时生效** | 经 `get_logger()`（无显式 level）创建的 logger 被登记到 `_managed`，后续调用 `set_level()` 会**实时**更新其级别 |
| **显式 level 的 logger 不受 `set_level` 影响** | 在 `get_logger(level=...)` 中显式指定过级别的 logger 被加入 `_fixed`，`set_level()` 不会覆盖它 |

**注意**: 如果用户先通过 `logging.basicConfig()` 配置了 handler，`_ensure_configured()` 会检测到 root 已有 handler，从而**不重复添加**自己的控制台 handler（尊重用户配置）。

---

## 使用示例

### 一行式 logger 工厂

```python
from stm_data_processing.logger import get_logger

logger = get_logger()   # name = 当前模块 __name__
logger.debug("verbose detail")
logger.info("loading data")
logger.warning("something odd")
logger.error("something failed")
```

### 全局级别切换（实时生效）

```python
from stm_data_processing.logger import get_logger, set_level

a = get_logger("module.a")
b = get_logger("module.b")

set_level("debug")   # a、b 级别实时降为 DEBUG

fixed = get_logger("module.c", level="error")  # 显式 level，免疫 set_level
set_level("info")     # a、b 升为 INFO，c 仍为 ERROR
```

### 一次性配置

```python
from stm_data_processing.logger import setup

logger = setup(level="debug", file="logs/run.log")
logger.info("ready")
```

### 文件输出开关

```python
from stm_data_processing.logger import enable_file, disable_file

handler = enable_file("logs/run.log", level="debug")
# ... 记录日志 ...
disable_file()   # 关闭并移除文件 handler
```

### 自定义格式

```python
from stm_data_processing.logger import set_format

set_format("%(levelname)s | %(message)s", "%H:%M:%S")
```

### 与原生 logging 混用

```python
import logging
from stm_data_processing.logger import setup

setup(level="info")  # 挂载 root handler，包级可见

# 包内既有风格仍然可用
native = logging.getLogger("stm_data_processing.stm.some_module")
native.info("works without extra config")
```

---

## 依赖项

| 依赖 | 必需 | 说明 |
|------|------|------|
| `logging` | 是 | 标准库日志核心 |
| `inspect` | 是 | 解析调用方模块 `__name__` |
| `sys` | 是 | 控制台 handler 输出到 `stderr` |
| `pathlib.Path` | 是 | 文件路径处理与父目录创建 |

---

## 错误处理

| 异常 | 触发条件 |
|------|----------|
| `ValueError` | `_resolve_level()` 收到未知级别名称（如 `"verbose"`），报错信息列出合法级别 |

**注意**: `get_logger()` / `set_level()` 传入的级别字符串会经 `.lower()` 归一化，大小写不敏感。

---

## 接口对齐检查清单

生成调用此模块的代码时，请确保：

- [ ] **通过 `get_logger()` 或 `setup()` 获取 logger，默认名称为调用方 `__name__`**
- [ ] **级别接受名称（`"debug"`/`"info"`/...）或数值（`logging.DEBUG` 等），大小写不敏感**
- [ ] **handler 挂在 root，包内 `logging.getLogger(__name__)` 无需额外配置即可工作**
- [ ] **`get_logger(level=...)` 显式级别会锁定该 logger，使其免疫后续 `set_level()`**
- [ ] **`get_logger()` 无显式级别时，logger 跟随全局级别并实时响应 `set_level()`**
- [ ] **`enable_file()` 重复调用会替换旧文件 handler；`disable_file()` 关闭并移除**
- [ ] **`set_format()` 会即时应用到现有控制台与文件 handler**
- [ ] **未知级别名称抛 `ValueError`**
- [ ] **若用户先调用 `logging.basicConfig()`，本模块不会重复添加控制台 handler**

---

## 版本信息

- 模块路径：`src/stm_data_processing/logger.py`
- 默认级别：`"INFO"`（`DEFAULT_LEVEL`）
- 默认格式：`"%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"`
- 默认时间格式：`"%Y-%m-%d %H:%M:%S"`
- 包级 logger：`"stm_data_processing"`
- 公开 API：`get_logger`、`set_level`、`set_format`、`enable_file`、`disable_file`、`setup`
