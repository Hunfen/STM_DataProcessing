# 涡旋数计算模块接口文档

## 模块概述

`vortex_num.py` 提供涡旋数（vortex number）的计算功能，根据外加磁场的磁通量与磁通量子之比计算涡旋数量。

**功能**: 输入磁场强度 `field`（单位 T）与面积 `area`（单位 m²），输出涡旋数（无量纲）。

**公式**: `Φ = B · A / Φ₀`，其中 `Φ₀ = 2.067833848e-15 Wb` 为磁通量子。

---

## 核心函数：`vortex_num`

### 函数定义

```python
def vortex_num(field: float = 0, area: float = 0)
```

### 参数表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `field` | `float` | `0` | 磁场强度（单位：特斯拉 T）。源码 docstring 中拼写为 `filed`，实际变量名为 `field` |
| `area` | `float` | `0` | 面积（单位：平方米 m²） |

### 返回值

| 类型 | 说明 |
|------|------|
| `float` | 计算得到的涡旋数，即 `field * area / Φ₀` |

---

## 数学公式

涡旋数计算公式：

```
N = B · A / Φ₀
```

其中：
- `B` : 磁场强度（`field`，单位 T）
- `A` : 面积（`area`，单位 m²）
- `Φ₀` : 磁通量子，`2.067833848e-15 Wb`（源码中硬编码，注释标注 `1 Wb <=> 1 T*m²`）

**注意**: 函数为纯标量运算，不依赖任何后端（NumPy/CuPy），返回值为 Python `float`。

---

## 使用示例

### 基础用法

```python
from stm_data_processing.stm.vortex_num import vortex_num

# 计算 1 T 磁场穿过 1 m² 面积时的涡旋数
n = vortex_num(field=1.0, area=1.0)
print(n)  # 约 4.836e14

# 默认参数（field=0, area=0）
n_zero = vortex_num()
print(n_zero)  # 0.0
```

### 典型物理场景

```python
from stm_data_processing.stm.vortex_num import vortex_num

# 10 T 磁场下，1 µm × 1 µm 区域内的涡旋数
B = 10.0              # T
area = (1e-6) ** 2    # m² (1 µm²)
n_vortices = vortex_num(B, area)
print(f"涡旋数 = {n_vortices:.4f}")
```

---

## 依赖项

| 依赖 | 必需 | 说明 |
|------|------|------|
| 无 | - | 该模块无任何外部依赖，仅使用 Python 内置运算 |

---

## 错误处理

| 异常 | 触发条件 |
|------|----------|
| 无 | 函数为纯标量除法，不抛出异常；传入非数值类型时可能由 Python 解释器抛出 `TypeError`（如字符串） |

**注意**: 函数不对输入做数值校验，负值或零值均可正常计算（`area=0` 时返回 `0`，不会触发除零错误，因为 `Φ₀` 为常量非零）。

---

## 接口对齐检查清单

生成调用此模块的代码时，请确保：

- [ ] **`field` 单位为特斯拉 (T)，`area` 单位为平方米 (m²)**
- [ ] **两个参数均为 `float`，默认值均为 `0`**
- [ ] **返回值为无量纲涡旋数 `float`**
- [ ] **无需导入后端配置，本函数不涉及 CPU/GPU 后端**
- [ ] **`Φ₀` 为硬编码常量 `2.067833848e-15`，不可通过参数覆盖**

---

## 版本信息

- 模块路径：`src/stm_data_processing/stm/vortex_num.py`
- 函数数量：1（`vortex_num`）
- 磁通量子：`Φ₀ = 2.067833848e-15 Wb`（源码硬编码）
- 日志级别：无（静默计算）
- 后端依赖：无
