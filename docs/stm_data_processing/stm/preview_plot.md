# 预览绘图 colormap 模块接口文档

## 模块概述

`preview_plot.py` 提供 Gwyddion 风格的 colormap（颜色映射）对象，用于 STM 数据预览图的可视化渲染。

**功能**: 定义并导出一个 `matplotlib.colors.LinearSegmentedColormap` 实例 `gwyddion`，其颜色分段数据（红/绿/蓝通道）模拟 Gwyddion 软件的配色风格。

---

## 核心对象：`gwyddion`

### 对象定义

```python
gwyddion = LinearSegmentedColormap("gwyddion", segmentdata=cdict_gwyddion, N=4096)
```

| 属性 | 值 | 说明 |
|------|-----|------|
| 名称 | `"gwyddion"` | colormap 的注册名 |
| 分段数据 | `cdict_gwyddion` | 红/绿/蓝三通道分段表（见下） |
| `N` | `4096` | 颜色查表（LUT）的采样点数 |

### 颜色分段数据：`cdict_gwyddion`

```python
cdict_gwyddion: dict = {
    "red":   [...],  # 4 个锚点
    "green": [...],  # 4 个锚点
    "blue":  [...],  # 4 个锚点
}
```

每个通道使用 **4 个锚点**（`(位置, 左值, 右值)`），位置分别为 `0.0`、`0.344671`、`0.687075`、`1.0`：

| 通道 | 位置 0.0 | 位置 0.344671 | 位置 0.687075 | 位置 1.0 |
|------|----------|----------------|----------------|-----------|
| `red` | 0.0 | 0.658824 | 0.953506 | 1.0 |
| `green` | 0.0 | 0.156863 | 0.759686 | 1.0 |
| `blue` | 0.0 | 0.0588235 | 0.363821 | 1.0 |

**视觉特征**: 低端为黑色（三通道均为 0），经过暗红/暗绿色调过渡到高端白色（三通道均为 1），整体呈暖色系渐变。

---

## 使用示例

### 在 matplotlib 中使用

```python
import matplotlib.pyplot as plt
import numpy as np
from stm_data_processing.stm.preview_plot import gwyddion

# 生成示例 STM 数据
data = np.random.rand(256, 256)

# 使用 Gwyddion 风格 colormap 渲染
plt.imshow(data, cmap=gwyddion)
plt.colorbar()
plt.show()
```

### 直接访问 colormap 对象

```python
from stm_data_processing.stm.preview_plot import gwyddion

print(gwyddion.name)   # 'gwyddion'
print(gwyddion.N)      # 4096

# 将归一化值映射为 RGBA 颜色
rgba = gwyddion(0.5)   # 返回 (r, g, b, a) 元组
print(rgba)
```

### 注册为全局 colormap（可选）

```python
import matplotlib.pyplot as plt
from stm_data_processing.stm.preview_plot import gwyddion

# 注册后可通过字符串名引用
plt.colormaps.register(gwyddion, name="gwyddion")
plt.imshow(data, cmap="gwyddion")
```

---

## 依赖项

| 依赖 | 必需 | 说明 |
|------|------|------|
| `matplotlib.colors.LinearSegmentedColormap` | 是 | colormap 构造 |

---

## 错误处理

| 异常 | 触发条件 |
|------|----------|
| 无 | 模块仅在导入时构造 colormap；传入非法分段数据可能导致 `LinearSegmentedColormap` 在构造时抛错，但当前数据为合法内置值 |

---

## 接口对齐检查清单

生成调用此模块的代码时，请确保：

- [ ] **通过 `from stm_data_processing.stm.preview_plot import gwyddion` 导入 colormap**
- [ ] **`gwyddion` 是 `LinearSegmentedColormap` 实例，直接用于 `cmap=` 参数**
- [ ] **colormap 名称为 `"gwyddion"`，查表点数 `N=4096`**
- [ ] **颜色分段锚点位置为 `0.0`、`0.344671`、`0.687075`、`1.0`（勿假设等间距）**
- [ ] **本模块仅提供 colormap，不提供任何绘图函数**

---

## 版本信息

- 模块路径：`src/stm_data_processing/stm/preview_plot.py`
- 导出对象：1（`gwyddion`）以及颜色分段数据字典 `cdict_gwyddion`
- colormap 名称：`gwyddion`
- 查表点数：`N=4096`
- 日志级别：无（静默）
- 后端依赖：无（纯 matplotlib colormap 定义）
