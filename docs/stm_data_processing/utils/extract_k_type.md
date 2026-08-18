# K 型热电偶分度表解析模块接口文档

## 模块概述

- **模块路径**：`src/stm_data_processing/utils/extract_k_type.py`
- **职责**：从 PDF 抽取的文本中解析 K 型热电偶（NiCr-NiAl）分度表，输出 `{温度(°C): 电压(mV)}` 字典，并支持写入 CSV。
- **性质**：`parse_thermocouple_table` / `fix_first_row` / `save_csv` 均为纯函数/普通函数；所有副作用（写 CSV、打印示例）均已移入 `if __name__ == "__main__"` 块，因此模块可安全 `import` 而不会触发文件写入或打印。
- **模块级变量**：`pdf_text = ""`，为占位文本，`__main__` 块从它读取待解析内容（需使用者自行填入真实 PDF 文本）。

---

## 核心函数

### `parse_thermocouple_table(text)`

从 PDF 文本解析 K 型热电偶分度表，返回温度 → 电压字典。

```python
def parse_thermocouple_table(text):
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `text` | `str` | PDF 抽取的原始文本（按行分割处理） |

**返回**：`dict`，键为温度（`int`，单位 °C），值为对应热电动势（`float`，单位 mV）。

**解析策略**（按源码逻辑）：

1. 逐行 `strip()`，跳过空行及以 `=`, `<`, `#`, `"10. Appendix"`, `"MULTIPROBE"` 开头的标题/标记行。
2. 用正则 `(-?\d+)\s*$` 提取行尾基准温度（整数）。
3. 取行尾温度之前的部分作为数据主体。
4. 用正则 `-?\d*\.\d+` 提取所有带小数点的数字（兼容缺失前导零的 `.039` 形式）。
5. 若数字个数少于 10，视为 OCR 噪声行跳过。
6. 取前 10 个电压，依次对应 `base_temp, base_temp+1, ..., base_temp+9`。

---

### `fix_first_row(data)`

补齐/修正第一行：确保 `-270 °C` 数据点存在（对应 `-6.458 mV`）。

```python
def fix_first_row(data):
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `data` | `dict` | `parse_thermocouple_table` 返回的温度 → 电压字典 |

**返回**：`dict`，在原字典上修改后返回（`-270` 缺失时写入 `-6.458`）。该函数就地修改传入字典并返回同一对象。

---

### `save_csv(data, filename="thermocouple.csv")`

将分度表字典写入 CSV。

```python
def save_csv(data, filename="thermocouple.csv"):
```

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `data` | `dict` | - | 温度 → 电压字典 |
| `filename` | `str` | `"thermocouple.csv"` | 输出 CSV 文件名 |

**返回**：`None`。写出的 CSV 表头为 `Temperature_C, EMF_mV`，按温度升序逐行输出。

---

## 使用示例

```python
from stm_data_processing.utils.extract_k_type import (
    parse_thermocouple_table,
    fix_first_row,
    save_csv,
)

pdf_text = """
MULTIPROBE 10. Appendix: K-type thermocouple table
... 0.000 0.039 0.079 0.119 0.158 0.198 0.238 0.277 0.317 0.357 0
... 0.397 0.437 0.477 0.517 0.557 0.597 0.637 0.677 0.718 0.758 10
"""

data = parse_thermocouple_table(pdf_text)  # {0: 0.000, 1: 0.039, ..., 10: 0.397, ...}
data = fix_first_row(data)                 # 确保 -270 -> -6.458 存在
save_csv(data, "thermocouple.csv")         # 写入 CSV（副作用显式由调用方触发）
```

> 说明：模块导入本身不会执行 `__main__` 块，因此不会写文件或打印；`save_csv` 需显式调用才会产生副作用。

---

## 依赖项

| 依赖 | 必需 | 说明 |
|------|------|------|
| `csv` | 是 | 写 CSV 文件 |
| `re` | 是 | 正则解析 |

---

## 错误处理

| 异常 | 触发条件 |
|------|----------|
| `ValueError` | `parse_thermocouple_table` 中 `float(v_str)` 转换失败（理论上正则 `-?\d*\.\d+` 匹配项均可转 float，正常不会触发） |
| `OSError` | `save_csv` 无法创建/写入目标文件（如路径无权限） |

> 解析函数对不完整/噪声行采取「跳过」而非抛错策略，返回的字典可能缺失部分温度点。

---

## 接口对齐检查清单

生成调用此模块的代码时，请确保：

- [ ] **模块可安全 `import`：写 CSV、打印示例均在 `if __name__ == "__main__"` 块内**
- [ ] **`parse_thermocouple_table` 返回 `{int 温度: float 电压}` 字典**
- [ ] **每行需包含至少 10 个带小数的电压值，否则该行被跳过**
- [ ] **`fix_first_row` 就地修改并返回同一字典，`-270` 缺省补 `-6.458`**
- [ ] **`save_csv` 按温度升序输出，表头为 `Temperature_C, EMF_mV`**
- [ ] **`save_csv` 是唯一产生文件副作用的函数，需显式调用**

---

## 版本信息

- 模块路径：`src/stm_data_processing/utils/extract_k_type.py`
- 解析目标：K 型热电偶分度表 PDF 文本
- 日志级别：无（仅在 `__main__` 块 `print`）
- **纯函数 + 显式 `save_csv`，可安全 import**
