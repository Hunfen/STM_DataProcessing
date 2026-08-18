import csv
import re

pdf_text = ""


# ---------- 解析核心 ----------
def parse_thermocouple_table(text):
    """
    从PDF文本解析K型热电偶分度表，返回 {温度(°C): 电压(mV)} 字典。
    策略：每一行末尾都重复了该行的基准温度（整数），
    提取它之后，行内剩余的数字均为电压值（取前10个）。
    """
    data = {}
    for line in text.splitlines():
        line = line.strip()
        # 跳过标题、表格标记等无关行
        if not line or line.startswith(("=", "<", "#", "10. Appendix", "MULTIPROBE")):
            continue

        # 1. 提取行尾的基准温度（允许前后空格）
        tail_match = re.search(r"(-?\d+)\s*$", line)
        if not tail_match:
            continue
        base_temp = int(tail_match.group(1))

        # 2. 去掉行尾温度，得到电压数据部分
        body = line[: tail_match.start()]

        # 3. 提取所有带小数点的数字（允许缺少前导零，如 ".039"）
        voltage_strs = re.findall(r"-?\d*\.\d+", body)
        if len(voltage_strs) < 10:
            continue  # 数据不完整，跳过（例如OCR噪声行）

        # 4. 只取头10个电压，依次对应 base_temp, base_temp+1, ..., base_temp+9
        for i, v_str in enumerate(voltage_strs[:10]):
            data[base_temp + i] = float(v_str)

    return data


def fix_first_row(data):
    # -270°C 对应 -6.458 mV
    if -270 not in data:
        data[-270] = -6.458
    return data


def save_csv(data, filename="thermocouple.csv"):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Temperature_C", "EMF_mV"])
        # 按温度升序
        for temp in sorted(data.keys()):
            writer.writerow([temp, data[temp]])


if __name__ == "__main__":
    # 解析
    data = parse_thermocouple_table(pdf_text)
    data = fix_first_row(data)

    save_csv(data)
    print("CSV已生成: thermocouple.csv")
    print(f"共包含 {len(data)} 个温度点。")

    # 显示前几行示例
    print("\n示例数据（前10行）:")
    with open("thermocouple.csv") as f:
        for i, line in enumerate(f):
            if i < 11:
                print(line.strip())
