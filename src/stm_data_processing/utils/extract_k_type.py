import csv
import re
from pathlib import Path

pdf_text = ""


# ---------- Core parsing ----------
def parse_thermocouple_table(text):
    """
    Parse a K-type thermocouple calibration table from PDF text.

    Returns a {temperature (deg C): voltage (mV)} dict. Each data row ends
    with its base temperature as an integer; the decimal numbers before it
    are the voltages for base_temp, base_temp+1, ..., up to 10 columns.
    Rows with fewer voltage columns are written with the columns present
    instead of being dropped, and trailing integers outside the K-type
    temperature range are rejected so the tail regex never mistakes the
    fractional part of a decimal voltage for a temperature (e.g. 41.276
    must not yield 276).
    """
    data = {}
    for line in text.splitlines():
        line = line.strip()
        # Skip headers, table markers and other unrelated lines.
        if not line or line.startswith(("=", "<", "#", "10. Appendix", "MULTIPROBE")):
            continue

        # 1. Extract the trailing base temperature (allowing surrounding spaces).
        #    The negative lookbehind rejects integers that are the fractional
        #    part of a trailing decimal voltage (e.g. the "276" in "41.276").
        tail_match = re.search(r"(?<![.\d])(-?\d+)\s*$", line)
        if not tail_match:
            continue
        base_temp = int(tail_match.group(1))
        # Plausibility check: the base temperature must lie in the K-type
        # thermocouple range (deg C). Out-of-range values mean the tail regex
        # caught some other trailing integer (e.g. a page number or year).
        if not (-270 <= base_temp <= 1372):
            continue

        # 2. Strip the trailing temperature to get the voltage data part.
        body = line[: tail_match.start()]

        # 3. Extract all decimal numbers (allowing a missing leading zero,
        #    e.g. ".039").
        voltage_strs = re.findall(r"-?\d*\.\d+", body)

        # 4. Use all available voltage columns (at most 10), mapped to
        #    base_temp, base_temp+1, ... Rows with fewer than 10 columns are
        #    written with the columns present instead of being dropped.
        for i, v_str in enumerate(voltage_strs[:10]):
            data[base_temp + i] = float(v_str)

    return data


def fix_first_row(data):
    # -270 deg C corresponds to -6.458 mV.
    if -270 not in data:
        data[-270] = -6.458
    return data


def save_csv(data, filename="thermocouple.csv"):
    with Path(filename).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Temperature_C", "EMF_mV"])
        # Sort by ascending temperature.
        for temp in sorted(data.keys()):
            writer.writerow([temp, data[temp]])


if __name__ == "__main__":
    # Parse the table.
    data = parse_thermocouple_table(pdf_text)
    data = fix_first_row(data)

    save_csv(data)
    print("CSV已生成: thermocouple.csv")
    print(f"共包含 {len(data)} 个温度点。")

    # Show the first few sample rows.
    print("\n示例数据 (前10行):")
    with Path("thermocouple.csv").open() as f:
        for i, line in enumerate(f):
            if i < 11:
                print(line.strip())
