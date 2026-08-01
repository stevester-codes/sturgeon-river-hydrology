#!/usr/bin/env python3
from pathlib import Path

path = Path('.github/workflows/sturgeon-operational.yml')
text = path.read_text()

if "python qpf_forecast_rollover.py" not in text:
    if "python qpf_forecast_v2.py" not in text:
        raise RuntimeError("Short-range QPF command was not found")
    text = text.replace("python qpf_forecast_v2.py", "python qpf_forecast_rollover.py", 1)

if "python medium_range_qpf_rollover.py" not in text:
    if "python medium_range_qpf.py" not in text:
        raise RuntimeError("Medium-range QPF command was not found")
    text = text.replace("python medium_range_qpf.py", "python medium_range_qpf_rollover.py", 1)

trigger_anchor = "      - 'qpf_forecast_v2.py'\n"
if "      - 'qpf_forecast_rollover.py'\n" not in text:
    if trigger_anchor not in text:
        raise RuntimeError("QPF trigger anchor was not found")
    text = text.replace(
        trigger_anchor,
        trigger_anchor + "      - 'qpf_forecast_rollover.py'\n",
        1,
    )

medium_trigger = "      - 'medium_range_qpf.py'\n"
if "      - 'medium_range_qpf_rollover.py'\n" not in text:
    if medium_trigger not in text:
        raise RuntimeError("Medium-range trigger anchor was not found")
    text = text.replace(
        medium_trigger,
        medium_trigger + "      - 'medium_range_qpf_rollover.py'\n",
        1,
    )

compile_anchor = "            qpf_forecast_v2.py \\\n"
if "            qpf_forecast_rollover.py \\\n" not in text:
    if compile_anchor not in text:
        raise RuntimeError("QPF compile anchor was not found")
    text = text.replace(
        compile_anchor,
        compile_anchor + "            qpf_forecast_rollover.py \\\n",
        1,
    )

medium_compile = "            medium_range_qpf.py \\\n"
if "            medium_range_qpf_rollover.py \\\n" not in text:
    if medium_compile not in text:
        raise RuntimeError("Medium-range compile anchor was not found")
    text = text.replace(
        medium_compile,
        medium_compile + "            medium_range_qpf_rollover.py \\\n",
        1,
    )

path.write_text(text)
print('Operational workflow now uses rollover-aware weather wrappers.')
