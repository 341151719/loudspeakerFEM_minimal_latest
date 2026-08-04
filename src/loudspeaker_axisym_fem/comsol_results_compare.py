from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from .json_utils import write_json

REFERENCE_FIGURES = [
    {"figure": 3, "name": "Static magnetic field", "target": "H field concentrated in magnetic gap"},
    {"figure": 4, "name": "Effective relative permeability", "target": "soft iron close to saturation in pole center"},
    {"figure": 5, "name": "Induced current density", "target": "skin effect stronger at 900 Hz than 50 Hz"},
    {"figure": 6, "name": "Blocked coil inductance", "target": "inductance decreases with frequency"},
    {"figure": 7, "name": "8 kHz SPL and displacement", "target": "cone breakup visible at high frequency"},
    {"figure": 8, "name": "Sensitivity and phase", "target": "flat roughly 100-1500 Hz; NRA damps 600 Hz cavity mode"},
    {"figure": 9, "name": "Back-cavity pressure mode", "target": "pressure phase changes around 600/630 Hz without NRA"},
    {"figure": 10, "name": "Total electric impedance", "target": "peak about 50 Hz, DC 5.6 ohm, nominal 6.3 ohm"},
    {"figure": 11, "name": "Structural eigenmodes", "target": "first mode just above 50 Hz, first breakup around 2350 Hz"},
    {"figure": 12, "name": "Directivity plot", "target": "-90 to 90 deg, normalized to 0 deg"},
]


def write_figure_checklist(path: str | Path) -> None:
    lines = ["# COMSOL PDF 图示逐图复现检查表", "", "| Figure | 名称 | COMSOL 目标 | Python 输出 | 差异 | 修正状态 |", "|---:|---|---|---|---|---|"]
    for f in REFERENCE_FIGURES:
        lines.append(f"| {f['figure']} | {f['name']} | {f['target']} | pending | pending | not started |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
