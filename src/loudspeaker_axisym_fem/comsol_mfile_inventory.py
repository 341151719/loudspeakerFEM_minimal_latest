from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List, Tuple

from .json_utils import write_json


def _parse_int_list(s: str) -> List[int]:
    return [int(x) for x in re.findall(r"-?\d+", s)]


def parse_mfile_inventory(path: str | Path) -> dict:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    inv: dict = {"source": str(path)}

    inv["geometry"] = {
        "axisymmetric": "axisymmetric(true)" in text,
        "insert_file": re.findall(r"geom\('geom1'\)\.insertFile\('([^']+)'", text),
        "run_feature": re.findall(r"geom\('geom1'\)\.run\('([^']+)'\)", text),
    }
    inv["parameters"] = dict(re.findall(r"model\.param\.set\('([^']+)',\s*'([^']+)'\);", text))
    inv["physics"] = [
        {"tag": m.group(1), "type": m.group(2), "geom": m.group(3)}
        for m in re.finditer(r"physics\.create\('([^']+)',\s*'([^']+)',\s*'([^']+)'\);", text)
    ]
    inv["multiphysics"] = [
        {"tag": m.group(1), "type": m.group(2), "dim": int(m.group(3))}
        for m in re.finditer(r"multiphysics\.create\('([^']+)',\s*'([^']+)',\s*(\d+)\);", text)
    ]

    selections = {}
    for m in re.finditer(r"selection\('([^']+)'\)\.label\((.*?)\);\s*model\.component\('comp1'\)\.selection\('\1'\)\.set\(\[([^\]]*)\]\);", text, re.S):
        selections[m.group(1)] = {"label_raw": m.group(2).strip(), "entities": _parse_int_list(m.group(3))}
    # More robust: include all .selection(...).set([...]) even without label nearby.
    for m in re.finditer(r"selection\('([^']+)'\)\.set\(\[([^\]]*)\]\);", text):
        selections.setdefault(m.group(1), {"label_raw": None, "entities": _parse_int_list(m.group(2))})
    inv["selections"] = selections

    materials = {}
    for m in re.finditer(r"material\.create\('([^']+)',\s*'([^']+)'\);", text):
        tag, typ = m.group(1), m.group(2)
        label_match = re.search(rf"material\('{tag}'\)\.label\('([^']+)'\);", text)
        sel_match = re.search(rf"material\('{tag}'\)\.selection\.named\('([^']+)'\);", text)
        materials[tag] = {"type": typ, "label": label_match.group(1) if label_match else None, "selection": sel_match.group(1) if sel_match else None, "properties": {}}
    for m in re.finditer(r"material\('([^']+)'\)\.propertyGroup\('def'\)\.set\('([^']+)',\s*'([^']+)'\);", text):
        tag, key, val = m.group(1), m.group(2), m.group(3)
        materials.setdefault(tag, {"type": None, "label": None, "selection": None, "properties": {}})["properties"][key] = val
    inv["materials"] = materials

    inv["physics_features"] = []
    for m in re.finditer(r"physics\('([^']+)'\)\.create\('([^']+)',\s*'([^']+)',\s*(\d+)\);", text):
        physics, tag, typ, dim = m.group(1), m.group(2), m.group(3), int(m.group(4))
        block = _nearby_block(text, m.start(), 1200)
        sel_named = re.findall(rf"physics\('{physics}'\)\.feature\('{tag}'\)\.selection\.named\('([^']+)'\)", block)
        sel_set = re.findall(rf"physics\('{physics}'\)\.feature\('{tag}'\)\.selection\.set\(\[([^\]]*)\]\)", block)
        sets = re.findall(rf"physics\('{physics}'\)\.feature\('{tag}'\)\.set\('([^']+)',\s*([^;]+)\);", block)
        inv["physics_features"].append({
            "physics": physics,
            "tag": tag,
            "type": typ,
            "dim": dim,
            "selection_named": sel_named,
            "selection_set": [_parse_int_list(x) for x in sel_set],
            "settings": {k: v.strip() for k, v in sets},
        })

    inv["mesh"] = {
        "global_size": dict(re.findall(r"mesh\('mesh1'\)\.feature\('size'\)\.set\('([^']+)',\s*'?([^'\);]+)'?\);", text)),
        "mapped_domains": _first_int_list(text, r"feature\('map1'\)\.selection\.set\(\[([^\]]+)\]\)"),
        "size_features": [],
        "distribution_features": [],
        "boundary_layer_domains": _first_int_list(text, r"feature\('bl1'\)\.selection\.set\(\[([^\]]+)\]\)"),
    }
    for m in re.finditer(r"feature\('map1'\)\.feature\('(size\d+)'\)\.selection\.set\(\[([^\]]*)\]\);.*?feature\('map1'\)\.feature\('\1'\)\.set\('hmax',\s*'([^']+)'\);", text, re.S):
        inv["mesh"]["size_features"].append({"tag": m.group(1), "domains": _parse_int_list(m.group(2)), "hmax": m.group(3)})
    for m in re.finditer(r"feature\('map1'\)\.feature\('(dis\d+)'\)\.selection\.set\(\[([^\]]*)\]\);.*?feature\('map1'\)\.feature\('\1'\)\.set\('numelem',\s*(\d+)\);", text, re.S):
        inv["mesh"]["distribution_features"].append({"tag": m.group(1), "boundaries": _parse_int_list(m.group(2)), "numelem": int(m.group(3))})

    inv["studies"] = []
    for m in re.finditer(r"study(?:\.create)?\('([^']+)'\)", text):
        tag = m.group(1)
        if tag not in [s.get("tag") for s in inv["studies"]]:
            inv["studies"].append({"tag": tag})
    inv["frequency_lists"] = re.findall(r"feature\('frlin'\)\.set\('plist',\s*'([^']+)'\);", text)
    inv["result_expressions"] = re.findall(r"setIndex\('expr',\s*'([^']+)'", text) + re.findall(r"set\('expr',\s*'([^']+)'\)", text)
    inv["key_boundaries"] = {
        "exterior_field_boundary": _first_int_list(text, r"feature\('efc1'\)\.selection\.set\(\[([^\]]+)\]\)"),
        "narrow_region_1_domains": _first_int_list(text, r"feature\('nra1'\)\.selection\.set\(\[([^\]]+)\]\)"),
        "narrow_region_2_domains": _first_int_list(text, r"feature\('nra2'\)\.selection\.set\(\[([^\]]+)\]\)"),
        "fixed_boundaries": _first_int_list(text, r"feature\('fix1'\)\.selection\.set\(\[([^\]]+)\]\)"),
    }
    return inv


def _nearby_block(text: str, start: int, n: int) -> str:
    return text[start:start+n]


def _first_int_list(text: str, pattern: str) -> List[int]:
    m = re.search(pattern, text, re.S)
    return _parse_int_list(m.group(1)) if m else []


def write_mfile_inventory(mfile: str | Path, out_json: str | Path) -> dict:
    inv = parse_mfile_inventory(mfile)
    write_json(out_json, inv, indent=2)
    return inv
