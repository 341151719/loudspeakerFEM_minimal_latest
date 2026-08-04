from __future__ import annotations

import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "comsol_req7_raw"
PKG = ROOT / "COMSOL_loudspeaker_req7_export_package"
ZIP = ROOT / "COMSOL_loudspeaker_req7_export_package.zip"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def summarize_csv(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    out = {"file": path.name, "bytes": path.stat().st_size, "rows": len(rows), "columns": reader.fieldnames or []}
    if rows and "solved_freq_Hz" in rows[0]:
        out["solved_freq_Hz_values"] = sorted({round(float(r["solved_freq_Hz"]), 9) for r in rows if r.get("solved_freq_Hz")})
    if rows and "domain_id" in rows[0]:
        out["domain_ids"] = sorted({int(float(r["domain_id"])) for r in rows if r.get("domain_id")})
    if rows and "depth_bin" in rows[0]:
        out["depth_bins"] = dict(Counter(r["depth_bin"] for r in rows))
    return out


def write_validation_summary(summaries: list[dict]) -> None:
    with (PKG / "validation_summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["file", "rows", "cols", "bytes", "freq_count", "domains", "depth_bins"])
        for s in summaries:
            w.writerow([
                s["file"], s["rows"], len(s["columns"]), s["bytes"],
                len(s.get("solved_freq_Hz_values", [])),
                json.dumps(s.get("domain_ids", [])),
                json.dumps(s.get("depth_bins", {}), ensure_ascii=False),
            ])


def write_numeric_summaries() -> None:
    jac = read_rows(RAW / "pml_coordinate_jacobian_points.csv")
    acc = defaultdict(lambda: {"n": 0, "det_abs_min": float("inf"), "det_abs_max": 0.0, "vol_abs_min": float("inf"), "vol_abs_max": 0.0})
    for r in jac:
        key = (r["solved_freq_Hz"], r["domain_id"])
        det_abs = math.hypot(float(r["detJ_real"]), float(r["detJ_imag"]))
        vol_abs = math.hypot(float(r["axisym_volume_factor_real"]), float(r["axisym_volume_factor_imag"]))
        d = acc[key]
        d["n"] += 1
        d["det_abs_min"] = min(d["det_abs_min"], det_abs)
        d["det_abs_max"] = max(d["det_abs_max"], det_abs)
        d["vol_abs_min"] = min(d["vol_abs_min"], vol_abs)
        d["vol_abs_max"] = max(d["vol_abs_max"], vol_abs)
    with (PKG / "pml_jacobian_summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["freq_Hz", "domain_id", "samples", "detJ_abs_min", "detJ_abs_max", "axisym_volume_factor_abs_min", "axisym_volume_factor_abs_max"])
        for (freq, dom), d in sorted(acc.items(), key=lambda x: (float(x[0][0]), int(x[0][1]))):
            w.writerow([freq, dom, d["n"], d["det_abs_min"], d["det_abs_max"], d["vol_abs_min"], d["vol_abs_max"]])

    pressure = read_rows(RAW / "pml_pressure_linecuts.csv")
    pacc = defaultdict(lambda: {"n": 0, "p_abs_sum": 0.0, "p_abs_max": 0.0})
    for r in pressure:
        key = (r["solved_freq_Hz"], r["domain_id"], r["depth_bin"])
        p = float(r["p_abs"])
        d = pacc[key]
        d["n"] += 1
        d["p_abs_sum"] += p
        d["p_abs_max"] = max(d["p_abs_max"], p)
    with (PKG / "pml_pressure_depth_summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["freq_Hz", "domain_id", "depth_bin", "samples", "mean_p_abs", "max_p_abs"])
        for (freq, dom, depth), d in sorted(pacc.items(), key=lambda x: (float(x[0][0]), int(x[0][1]), x[0][2])):
            w.writerow([freq, dom, depth, d["n"], d["p_abs_sum"] / d["n"], d["p_abs_max"]])


def write_manifests(summaries: list[dict]) -> None:
    props = read_rows(RAW / "pml_feature_properties.csv")
    prop_map = {(r["node"], r["property"]): r["value"] for r in props}
    manifest = {
        "package": "COMSOL_loudspeaker_req7_export_package",
        "created_local_time": datetime.now().isoformat(timespec="seconds"),
        "purpose": "REQ7 COMSOL PML feature, typical wavelength, coordinate stretching/Jacobian, axisymmetric volume factor, and pressure field audit.",
        "pml_domains": [1, 5],
        "pml_coordinate_system_tag": "pml1",
        "pml_key_properties": {
            "ScalingType": prop_map.get(("coordSystem.pml1", "ScalingType")),
            "stretchingType": prop_map.get(("coordSystem.pml1", "stretchingType")),
            "wavelengthSourceType": prop_map.get(("coordSystem.pml1", "wavelengthSourceType")),
            "wavelengthSource": prop_map.get(("coordSystem.pml1", "wavelengthSource")),
            "typicalWavelength": prop_map.get(("coordSystem.pml1", "typicalWavelength")),
            "PMLfactor": prop_map.get(("coordSystem.pml1", "PMLfactor")),
            "PMLgamma": prop_map.get(("coordSystem.pml1", "PMLgamma")),
            "directions": prop_map.get(("coordSystem.pml1", "directions")),
            "d": prop_map.get(("coordSystem.pml1", "d")),
            "dmax": prop_map.get(("coordSystem.pml1", "dmax")),
        },
        "coordinate_note": "COMSOL exposes pml1.r and pml1.z but not explicit J11/detJ variables. Jacobian columns are exported by evaluating d(real/imag(pml1.r/1[m]), r/z) and d(real/imag(pml1.z/1[m]), r/z); determinant, inverse, and axisymmetric radial factor are computed in the export script.",
        "files": summaries,
    }
    write_json(PKG / "export_manifest.json", manifest)
    write_json(PKG / "comsol_model_info.json", {
        "model_file_loaded": "loudspeaker_driver_req2_solved.mph",
        "batch_log": "comsol_req7_export.log",
        "probe_log": "comsol_req7_probe.log",
        "source_java": "ComsolReq7Export.java",
        "probe_java": "ComsolReq7Probe.java",
        "generated_work_copies_not_packaged": ["ComsolReq7Export_req7.mph", "ComsolReq7Probe_req7probe.mph"],
    })


def write_readme() -> None:
    (PKG / "README_REQ7_EXPORT.txt").write_text(
        """COMSOL loudspeaker req7 export package

Purpose:
- Audit COMSOL PML configuration and transformed-coordinate behavior for domains 1 and 5.

Core files:
- pml_feature_properties.csv
- pml_typical_wavelength_vs_frequency.csv
- pml_coordinate_jacobian_points.csv
- pml_pressure_linecuts.csv
- pml_expression_probe.csv
- pml_source_model_snippet.csv

Key settings found:
- PML coordinate system tag: pml1
- PML domains: 1, 5 via selection sel8
- ScalingType=Spherical
- stretchingType=polynomial
- wavelengthSourceType=fromPhysics
- wavelengthSource=acpr
- PMLfactor=1
- PMLgamma=3

Notes:
- COMSOL did not expose explicit J11/detJ variable names in result evaluation. The package exports pml1.r/pml1.z and derivative-based Jacobian entries instead.
- typical_wavelength_m is explicitly converted to meters using (2*pi/acpr.k)/1[m].
""",
        encoding="utf-8",
    )


def main() -> None:
    if PKG.exists():
        shutil.rmtree(PKG)
    PKG.mkdir()
    shutil.copytree(RAW, PKG / "comsol_req7_raw")
    for name in [
        "要求7.txt",
        "ComsolReq7Export.java",
        "ComsolReq7Probe.java",
        "build_req7_package.py",
        "comsol_req7_export.log",
        "comsol_req7_probe.log",
        "comsol_req7_probe_stdout.txt",
        "ComsolReq7Export.class.status",
        "ComsolReq7Probe.class.status",
    ]:
        p = ROOT / name
        if p.exists():
            shutil.copy2(p, PKG / name)

    summaries = [summarize_csv(p) for p in sorted(RAW.glob("*.csv"))]
    write_validation_summary(summaries)
    write_numeric_summaries()
    write_manifests(summaries)
    write_readme()

    if ZIP.exists():
        ZIP.unlink()
    with ZipFile(ZIP, "w", ZIP_DEFLATED) as zf:
        for p in PKG.rglob("*"):
            zf.write(p, p.relative_to(PKG.parent))
    print(ZIP)


if __name__ == "__main__":
    main()
