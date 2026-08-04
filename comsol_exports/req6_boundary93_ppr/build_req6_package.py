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
RAW = ROOT / "comsol_req6_raw"
PKG = ROOT / "COMSOL_loudspeaker_req6_export_package"
ZIP = ROOT / "COMSOL_loudspeaker_req6_export_package.zip"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def summarize_csv(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    out = {
        "file": path.name,
        "bytes": path.stat().st_size,
        "rows": len(rows),
        "columns": reader.fieldnames or [],
    }
    if rows and "variant_label" in rows[0]:
        out["variants"] = dict(Counter(r["variant_label"] for r in rows))
    if rows and "solved_freq_Hz" in rows[0]:
        vals = sorted({round(float(r["solved_freq_Hz"]), 9) for r in rows if r.get("solved_freq_Hz") and r.get("theta_deg") != "ERROR"})
        out["solved_freq_Hz_values"] = vals
    if rows and "radius_m" in rows[0]:
        vals = sorted({round(float(r["radius_m"]), 9) for r in rows if r.get("radius_m")})
        out["radius_m_values"] = vals
    return out


def write_validation_summary() -> None:
    summaries = [summarize_csv(p) for p in sorted(RAW.glob("*.csv"))]
    with (PKG / "validation_summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["file", "rows", "cols", "bytes", "solved_freq_count", "radius_count", "variants"])
        for s in summaries:
            w.writerow([
                s["file"],
                s["rows"],
                len(s["columns"]),
                s["bytes"],
                len(s.get("solved_freq_Hz_values", [])),
                len(s.get("radius_m_values", [])),
                json.dumps(s.get("variants", {}), ensure_ascii=False),
            ])
    return summaries


def write_operator_summary_for(source_name: str, radius_out: str, variant_out: str, original_label: str = "original") -> None:
    rows = read_rows(RAW / source_name)
    ok = [r for r in rows if r.get("theta_deg") != "ERROR"]
    by_key = defaultdict(list)
    for r in ok:
        by_key[(r["variant_label"], r["solved_freq_Hz"], r["radius_m"])].append(r)

    with (PKG / radius_out).open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["variant", "freq_Hz", "radius_m", "theta_count", "axis_abs_Pa", "axis_SPL_dB", "axis_abs_times_R", "max_rel_dB", "min_rel_dB"])
        for (variant, freq, radius), group in sorted(by_key.items(), key=lambda x: (x[0][0], float(x[0][1]), float(x[0][2]))):
            axis = next((r for r in group if r["theta_deg"] == "0"), None)
            rel = [float(r["SPL_relative_to_0deg_dB"]) for r in group if r.get("SPL_relative_to_0deg_dB")]
            if axis:
                w.writerow([
                    variant, freq, radius, len(group),
                    axis["pext_abs_Pa"], axis["SPL_abs_dB"], axis["amp_times_R"],
                    max(rel) if rel else "", min(rel) if rel else "",
                ])

    original = {}
    for r in ok:
        if r["variant_label"] == original_label:
            original[(r["solved_freq_Hz"], r["radius_m"], r["theta_deg"])] = r

    accum = defaultdict(lambda: {"n": 0, "sq": 0.0, "max": 0.0})
    for r in ok:
        if r["variant_label"] == original_label:
            continue
        base = original.get((r["solved_freq_Hz"], r["radius_m"], r["theta_deg"]))
        if not base:
            continue
        try:
            a = float(r["pext_abs_Pa"])
            b = float(base["pext_abs_Pa"])
            err = 20.0 * math.log10(max(a, 1e-300) / max(b, 1e-300))
        except ValueError:
            continue
        key = (r["variant_label"], r["solved_freq_Hz"], r["radius_m"])
        accum[key]["n"] += 1
        accum[key]["sq"] += err * err
        accum[key]["max"] = max(accum[key]["max"], abs(err))

    with (PKG / variant_out).open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["variant", "freq_Hz", "radius_m", "theta_count", "rmse_abs_dB_vs_original", "max_abs_dB_vs_original"])
        for (variant, freq, radius), d in sorted(accum.items(), key=lambda x: (x[0][0], float(x[0][1]), float(x[0][2]))):
            rmse = math.sqrt(d["sq"] / d["n"]) if d["n"] else ""
            w.writerow([variant, freq, radius, d["n"], rmse, d["max"]])


def write_operator_summary() -> None:
    write_operator_summary_for(
        "req6_pext_radius_theta_complex.csv",
        "operator_radius_scaling_summary.csv",
        "operator_variant_vs_original_summary.csv",
    )
    cloned = RAW / "req6_pext_cloned_variant_rerun_complex.csv"
    if cloned.exists():
        write_operator_summary_for(
            cloned.name,
            "operator_cloned_rerun_radius_scaling_summary.csv",
            "operator_cloned_rerun_variant_vs_original_summary.csv",
            "original_rerun",
        )


def write_manifests(summaries: list[dict]) -> None:
    props = read_rows(RAW / "req6_efc_and_plot_properties.csv")
    key_props = {
        (r["node"], r["property"]): r["value"]
        for r in props
        if r["property"] in {
            "UsePPR", "ReverseNormal", "ExtName", "SymmetryCondition0", "SymmetryCondition2",
            "IntType", "SymmetryType", "expr", "normalization", "normalizationangle", "radius"
        }
    }
    manifest = {
        "package": "COMSOL_loudspeaker_req6_export_package",
        "created_local_time": datetime.now().isoformat(timespec="seconds"),
        "purpose": "REQ6 COMSOL L9 Exterior Field Calculation operator audit.",
        "frequencies_requested_Hz": [100, 1000, 6300],
        "radii_m": [1, 2, 10],
        "theta_deg": "-90..90 step 1",
        "axisymmetric_coordinate_rule": "r=abs(R*sin(theta)), z=R*cos(theta)",
        "original_efc_key_properties": {
            "UsePPR": key_props.get(("acpr.efc1", "UsePPR")),
            "ReverseNormal": key_props.get(("acpr.efc1", "ReverseNormal")),
            "ExtName": key_props.get(("acpr.efc1", "ExtName")),
            "SymmetryCondition0": key_props.get(("acpr.efc1", "SymmetryCondition0")),
            "SymmetryCondition2": key_props.get(("acpr.efc1", "SymmetryCondition2")),
            "IntType": key_props.get(("acpr.efc1", "IntType")),
            "SymmetryType": key_props.get(("acpr.efc1", "SymmetryType")),
        },
        "directivity_plot_key_properties": {
            "expr": key_props.get(("result.pg10.dir1", "expr")),
            "normalization": key_props.get(("result.pg10.dir1", "normalization")),
            "normalizationangle": key_props.get(("result.pg10.dir1", "normalizationangle")),
            "radius": key_props.get(("result.pg10.dir1", "radius")),
        },
        "variant_note": "The first variant file toggles acpr.efc1 in-place. Because those toggles did not change pext numerically without recomputation, an additional cloned-variant rerun was executed: efc2/pext_nosym and efc4/pext_rev were added, Study 2 was rerun, and req6_pext_cloned_variant_rerun_complex.csv contains the recomputed A/B/D operator comparison. Far-field approximation candidate IntType values were rejected by the COMSOL API and the effective IntType remained FullIntegral.",
        "files": summaries,
    }
    write_json(PKG / "export_manifest.json", manifest)

    write_json(PKG / "comsol_model_info.json", {
        "model_file_loaded": "loudspeaker_driver_req2_solved.mph",
        "comsol_batch_log": "comsol_req6_export.log",
        "source_java": "ComsolReq6Export.java",
        "probe_java": "ComsolReq6Probe.java",
        "variant_rerun_java": "ComsolReq6VariantRerun.java",
        "variant_rerun_log": "comsol_req6_variant_rerun.log",
        "generated_work_copies_not_packaged": ["ComsolReq6Export_req6.mph", "ComsolReq6Probe_req6probe.mph", "ComsolReq6VariantRerun_req6var.mph"],
    })


def write_readme() -> None:
    (PKG / "README_REQ6_EXPORT.txt").write_text(
        """COMSOL loudspeaker req6 export package

Purpose:
- Audit the true COMSOL L9 Exterior Field Calculation operator used by this model.

Core files:
- req6_efc_and_plot_properties.csv: EFC and plot-node properties from COMSOL API.
- req6_pext_radius_theta_complex.csv: complex pext(theta,R,f) for original/no_symmetry/reverse_normal and attempted farfield variant.
- req6_pext_cloned_variant_rerun_complex.csv: recomputed pext/pext_nosym/pext_rev after cloned EFC variants were added and Study 2 rerun.
- req6_boundary93_gradient_recovery_audit.csv: plain, ppr, up, and down normal-gradient variants on Boundary93.
- req6_pg10_dir1_plot_export.csv: COMSOL Directivity plot export for the original pg10/dir1 plot.
- operator_radius_scaling_summary.csv and operator_variant_vs_original_summary.csv: compact post-check summaries.

Key COMSOL properties found:
- acpr.efc1 ExtName=pext
- IntType=FullIntegral
- UsePPR=1
- ReverseNormal=0
- SymmetryType=SymmetryPlanes
- SymmetryCondition2=1
- Directivity plot expression=acpr.efc1.Lp_pext
- Directivity normalization=angle, normalizationangle=0

Far-field approximation note:
- Several likely IntType candidate strings were rejected by COMSOL's Java API, so that variant is recorded as unavailable and remains FullIntegral in the effective-properties file.
- The cloned rerun file is the stronger A/B/D dataset for no-symmetry and reverse-normal comparisons.
""",
        encoding="utf-8",
    )


def main() -> None:
    if PKG.exists():
        shutil.rmtree(PKG)
    PKG.mkdir()
    shutil.copytree(RAW, PKG / "comsol_req6_raw")
    for name in [
        "要求6.txt",
        "ComsolReq6Export.java",
        "ComsolReq6Probe.java",
        "ComsolReq6VariantRerun.java",
        "build_req6_package.py",
        "comsol_req6_export.log",
        "comsol_req6_probe.log",
        "comsol_req6_variant_rerun.log",
        "ComsolReq6Export.class.status",
        "ComsolReq6Probe.class.status",
        "ComsolReq6VariantRerun.class.status",
    ]:
        p = ROOT / name
        if p.exists():
            shutil.copy2(p, PKG / name)

    summaries = write_validation_summary()
    write_operator_summary()
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
