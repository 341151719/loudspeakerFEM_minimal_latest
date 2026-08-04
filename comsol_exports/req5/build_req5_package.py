from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "comsol_req5_raw"
PKG = ROOT / "COMSOL_loudspeaker_req5_export_package"
ZIP = ROOT / "COMSOL_loudspeaker_req5_export_package.zip"

REQUESTED_CORE_FREQS = [20, 50, 100, 600, 630, 1000, 1300, 2000, 5000, 6300, 8000]
SOLVED_CORE_FREQS = [20, 50, 100, 600, 630, 1000, 1320, 2000, 5000, 6300, 8000]

LAYER_DESCRIPTIONS = {
    "layer00": "L0 geometry, mesh-node, selection, boundary coordinate and normal mapping.",
    "layer01": "L1 magnetostatics field state and magnetic boundary samples.",
    "layer02": "L2 induction current fields and blocked impedance full sweep.",
    "layer03": "L3 coil electrical input, blocked/total/motional impedance and axis pressure.",
    "layer04": "L4 Lorentz source density over voice-coil domains.",
    "layer05": "L5 solid displacement, velocity, acceleration and ASB structural state.",
    "layer06": "L6 ASB acoustic-structure coupling point fields and boundary work integrals.",
    "layer07": "L7 acoustic/NRA/PML pressure, gradients and velocity-proxy fields.",
    "layer08": "L8 Boundary93 source layer p, gradient, dpdn, vn, weights and intensity.",
    "layer09": "L9 COMSOL exterior-field far-field/directivity complex matrix.",
}


def csv_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    freqs: list[float] = []
    if rows and "freq_Hz" in rows[0]:
        vals = set()
        for row in rows:
            try:
                vals.add(round(float(row["freq_Hz"]), 9))
            except (KeyError, TypeError, ValueError):
                pass
        freqs = sorted(vals)
    bounds: list[int] = []
    if rows and "boundary_id" in rows[0]:
        vals = set()
        for row in rows:
            try:
                vals.add(int(float(row["boundary_id"])))
            except (KeyError, TypeError, ValueError):
                pass
        bounds = sorted(vals)
    domains: list[int] = []
    if rows and "domain_id" in rows[0]:
        vals = set()
        for row in rows:
            try:
                vals.add(int(float(row["domain_id"])))
            except (KeyError, TypeError, ValueError):
                pass
        domains = sorted(vals)
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "rows": len(rows),
        "columns": reader.fieldnames or [],
        "freq_Hz_values": freqs,
        "boundary_ids": bounds,
        "domain_ids": domains,
    }


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def layer_key(name: str) -> str:
    return name.split("_", 1)[0]


def build_manifests() -> None:
    csv_files = sorted(RAW.glob("*.csv"))
    summaries = [csv_summary(p) for p in csv_files]
    files_by_layer: dict[str, list[str]] = {}
    for s in summaries:
        files_by_layer.setdefault(layer_key(s["file"]), []).append(s["file"])

    export_manifest = {
        "package": "COMSOL_loudspeaker_req5_export_package",
        "created_local_time": datetime.now().isoformat(timespec="seconds"),
        "purpose": "REQ5 COMSOL intermediate-layer black-box decomposition for L0-L9 contracts.",
        "requested_core_freqs_Hz": REQUESTED_CORE_FREQS,
        "solved_core_freqs_Hz": SOLVED_CORE_FREQS,
        "frequency_note": "COMSOL solved set has 1320 Hz; requested 1300 Hz is exported as nearest solved frequency 1320 Hz.",
        "format_note": "CSV is used as the primary exchange format because several files are large; XLSX was intentionally not generated.",
        "layers": LAYER_DESCRIPTIONS,
        "files": summaries,
        "files_by_layer": files_by_layer,
    }
    write_json(PKG / "export_manifest.json", export_manifest)

    expression_manifest = {
        "complex_format": "Complex quantities are split into *_real and *_imag columns; magnitudes/phases are explicit where exported.",
        "axisymmetric_weight": "Boundary node quadrature proxy is axisym_weight_2pi_r_ds = 2*pi*r*ds from COMSOL boundary node ordering.",
        "files": {
            s["file"]: {
                "layer": layer_key(s["file"]),
                "description": LAYER_DESCRIPTIONS.get(layer_key(s["file"]), ""),
                "columns": s["columns"],
            }
            for s in summaries
        },
    }
    write_json(PKG / "expression_manifest.json", expression_manifest)

    selection_manifest = {
        "asb_boundary_ids": [15, 16, 17, 19, 21, 23, 25, 27, 31, 32, 34, 39, 40, 42, 43, 44, 46, 47, 48, 49, 51, 56, 57, 61, 62, 63, 64, 66, 67, 68, 69, 74, 75, 77, 78, 79, 80, 81, 91, 92, 99, 100, 101, 102],
        "boundary93": 93,
        "acoustic_domain_ids": [1, 3, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 21, 22, 25],
        "pml_domain_ids": [1, 5],
        "nra_domain_ids": [8, 22],
        "solid_domain_ids": [3, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 25],
        "coil_domain_ids": [17, 18, 19],
        "soft_iron_domain_ids": [6, 23],
        "inventory_source": "layer00_model_inventory.csv",
    }
    write_json(PKG / "selection_manifest.json", selection_manifest)

    mesh_manifest = {
        "mesh_nodes_file": "comsol_req5_raw/layer00_mesh_nodes.csv",
        "boundary_samples_file": "comsol_req5_raw/layer00_boundary_sample_points.csv",
        "geometry_boundaries_file": "comsol_req5_raw/layer00_geometry_boundaries.csv",
        "element_connectivity_file": "comsol_req5_raw/layer00_mesh_elements.csv",
        "element_connectivity_note": "Element connectivity was not exposed by the COMSOL numerical result Eval API used for this batch export. Node coordinates, boundary normals, arc length, and axisymmetric weights are exported.",
    }
    write_json(PKG / "mesh_manifest.json", mesh_manifest)

    model_info = {
        "model_file_loaded": "loudspeaker_driver_req2_solved.mph",
        "batch_log": "comsol_req5_export.log",
        "generated_work_copy_not_packaged": "ComsolReq5Export_req5.mph",
        "comsol_inventory_csv": "comsol_req5_raw/layer00_model_inventory.csv",
        "source_java": "ComsolReq5Export.java",
    }
    write_json(PKG / "comsol_model_info.json", model_info)

    with (PKG / "validation_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["file", "rows", "cols", "freq_count", "boundary_count", "domain_count", "bytes"])
        for s in summaries:
            writer.writerow([s["file"], s["rows"], len(s["columns"]), len(s["freq_Hz_values"]), len(s["boundary_ids"]), len(s["domain_ids"]), s["bytes"]])


def write_readme() -> None:
    (PKG / "README_REQ5_EXPORT.txt").write_text(
        """COMSOL loudspeaker req5 export package

Purpose:
- Complete L0-L9 COMSOL intermediate-layer black-box decomposition for Stage20/REQ5 auditing.

Primary format:
- CSV only. XLSX is intentionally skipped because the solid/Lorentz/acoustic point-cloud files are large.

Core coverage:
- L0 geometry/mesh-node/boundary-normal mapping.
- L1 magnetostatics.
- L2 induction current and blocked impedance.
- L3 total/blocked/motional impedance and power decomposition.
- L4 Lorentz force density.
- L5 solid displacement/velocity/acceleration, including 6300 Hz.
- L6 ASB p/u_n/v_n/a_n and complex work, including boundary integrals.
- L7 acoustic/NRA/PML pressure and gradient fields.
- L8 Boundary93 p/grad/dpdn/vn/source weights.
- L9 COMSOL exterior far-field complex directivity.

Frequency note:
- Requested 1300 Hz is exported as the nearest solved COMSOL frequency, 1320 Hz, consistently across layers.

Manifests:
- export_manifest.json
- expression_manifest.json
- selection_manifest.json
- mesh_manifest.json
- comsol_model_info.json

The large COMSOL work copy ComsolReq5Export_req5.mph is not packaged; it was automatically saved by COMSOL batch and is not needed for data review.
""",
        encoding="utf-8",
    )


def main() -> None:
    if not RAW.exists():
        raise SystemExit(f"Missing raw directory: {RAW}")
    if PKG.exists():
        shutil.rmtree(PKG)
    PKG.mkdir()

    shutil.copytree(RAW, PKG / "comsol_req5_raw")
    for name in [
        "要求5.txt",
        "ComsolReq5Export.java",
        "build_req5_package.py",
        "comsol_req5_export.log",
        "ComsolReq5Export.class.status",
    ]:
        p = ROOT / name
        if p.exists():
            shutil.copy2(p, PKG / name)

    build_manifests()
    write_readme()

    if ZIP.exists():
        ZIP.unlink()
    with ZipFile(ZIP, "w", ZIP_DEFLATED) as zf:
        for p in PKG.rglob("*"):
            zf.write(p, p.relative_to(PKG.parent))
    print(ZIP)


if __name__ == "__main__":
    main()
