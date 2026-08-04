from __future__ import annotations

from pathlib import Path
from typing import List, Dict

FEATURES: List[Dict[str, str]] = [
    {"area":"Geometry", "comsol":"Final 2D axisymmetric Geom2 object, 25 domains, 102 boundaries, Bezier curves", "python":"Implemented parser for mphtxt; physical domain/boundary inventory and debug .geo export", "status":"stage-1 implemented", "validation":"Domain/boundary IDs preserved; geometry check plot"},
    {"area":"Selections", "comsol":"PML, Soft Iron, Composite, Cloth, Foam, Coil, Glass Fiber, Ferrite, Air, Structural Domains", "python":"Hard-coded from .m and cross-checked against mphtxt adjacency", "status":"stage-1 implemented", "validation":"comsol_domain_boundary_map.json"},
    {"area":"Magnetostatics", "comsol":"mf static field, remanent ferrite 0.4 T, nonlinear soft-iron B-H", "python":"Next module axisym_magnetics.py skeleton; B-H tables extractable", "status":"planned next", "validation":"Fig. 3 H field, Fig. 4 mu_eff, BL=10.48 N/A"},
    {"area":"Blocked impedance", "comsol":"Frequency-domain perturbation, eddy currents, coil Zb, mf.LCoil_1/RCoil_1", "python":"planned", "status":"not yet solved", "validation":"Fig. 5 induced currents; Fig. 6 blocked inductance"},
    {"area":"Solid mechanics", "comsol":"Axisymmetric solid for structural domains with damping and fixed boundaries 81/85", "python":"planned axisymmetric elasticity module", "status":"not yet solved", "validation":"Fig. 7 displacement, Fig. 11 modes"},
    {"area":"Acoustics", "comsol":"Pressure acoustics in Air, PML, exterior field Boundary 93", "python":"existing pressure acoustics/facet-HK to be connected to COMSOL geometry", "status":"partial", "validation":"Fig. 7 SPL, Fig. 8 sensitivity, Fig. 12 directivity"},
    {"area":"Narrow Region Acoustics", "comsol":"Domain 8 slit 0.4 mm, Domain 22 slit 0.2 mm", "python":"planned equivalent complex duct-property model", "status":"not yet solved", "validation":"Fig. 8 with/without NRA and Fig. 9 600/630 Hz back cavity pressure"},
    {"area":"Lorentz/back EMF", "comsol":"Magnetomechanics on coil; only Lorentz force; back EMF through coil coupling", "python":"planned coupled matrix term BL*V0/Zb - BL^2/Zb*v", "status":"not yet solved", "validation":"Total impedance Fig. 10 and sensitivity Fig. 8"},
    {"area":"Studies", "comsol":"Study 1 magnetic fields, Study 2 complete, Study 3 without NRA, Study 4 eigenfrequency", "python":"planned study runner with chunked frequencies", "status":"partial infrastructure", "validation":"all Figure 3-12 outputs"},
    {"area":"Postprocessing", "comsol":"Sensitivity, phase, impedance, directivity, coil power/efficiency", "python":"existing plots extended by COMSOL comparison tool", "status":"stage-1 docs", "validation":"PDF figure-by-figure comparison report"},
]


def write_feature_matrix(path: str | Path) -> None:
    rows = ["# COMSOL Loudspeaker Driver 复现功能矩阵", "", "| Area | COMSOL 功能 | Python 当前实现 | 状态 | 验证目标 |", "|---|---|---|---|---|"]
    for f in FEATURES:
        rows.append(f"| {f['area']} | {f['comsol']} | {f['python']} | {f['status']} | {f['validation']} |")
    Path(path).write_text("\n".join(rows) + "\n", encoding="utf-8")
