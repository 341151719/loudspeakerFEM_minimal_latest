import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_native_blocked_runtime_has_no_comsol_table_dependency():
    cfg = json.loads((ROOT / "configs/best_model.json").read_text())
    b = cfg["blocked_coil"]
    assert b["mode"] == "native_voltage_constrained"
    assert b["runtime_COMSOL_table_required"] is False
    assert b["runtime_mode"] == "native_field"
    assert b["reference_identified"] is False
    assert b["linearized_mu_mode"] == "anisotropic_tangent"
    assert b["tangent_anisotropy_factor"] == 1.0
    assert b["field_mesh"].endswith("native_mqs_hybrid_bl_c48.msh")
    assert len(b["surrogate_R_ohm_chebyshev"]) == b["surrogate_degree"] + 1
    assert len(b["surrogate_L_H_chebyshev"]) == b["surrogate_degree"] + 1
    assert b["sigma_soft_iron_S_m"] == 1.12e7
    assert b["subgrid_closure_enabled"] is False


def test_boundary93_diagnostic_cannot_silently_replace_physical_hk():
    cfg = json.loads((ROOT / "configs/best_model.json").read_text())
    assert cfg["exterior"]["req6_ppr_parity"]["apply_to_hk"] is False
    p = ROOT / "inputs/reference_fields/req6_boundary93_parity_spline.json"
    d = json.loads(p.read_text())
    assert d["apply_to_HK_default"] is False


def test_stage33_required_sources_exist():
    required = [
        ROOT / "best_model/native_blocked_coil.py",
        ROOT / "best_model/interface_recovery.py",
        ROOT / "best_model/boundary93_parity.py",
        ROOT / "best_model/eigenmodes.py",
        ROOT / "comsol_exports/req10_figure5/ComsolReq10Figure5Export.java",
        ROOT / "comsol_exports/req11_eigen_mac/ComsolReq11EigenMacExport.java",
        ROOT / "README_CN.md",
    ]
    assert not [str(p) for p in required if not p.exists()]
