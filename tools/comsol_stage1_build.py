#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from loudspeaker_axisym_fem.comsol_geom_mphtxt import parse_mphtxt, DOMAIN_SELECTIONS_FROM_M, BOUNDARY_SELECTIONS_FROM_M
from loudspeaker_axisym_fem.comsol_mfile_inventory import parse_mfile_inventory
from loudspeaker_axisym_fem.comsol_feature_matrix import write_feature_matrix
from loudspeaker_axisym_fem.comsol_driver_model import ComsolDriverParameters, MATERIALS, COMSOL_TARGETS, SOFT_IRON_BH_TABLE
from loudspeaker_axisym_fem.axisym_magnetics import skin_depth_m, effective_mu_r, differential_mu_r
from loudspeaker_axisym_fem.json_utils import write_json, dumps_json


def main():
    ap = argparse.ArgumentParser(description="Build stage-1 COMSOL loudspeaker reproduction inventory and geometry artifacts.")
    ap.add_argument("--mphtxt", default=str(ROOT / "comsol_reference_inputs" / "Untitled.mphtxt"))
    ap.add_argument("--mfile", default=str(ROOT / "comsol_reference_inputs" / "loudspeaker_driver_exported.m"))
    ap.add_argument("--outdir", default=str(ROOT / "outputs_comsol_reproduction_stage1"))
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    geom = parse_mphtxt(args.mphtxt)
    geom.write_inventory(outdir / "comsol_geometry_inventory.json")
    geom.plot_edges(outdir / "comsol_geometry_edges_domains.png")
    geom.export_geo_polyline(outdir / "comsol_geometry_polyline_debug.geo")

    m_inv = parse_mfile_inventory(args.mfile)
    write_json(outdir / "comsol_mfile_inventory.json", m_inv, indent=2)

    domain_map = {
        "domain_groups_from_m": DOMAIN_SELECTIONS_FROM_M,
        "boundary_groups_from_m": BOUNDARY_SELECTIONS_FROM_M,
        "materials": MATERIALS,
        "targets": COMSOL_TARGETS,
    }
    write_json(outdir / "comsol_domain_boundary_map.json", domain_map, indent=2)

    write_feature_matrix(outdir / "COMSOL_FEATURE_MATRIX.md")

    params = ComsolDriverParameters()
    skin = {
        "f_Hz": 8000.0,
        "sigma_soft_iron_S_m": 1.12e7,
        "mu_r_peak_assumed": 1200.0,
        "skin_depth_m": skin_depth_m(8000.0, 1.12e7, 1200.0),
        "skin_depth_mm": 1000.0 * skin_depth_m(8000.0, 1.12e7, 1200.0),
        "comment": "COMSOL PDF states the skin depth at 8 kHz does not go below about 0.05 mm; this reproduces that scale.",
    }
    bh_samples = []
    for H in [663.146, 1067.5, 7957.75, 61213.4, 347507.836]:
        bh_samples.append({
            "H_A_m": H,
            "mu_eff_r": float(effective_mu_r(H, SOFT_IRON_BH_TABLE)),
            "mu_diff_r": float(differential_mu_r(H, SOFT_IRON_BH_TABLE)),
        })
    physics_seeds = {
        "parameters": {
            "N0": params.N0,
            "V0_peak_V": params.V0_peak_V,
            "f_loss_Hz": params.f_loss_Hz,
            "omega_loss_rad_s": params.omega_loss,
            "fmax_Hz": params.fmax_Hz,
            "c0_m_s": params.c0_m_s,
            "lam0_m": params.lam0_m,
        },
        "soft_iron_skin_depth_check": skin,
        "soft_iron_BH_samples": bh_samples,
        "targets": COMSOL_TARGETS,
    }
    write_json(outdir / "comsol_physics_seed_checks.json", physics_seeds, indent=2)

    summary = {
        "stage": "stage-1 geometry/inventory project scaffold",
        "outdir": str(outdir),
        "geometry": geom.inventory(),
        "mfile_key_physics": m_inv.get("physics"),
        "mfile_key_multiphysics": m_inv.get("multiphysics"),
        "key_boundaries": m_inv.get("key_boundaries"),
        "skin_depth_mm_8kHz_mu1200": skin["skin_depth_mm"],
        "next_stage": [
            "assemble nonlinear A_phi magnetostatics on COMSOL domains 6/23/24",
            "calibrate BL integral to 10.48 N/A",
            "implement blocked impedance perturbation and compare Figure 5/6",
            "add axisymmetric solid mechanics and structural eigenmodes",
            "couple pressure acoustics to solid and reproduce Figure 8/10/12",
        ],
    }
    write_json(outdir / "stage1_summary.json", summary, indent=2)
    print(dumps_json(summary, indent=2))


if __name__ == "__main__":
    main()
