#!/usr/bin/env python3
"""Run a temporary reference A/B Stage 3B2 comparison scan.

The command copies formal configs into a temporary work directory before any
geometry variant is applied.  It writes only temporary meshes and requested
JSON/CSV reports; it never writes ``runs`` or repository inputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from loudspeaker_axisym_fem.enclosure_validation import (  # noqa: E402
    make_temporary_geometry_config,
    scan_reference_cases,
    write_validation_outputs,
)


FORMAL_CONFIGS = {
    "A": ROOT / "configs" / "enclosures" / "open_back.json",
    "B": ROOT / "configs" / "enclosures" / "sealed_lossless.json",
}


def _csv_values(raw: str, cast):
    values = [cast(item.strip()) for item in str(raw).split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("comma-separated value list cannot be empty")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default="A,B", help="reference cases, currently A,B")
    parser.add_argument("--levels", default="L0,L1,L2", help="comma-separated L0/L1/L2")
    parser.add_argument("--frequencies", default="10,20,50,100", help="comma-separated Hz")
    parser.add_argument("--alpha", default="4", help="explicit alpha or comma-separated alphas")
    parser.add_argument(
        "--work-dir",
        default="/tmp/luna_enclosure_stage3b2_compare",
        help="temporary config/mesh directory",
    )
    parser.add_argument("--output-json", default=None, help="JSON report path")
    parser.add_argument("--output-csv", default=None, help="CSV report path")
    parser.add_argument("--geometry-variant", default="base", help="temporary variant label")
    parser.add_argument("--pml-inner-radius", type=float, default=None, help="temporary R0 in m")
    parser.add_argument("--pml-thickness", type=float, default=None, help="temporary PML thickness in m")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cases = tuple(item.strip().upper() for item in args.cases.split(",") if item.strip())
    if not cases or any(case not in {"A", "B"} for case in cases):
        raise SystemExit("--cases must select A and/or B")
    cases = tuple(case for case in ("A", "B") if case in cases)
    levels = _csv_values(args.levels, str)
    frequencies = _csv_values(args.frequencies, float)
    alphas = _csv_values(args.alpha, float)
    work_dir = Path(args.work_dir)
    config_dir = work_dir / "configs"
    mesh_dir = work_dir / "meshes"
    config_paths = {
        case: make_temporary_geometry_config(
            FORMAL_CONFIGS[case],
            config_dir,
            variant=args.geometry_variant,
            pml_inner_radius_m=args.pml_inner_radius,
            pml_thickness_m=args.pml_thickness,
        )
        for case in cases
    }
    report = scan_reference_cases(
        config_paths,
        levels,
        frequencies,
        mesh_dir=mesh_dir,
        pml_alphas=alphas,
    )
    json_path = Path(args.output_json) if args.output_json else work_dir / "validation.json"
    csv_path = Path(args.output_csv) if args.output_csv else work_dir / "validation.csv"
    write_validation_outputs(report, json_path, csv_path)
    print(f"JSON={json_path}")
    print(f"CSV={csv_path}")
    print(f"SOLVES={report['timing']['solve_count']}")
    print(f"WALL_S={report['timing']['wall_time_s']:.3f}")
    print(f"REFERENCE_ALPHA={report['reference_alpha']}")
    print(f"GRID_GATE={report['acceptance']['grid_gate']}")
    slope = report["acceptance"]["far_field_dipole_slope_dB_per_decade"]
    print(f"FAR_FIELD_SLOPE_DB_PER_DECADE={slope}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
