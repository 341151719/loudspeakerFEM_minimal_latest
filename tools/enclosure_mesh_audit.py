#!/usr/bin/env python3
"""Run the deterministic L0 enclosure mesh topology audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from loudspeaker_axisym_fem.enclosure_geometry import (  # noqa: E402
    REFERENCE_CONFIGS,
    generate_reference_mesh,
)
from loudspeaker_axisym_fem.enclosure_topology import (  # noqa: E402
    MeshTopologyAuditError,
    audit_mesh,
    write_audit_json,
)


def _config_paths(config_dir: Path) -> dict[str, Path]:
    return {
        case: config_dir / filename
        for case, filename in sorted(REFERENCE_CONFIGS.items())
    }


def _audit_one(
    case: str,
    mesh_path: Path,
    config_path: Path,
) -> dict[str, Any]:
    return audit_mesh(mesh_path, case_id=case, config_path=config_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mesh", type=Path, help="one existing L0 .msh file")
    mode.add_argument("--all", action="store_true", help="audit A-E meshes")
    parser.add_argument("--case", choices=tuple("ABCDE"), help="case ID for --mesh")
    parser.add_argument("--config", type=Path, help="validated config for --mesh")
    parser.add_argument(
        "--generate",
        action="store_true",
        help="generate missing/reference L0 meshes into --mesh-dir before auditing",
    )
    parser.add_argument(
        "--mesh-dir",
        type=Path,
        default=Path("/tmp/luna_enclosure_l0"),
        help="existing/generated mesh directory for --all (default: /tmp/luna_enclosure_l0)",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=ROOT / "configs" / "enclosures",
        help="phase-1 enclosure config directory",
    )
    parser.add_argument("--output", "-o", type=Path, help="optional JSON output path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    configs = _config_paths(args.config_dir)

    if args.mesh is not None:
        if args.case is None:
            raise SystemExit("--case is required with --mesh")
        case = args.case.upper()
        config_path = args.config or configs[case]
        mesh_path = args.mesh
        if args.generate:
            mesh_path.parent.mkdir(parents=True, exist_ok=True)
            generate_reference_mesh(config_path, "L0", mesh_path)
        report: Any = _audit_one(case, mesh_path, config_path)
    else:
        if args.case is not None or args.config is not None:
            raise SystemExit("--case/--config apply only to --mesh")
        reports: dict[str, Any] = {}
        args.mesh_dir.mkdir(parents=True, exist_ok=True) if args.generate else None
        for case in sorted(configs):
            mesh_path = args.mesh_dir / f"{case}.msh"
            if args.generate:
                generate_reference_mesh(configs[case], "L0", mesh_path)
            if not mesh_path.exists():
                raise SystemExit(f"missing mesh for case {case}: {mesh_path}")
            reports[case] = _audit_one(case, mesh_path, configs[case])
        report = {
            "schema": "luna.enclosure_mesh_topology_audit.batch.v1",
            "status": "pass" if all(item["status"] == "pass" for item in reports.values()) else "fail",
            "cases": reports,
        }

    serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        write_audit_json(report, args.output)
    print(serialized, end="")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MeshTopologyAuditError as exc:
        raise SystemExit(f"enclosure mesh audit error: {exc}") from exc
