#!/usr/bin/env python3
"""Run the deterministic enclosure mesh topology audit at one or all levels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import sys
from time import perf_counter
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


LEVELS = ("L0", "L1", "L2")


def _config_paths(config_dir: Path) -> dict[str, Path]:
    return {
        case: config_dir / filename
        for case, filename in sorted(REFERENCE_CONFIGS.items())
    }


def _rss_mb() -> float | None:
    """Return process peak RSS when the host exposes a reliable ru_maxrss."""

    try:
        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (AttributeError, OSError):
        return None
    # Linux reports KiB; macOS reports bytes.  The runner is Linux, but keep the
    # conversion explicit so a non-Linux invocation reports a sensible value.
    if sys.platform == "darwin":
        return value / (1024.0 * 1024.0)
    return value / 1024.0


def _mesh_path(mesh_dir: Path, case: str, level: str) -> Path:
    return mesh_dir / f"{case}_{level}.msh"


def _run_one(
    case: str,
    level: str,
    mesh_path: Path,
    config_path: Path,
    *,
    generate: bool,
) -> dict[str, Any]:
    started = perf_counter()
    if generate:
        mesh_path.parent.mkdir(parents=True, exist_ok=True)
        generate_reference_mesh(config_path, level, mesh_path)
    report = audit_mesh(mesh_path, case_id=case, config_path=config_path)
    elapsed = perf_counter() - started
    summary = {
        "case_id": case,
        "level": level,
        "elapsed_s": round(float(elapsed), 6),
        "mesh_path": str(mesh_path),
        "points": int(report["mesh_counts"]["points"]),
        "triangle_elements": int(report["mesh_counts"]["triangle_elements"]),
        "air_cavity_volume_m3": float(report["volume_contract"]["air_cavity_volume_m3"]),
        "volume_relative_error": float(report["volume_contract"]["air_cavity_relative_error"]),
        "minimum_triangle_quality": float(report["quality"]["minimum_triangle_quality"]),
        "geometry_signature_sha256": report["source"]["geometry_signature_sha256"],
        "status": report["status"],
        "failures": list(report["failures"]),
    }
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mesh", type=Path, help="one existing .msh file")
    mode.add_argument("--all", action="store_true", help="audit A-E meshes")
    parser.add_argument("--case", choices=tuple("ABCDE"), help="case ID for --mesh")
    parser.add_argument("--config", type=Path, help="validated config for --mesh")
    parser.add_argument("--level", choices=LEVELS, default="L0", help="mesh level (default: L0)")
    parser.add_argument(
        "--all-levels",
        action="store_true",
        help="with --all, audit A-E at L0, L1, and L2",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="generate reference meshes into --mesh-dir before auditing",
    )
    parser.add_argument(
        "--mesh-dir",
        type=Path,
        default=Path("/tmp/luna_enclosure_meshes"),
        help="existing/generated mesh directory (default: /tmp/luna_enclosure_meshes)",
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
    if args.all_levels and args.mesh is not None:
        raise SystemExit("--all-levels requires --all")
    selected_levels = LEVELS if args.all_levels else (args.level,)
    batch_started = perf_counter()

    if args.mesh is not None:
        if args.case is None:
            raise SystemExit("--case is required with --mesh")
        if args.all_levels:
            raise SystemExit("--all-levels requires --all")
        case = args.case.upper()
        config_path = args.config or configs[case]
        mesh_path = args.mesh
        summary = _run_one(case, args.level, mesh_path, config_path, generate=args.generate)
        report: Any = {
            "schema": "luna.enclosure_mesh_topology_audit.summary.v2",
            "status": summary["status"],
            "meshes": [summary],
        }
    else:
        if args.case is not None or args.config is not None:
            raise SystemExit("--case/--config apply only to --mesh")
        if args.generate:
            args.mesh_dir.mkdir(parents=True, exist_ok=True)
        summaries: list[dict[str, Any]] = []
        for case in sorted(configs):
            for level in selected_levels:
                mesh_path = _mesh_path(args.mesh_dir, case, level)
                if not args.generate and not mesh_path.exists():
                    raise SystemExit(f"missing mesh for {case} {level}: {mesh_path}")
                summaries.append(
                    _run_one(
                        case,
                        level,
                        mesh_path,
                        configs[case],
                        generate=args.generate,
                    )
                )
        report = {
            "schema": "luna.enclosure_mesh_topology_audit.batch.v2",
            "status": "pass" if all(item["status"] == "pass" for item in summaries) else "fail",
            "mesh_count": len(summaries),
            "levels": list(selected_levels),
            "meshes": summaries,
        }

    report["wall_time_s"] = round(float(perf_counter() - batch_started), 6)
    report["peak_rss_mb"] = _rss_mb()
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
