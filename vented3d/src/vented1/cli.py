from __future__ import annotations

import argparse
import copy
import json
import sys
import traceback
from pathlib import Path

import numpy as np

from .core import (
    configuration, missing_inputs, provenance, safe_output, atomic_json,
    atomic_npz, parse_frequencies, clean
)
from .mesh import load, number_spaces
from .model import build, describe
from .elements import Motor
from .engine import initialize_output, solve_frequency, structural_modes, export_fields


def parser():
    p = argparse.ArgumentParser(prog="vented1")
    p.add_argument("--config", type=Path, default=Path("configs/benchmark.json"))
    p.add_argument("--root")
    p.add_argument("--out", type=Path)
    p.add_argument("--level", type=int, choices=(0, 1, 2))
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("inspect")
    sub.add_parser("audit")
    solve = sub.add_parser("solve")
    solve.add_argument("--frequency", type=float, required=True)
    solve.add_argument("--voltage", type=complex)
    solve.add_argument("--resume", action="store_true")
    sweep = sub.add_parser("sweep")
    sweep.add_argument("--frequencies", default="selected")
    sweep.add_argument("--voltage", type=complex)
    sweep.add_argument("--resume", action="store_true")
    modes = sub.add_parser("modes")
    modes.add_argument("--count", type=int, default=20)
    compare = sub.add_parser("compare")
    compare.add_argument("--references", type=Path, required=True)
    exp = sub.add_parser("export")
    group = exp.add_mutually_exclusive_group(required=True)
    group.add_argument("--frequency", type=float)
    group.add_argument("--mode", type=int)
    conv = sub.add_parser("convergence")
    conv.add_argument("--levels", default="0,1,2")
    conv.add_argument("--frequencies", default="selected")
    conv.add_argument("--resume", action="store_true")
    analytic = sub.add_parser("analytic")
    analytic.add_argument("--levels", default="1,2")
    analytic.add_argument("--ka", type=float, default=1.)
    analytic.add_argument("--half", action="store_true")
    native = sub.add_parser("native")
    native.add_argument("--file", type=Path, required=True)
    native.add_argument("--dofs", type=Path, required=True)
    native.add_argument("--components", default="u,v,w,u2,v2,w2,arx,ary,arz")
    return p


def main():
    args = parser().parse_args()
    cfg, root = configuration(args.config, args.root, args.level)
    output = args.out.resolve() if args.out else None
    try:
        if args.command == "analytic":
            from .bem import sphere_check
            results = [sphere_check(int(level), args.ka, cfg["bem"], args.half)
                       for level in args.levels.split(",")]
            print(json.dumps(clean(results), indent=2))
            if output:
                output = safe_output(root, output)
                atomic_json(output/"analytic_sphere.json", results)
            return
        if args.command == "native":
            if output is None:
                raise ValueError("--out is required")
            output = safe_output(root, output)
            from .reference import native_vector
            groups = native_vector(args.file, args.dofs, set(args.components.split(",")))
            for i, (component, arrays) in enumerate(groups.items()):
                atomic_npz(output/f"component_{i:03d}.npz", **arrays)
            atomic_json(output/"native_components.json", {
                "components": list(groups),
                "note": "Coordinates are metres; raw node/vector indices are not geometry DOFs"
            })
            return
        if args.command in ("compare", "export"):
            if output is None:
                raise ValueError("--out must name an existing independent result directory")
            if args.command == "compare":
                from .reference import compare
                compare(output, args.references.resolve(), cfg)
            else:
                export_fields(output, args.frequency, args.mode)
            return
        missing = missing_inputs(root)
        if missing:
            report = {"model_inputs_available": False, "root": root,
                      "missing": missing, "solve_allowed": False}
            print(json.dumps(clean(report), indent=2))
            if output:
                output = safe_output(root, output)
                atomic_json(output/"inspection.json", report)
            if args.command == "inspect":
                return
            raise FileNotFoundError("Required model inputs are missing")
        if args.command == "convergence":
            if output is None:
                raise ValueError("--out is required")
            output = safe_output(root, output)
            levels = [int(v) for v in args.levels.split(",")]
            if levels != sorted(set(levels)) or any(v not in (0, 1, 2) for v in levels):
                raise ValueError("Use increasing distinct levels chosen from 0,1,2")
            frequencies = parse_frequencies(args.frequencies)
            from .reference import convergence_summary
            for level in levels:
                local = copy.deepcopy(cfg)
                local["level"] = level
                destination = output/f"L{level}"
                destination.mkdir(exist_ok=True)
                prov = provenance(root, local)
                model = build(load(root, local), local)
                initialize_output(model, destination, prov)
                motor = Motor.load(root, local["strict_bundle"])
                voltage = complex(*local["voltage_peak_V"])
                for f in frequencies:
                    solve_frequency(model, motor, float(f), voltage, local, destination,
                                    prov["compatibility_hash"], args.resume)
                del model
            convergence_summary(output, levels, frequencies)
            return
        mesh = load(root, cfg)
        if args.command == "inspect":
            report = describe(mesh, number_spaces(mesh), cfg)
            report.update(model_inputs_available=True, reference_results_read=False)
            print(json.dumps(clean(report), indent=2))
            if output:
                output = safe_output(root, output)
                atomic_json(output/"inspection.json", report)
            return
        if output is None:
            raise ValueError("--out is required for assembly and solves")
        output = safe_output(root, output)
        prov = provenance(root, cfg)
        model = build(mesh, cfg)
        initialize_output(model, output, prov)
        if args.command == "audit":
            print(json.dumps(clean(model.report), indent=2))
            return
        if args.command == "modes":
            structural_modes(model, cfg, output, args.count)
            return
        motor = Motor.load(root, cfg["strict_bundle"])
        voltage = args.voltage if args.voltage is not None else complex(*cfg["voltage_peak_V"])
        frequencies = ([args.frequency] if args.command == "solve"
                       else parse_frequencies(args.frequencies))
        for f in frequencies:
            result = solve_frequency(model, motor, float(f), voltage, cfg, output,
                                     prov["compatibility_hash"], args.resume)
            print(f"{f:g} Hz: solver_success={result['solver_success']}")
        atomic_json(output/"requested_run_complete.json", {
            "frequencies_Hz": list(frequencies), "solver_success": True,
            "benchmark_acceptance": None, "mesh_convergence_acceptance": None,
            "compatibility_hash": prov["compatibility_hash"]
        })
    except Exception as exc:
        diagnostic = {
            "command": args.command, "exception": type(exc).__name__,
            "message": str(exc), "traceback": traceback.format_exc(),
            "solver_success": False
        }
        if output is not None and output.exists():
            # This file never replaces a successful frequency checkpoint.
            atomic_json(output/"last_failure.json", diagnostic)
        print(diagnostic["traceback"], file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
