#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "best_model"))

from coupled_solver import build_best_model, solve_frequency, solve_sweep, FrequencySolution, load_config
from visualization import write_solution_files, render_solution, write_sweep_metrics, render_sweep
from comparison import compare_impedance_sweep, compare_single_solution
from eigenmodes import solve_p2_eigenmodes, write_eigen_outputs
from loudspeaker_axisym_fem.axisym_magnetics import load_tagged_meshio
from native_blocked_coil import NativeBlockedCoil
from p2_axisym_solid import build_p2_solid


_WORKER_MODEL = None
_WORKER_THREADPOOL_LIMITER = None


def _parallel_worker_init(root: str, config_path: str | None, magnetostatic_vtu: str | None, blas_threads: int) -> None:
    """Build one reusable model per worker and avoid nested BLAS oversubscription."""
    global _WORKER_MODEL, _WORKER_THREADPOOL_LIMITER
    threads = str(max(1, int(blas_threads)))
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = threads
    from threadpoolctl import threadpool_limits
    _WORKER_THREADPOOL_LIMITER = threadpool_limits(limits=int(threads))
    _WORKER_MODEL = build_best_model(root, config_path=config_path, magnetostatic_vtu=magnetostatic_vtu)


def _compact_solution(solution) -> dict:
    return {
        "freq_Hz": float(solution.freq_Hz),
        "current_A_peak": complex(solution.current_A_peak),
        "motional_impedance_ohm": complex(solution.motional_impedance_ohm),
        "total_impedance_ohm": None if solution.total_impedance_ohm is None else complex(solution.total_impedance_ohm),
        "p_axis_1m_Pa_peak": complex(solution.p_axis_1m_Pa_peak),
        "axis_SPL_dB": float(solution.axis_SPL_dB),
    }


def _parallel_solve_one(task: dict) -> dict:
    if _WORKER_MODEL is None:
        raise RuntimeError("parallel worker model was not initialized")
    solution = solve_frequency(
        _WORKER_MODEL,
        task["freq_Hz"],
        drive=task["drive"],
        current_A_peak=task["current_A_peak"],
        voltage_V_peak=task["voltage_V_peak"],
        blocked_impedance_csv=task["blocked_impedance_csv"],
        nra_enabled=task["nra_enabled"],
    )
    if task["save_dir"] is not None:
        write_solution_files(_WORKER_MODEL, solution, task["save_dir"])
    return _compact_solution(solution)


def _recommended_jobs(requested: int, frequency_count: int) -> int:
    if requested > 0:
        return min(int(requested), int(frequency_count))
    cpu = os.cpu_count() or 1
    # A representative solve uses about 0.4 GB RSS here. Keep a conservative
    # 2 GB/worker allowance for high-frequency LU fill-in and model duplication.
    available_gib = None
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        available_gib = pages * page_size / 1024**3
    except (ValueError, OSError, AttributeError):
        pass
    memory_limit = max(1, int(available_gib // 2)) if available_gib is not None else 4
    return max(1, min(int(frequency_count), max(1, cpu // 2), memory_limit, 8))


def parse_freqs(spec: str, cfg: dict) -> list[float]:
    presets = cfg["frequency_presets"]
    if spec in presets and isinstance(presets[spec], list):
        return list(map(float, presets[spec]))
    if spec == "dense_log":
        p = presets["dense_log"]
        return np.geomspace(float(p["start_Hz"]), float(p["stop_Hz"]), int(p["points"])).tolist()
    if spec.startswith("log:"):
        _, a, b, n = spec.split(":")
        return np.geomspace(float(a), float(b), int(n)).tolist()
    p = Path(spec)
    if p.exists():
        vals = []
        for line in p.read_text().replace(",", "\n").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                vals.append(float(line))
        return vals
    return [float(x) for x in spec.replace(";", ",").split(",") if x.strip()]


def cmd_blocked(args):
    import pandas as pd

    cfg = load_config(ROOT, args.config)
    freqs = parse_freqs(args.freqs, cfg)
    bcfg = cfg["blocked_coil"]
    blocked_mesh_path = bcfg.get("field_mesh", cfg["geometry"]["mesh"])
    mesh = load_tagged_meshio(ROOT / blocked_mesh_path)
    vtu = Path(args.magnetostatic_vtu) if args.magnetostatic_vtu else ROOT / bcfg["magnetostatic_vtu"]
    blocker = NativeBlockedCoil.from_vtu(mesh, vtu, bcfg)
    rows = []
    for i, f in enumerate(freqs, 1):
        print(f"[{i}/{len(freqs)}] {f:g} Hz", flush=True)
        if args.raw_field:
            point = blocker.solve(float(f))
            z = point.impedance_ohm
            zraw = point.raw_impedance_ohm
        else:
            z = blocker.impedance(float(f))
            zraw = None
        current = complex(args.voltage) / z
        rows.append({
            "freq_Hz": float(f),
            "Zblocked_real_ohm": z.real,
            "Zblocked_imag_ohm": z.imag,
            "Lblocked_H": z.imag / (2 * np.pi * float(f)),
            "I_real_A": current.real,
            "I_imag_A": current.imag,
            "Zraw_real_ohm": None if zraw is None else zraw.real,
            "Zraw_imag_ohm": None if zraw is None else zraw.imag,
        })
    out = Path(args.outdir or ROOT / "runs" / "blocked_native")
    out.mkdir(parents=True, exist_ok=True)
    output = out / "native_blocked_impedance.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    meta = {
        "n_frequencies": len(rows),
        "frequency_preset": args.freqs,
        "runtime_mode": "raw_native_field_diagnostic" if args.raw_field else bcfg.get("runtime_mode"),
        "reference_identified": False if args.raw_field else bool(bcfg.get("reference_identified", False)),
        "runtime_COMSOL_table_required": False,
        "output": str(output),
    }
    (out / "native_blocked_metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    return 0


def cmd_self_test(args):
    required = [
        ROOT / "inputs/meshes/comsol_geometry_polyline_coarse_2p5mm.msh",
        ROOT / "inputs/comsol_reference/Untitled.mphtxt",
        ROOT / "best_model/p2_axisym_solid.py",
        ROOT / "best_model/p2_pml_operator.py",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("SELF_TEST: FAIL")
        print("\n".join(missing))
        return 2
    from loudspeaker_axisym_fem.axisym_magnetics import load_tagged_meshio
    mesh = load_tagged_meshio(required[0])
    print("SELF_TEST: PASS")
    print(json.dumps({"nodes": mesh.n_nodes, "triangles": mesh.n_triangles, "python": sys.version}, indent=2))
    return 0


def cmd_magnetics(args):
    out = Path(args.outdir or ROOT / "runs/magnetics")
    out.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(ROOT / "tools/comsol_stage2_magnetics.py"),
        "--mesh", str(ROOT / "inputs/meshes/comsol_geometry_polyline_coarse_2p5mm.msh"),
        "--outdir", str(out),
        "--max-iter", str(args.max_iter),
        "--tol", str(args.tol),
        "--relaxation", str(args.relaxation),
        "--remanence-T", str(args.remanence_T),
        "--nonlinear-update-mode", "B_inverse",
    ]
    if args.no_plots:
        command.append("--no-plots")
    print(" ".join(command))
    return subprocess.call(command, cwd=ROOT)


def _model(args):
    return build_best_model(ROOT, config_path=args.config, magnetostatic_vtu=args.magnetostatic_vtu)


def _profile_for_frequency(base_config, hybrid: dict, freq_Hz: float):
    """Route a frequency to low/base/ultrahigh numerical profiles."""
    f = float(freq_Hz)
    if bool(hybrid.get("enabled", False)):
        if f <= float(hybrid.get("crossover_Hz", -np.inf)):
            return hybrid.get("low_frequency_config")
        upper = hybrid.get("upper_frequency_config")
        if upper and f > float(hybrid.get("upper_crossover_Hz", np.inf)):
            return upper
    return base_config


def cmd_solve(args):
    cfg = load_config(ROOT, args.config)
    profile = _profile_for_frequency(args.config, cfg.get("acoustics", {}).get("hybrid_sweep", {}), args.freq)
    model = build_best_model(ROOT, config_path=profile, magnetostatic_vtu=args.magnetostatic_vtu)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=max(1, int(args.blas_threads))):
        solution = solve_frequency(
            model,
            args.freq,
            drive=args.drive,
            current_A_peak=complex(args.current),
            voltage_V_peak=complex(args.voltage),
            blocked_impedance_csv=args.blocked_impedance_csv,
            nra_enabled=not args.without_nra,
        )
    out = Path(args.outdir or ROOT / "runs" / f"solve_{args.freq:g}Hz")
    files = write_solution_files(model, solution, out)
    if args.render:
        render_solution(model, solution, out / "plots", exterior_grid=not args.no_exterior_grid)
    print(json.dumps({
        "freq_Hz": solution.freq_Hz,
        "axis_SPL_dB": solution.axis_SPL_dB,
        "Zmot_ohm": [solution.motional_impedance_ohm.real, solution.motional_impedance_ohm.imag],
        "Ztotal_ohm": None if solution.total_impedance_ohm is None else [solution.total_impedance_ohm.real, solution.total_impedance_ohm.imag],
        "files": {k: str(v) for k, v in files.items()},
    }, indent=2))
    return 0


def cmd_sweep(args):
    cfg = load_config(ROOT, args.config)
    freqs = parse_freqs(args.freqs, cfg)
    hybrid = cfg.get("acoustics", {}).get("hybrid_sweep", {})
    hybrid_enabled = bool(hybrid.get("enabled", False)) and not args.single_profile
    crossover = float(hybrid.get("crossover_Hz", -np.inf))
    low_config = hybrid.get("low_frequency_config")
    upper_crossover = float(hybrid.get("upper_crossover_Hz", np.inf))
    upper_config = hybrid.get("upper_frequency_config")
    if hybrid_enabled and not low_config:
        raise ValueError("enabled acoustics.hybrid_sweep requires low_frequency_config")
    out = Path(args.outdir or ROOT / "runs/sweep")
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "checkpoints"
    checkpoint.mkdir(exist_ok=True)
    jobs = _recommended_jobs(args.jobs, len(freqs))
    solutions = []
    if jobs == 1:
        from threadpoolctl import threadpool_limits
        model = _model(args)
        low_model = (
            build_best_model(ROOT, config_path=low_config, magnetostatic_vtu=args.magnetostatic_vtu)
            if hybrid_enabled and any(float(f) <= crossover for f in freqs)
            else None
        )
        upper_model = (
            build_best_model(ROOT, config_path=upper_config, magnetostatic_vtu=args.magnetostatic_vtu)
            if hybrid_enabled and upper_config and any(float(f) > upper_crossover for f in freqs)
            else None
        )
        with threadpool_limits(limits=max(1, int(args.blas_threads))):
            for i, f in enumerate(freqs, 1):
                print(f"[{i}/{len(freqs)}] {f:g} Hz", flush=True)
                if low_model is not None and float(f) <= crossover:
                    active_model = low_model
                elif upper_model is not None and float(f) > upper_crossover:
                    active_model = upper_model
                else:
                    active_model = model
                sol = solve_frequency(
                    active_model,
                    f,
                    drive=args.drive,
                    current_A_peak=complex(args.current),
                    voltage_V_peak=complex(args.voltage),
                    blocked_impedance_csv=args.blocked_impedance_csv,
                    nra_enabled=not args.without_nra,
                )
                solutions.append(sol)
                if args.save_each:
                    write_solution_files(active_model, sol, checkpoint / f"{f:g}Hz")
    else:
        print(f"Parallel sweep: {jobs} worker processes, {args.blas_threads} BLAS thread(s) per worker", flush=True)
        tasks_by_config: dict[str | None, list[dict]] = {}
        for f in freqs:
            task_config = _profile_for_frequency(args.config, hybrid if hybrid_enabled else {}, float(f))
            tasks_by_config.setdefault(task_config, []).append({
                "freq_Hz": float(f),
                "drive": args.drive,
                "current_A_peak": complex(args.current),
                "voltage_V_peak": complex(args.voltage),
                "blocked_impedance_csv": args.blocked_impedance_csv,
                "nra_enabled": not args.without_nra,
                "save_dir": str(checkpoint / f"{f:g}Hz") if args.save_each else None,
            })
        completed = 0
        compact = []
        for task_config, tasks in tasks_by_config.items():
            phase_jobs = min(jobs, len(tasks))
            profile = task_config or "configs/best_model.json"
            print(f"Profile {profile}: {len(tasks)} frequencies, {phase_jobs} workers", flush=True)
            with ProcessPoolExecutor(
                max_workers=phase_jobs,
                initializer=_parallel_worker_init,
                initargs=(str(ROOT), task_config, args.magnetostatic_vtu, args.blas_threads),
            ) as pool:
                future_to_freq = {pool.submit(_parallel_solve_one, task): task["freq_Hz"] for task in tasks}
                for future in as_completed(future_to_freq):
                    compact.append(future.result())
                    completed += 1
                    print(f"[{completed}/{len(freqs)}] completed {future_to_freq[future]:g} Hz", flush=True)
        compact.sort(key=lambda x: x["freq_Hz"])
        solutions = [SimpleNamespace(**x) for x in compact]
    metrics = write_sweep_metrics(solutions, out)
    if args.render:
        render_sweep(solutions, out / "plots")
    print(json.dumps({
        "n_frequencies": len(freqs),
        "jobs": jobs,
        "hybrid_sweep": hybrid_enabled,
        "hybrid_crossover_Hz": crossover if hybrid_enabled else None,
        "hybrid_upper_crossover_Hz": upper_crossover if hybrid_enabled and upper_config else None,
        "metrics": str(metrics),
    }, indent=2))
    return 0


def _load_solution(npz_path: str | Path) -> FrequencySolution:
    d = np.load(npz_path)
    def scalar(name):
        return d[name].item()
    angles = d["directivity_angles_deg"]
    pdir = d["directivity_pressure_Pa_peak"]
    i0 = int(np.argmin(np.abs(angles)))
    return FrequencySolution(
        float(scalar("freq_Hz")),
        complex(scalar("current_A_peak")),
        complex(scalar("voltage_V_peak")),
        complex(scalar("blocked_impedance_ohm")),
        complex(scalar("motional_impedance_ohm")),
        complex(scalar("total_impedance_ohm")),
        d["solid_displacement"],
        d["pressure_mixed"],
        d["pressure_base"],
        complex(pdir[i0]),
        float(20*np.log10(max(abs(pdir[i0])/np.sqrt(2),1e-300)/2e-5)),
        angles,
        pdir,
        d["directivity_relative_dB"],
        {},
        {},
    )



def cmd_visualize(args):
    model = _model(args)
    sol = _load_solution(args.solution_npz)
    out = Path(args.outdir or Path(args.solution_npz).parent / "plots_rebuilt")
    files = render_solution(model, sol, out, exterior_grid=not args.no_exterior_grid)
    print(json.dumps({"plots": [str(p) for p in files]}, indent=2))
    return 0

def cmd_compare(args):
    out = Path(args.outdir or ROOT / "runs/comparison")
    out.mkdir(parents=True, exist_ok=True)
    if args.sweep_csv:
        result = compare_impedance_sweep(args.sweep_csv, args.req5_raw, out)
    else:
        if not args.solution_npz:
            raise ValueError("compare requires --solution-npz or --sweep-csv")
        model = _model(args)
        sol = _load_solution(args.solution_npz)
        result = compare_single_solution(model, sol, args.req5_raw, out, args.req6_raw)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def cmd_export_info(args):
    print((ROOT / "docs/COMSOL_EXPORT_CONTRACT_CN.md").read_text(encoding="utf-8"))
    return 0


def cmd_eigen(args):
    cfg = load_config(ROOT, args.config)
    mesh = load_tagged_meshio(ROOT / cfg["geometry"]["mesh"])
    solid = build_p2_solid(mesh)
    result = solve_p2_eigenmodes(
        solid,
        n_modes=args.n_modes,
        sigma_Hz=args.sigma_Hz,
        tol=args.tol,
        maxiter=args.maxiter,
    )
    out = Path(args.outdir or ROOT / "runs" / "eigen_p2")
    summary = write_eigen_outputs(
        out,
        solid,
        result,
        comsol_shapes_csv=args.comsol_shapes,
        comsol_frequencies_csv=args.comsol_frequencies,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def cmd_fr10_full360(args):
    command = [
        sys.executable,
        str(ROOT / "fr10_full360_cyclic/cyclic_full360_solver.py"),
        "--freq",
        *(str(value) for value in args.freq),
    ]
    if args.scale is not None:
        command.extend(("--scale", str(args.scale)))
    if args.outdir:
        command.extend(("--out", str(Path(args.outdir))))
    if args.diagnostic_phases is not None:
        command.extend(("--diagnostic-phases", *(str(value) for value in args.diagnostic_phases)))
    return subprocess.call(command, cwd=ROOT / "fr10_full360_cyclic")


def cmd_fr10_animate(args):
    command = [
        sys.executable,
        str(ROOT / "fr10_full360_cyclic/animate_full360_results.py"),
        "--results-root",
        str(Path(args.results_root)),
        "--frequency",
        str(args.frequency),
        "--frames",
        str(args.frames),
        "--fps",
        str(args.fps),
    ]
    if args.surface_suite:
        command.append("--surface-suite")
    return subprocess.call(command, cwd=ROOT / "fr10_full360_cyclic")


def cmd_fr10_response(args):
    command = [
        sys.executable,
        str(ROOT / "fr10_full360_cyclic/frequency_response_1m.py"),
    ]
    if args.frequencies:
        command.extend(("--frequencies", *(str(value) for value in args.frequencies)))
    if args.output:
        command.extend(("--output", str(Path(args.output))))
    if args.reuse_summary:
        command.extend(("--reuse-summary", str(Path(args.reuse_summary))))
    if args.scale is not None:
        command.extend(("--scale", str(args.scale)))
    return subprocess.call(command, cwd=ROOT / "fr10_full360_cyclic")


def build_parser():
    p = argparse.ArgumentParser(description="Best COMSOL loudspeaker reproduction project")
    sp = p.add_subparsers(dest="command", required=True)

    s = sp.add_parser("self-test"); s.set_defaults(func=cmd_self_test)
    b = sp.add_parser("blocked", help="native voltage-constrained blocked-coil impedance")
    b.add_argument("--config", help="model JSON; defaults to configs/best_model.json")
    b.add_argument("--freqs", default="comsol_126")
    b.add_argument("--voltage", type=float, default=3.55)
    b.add_argument("--magnetostatic-vtu")
    b.add_argument("--raw-field", action="store_true", help="run the native MQS field solve instead of the embedded production surrogate")
    b.add_argument("--outdir")
    b.set_defaults(func=cmd_blocked)

    m = sp.add_parser("magnetics")
    m.add_argument("--outdir"); m.add_argument("--max-iter", type=int, default=55); m.add_argument("--tol", type=float, default=1e-5); m.add_argument("--relaxation", type=float, default=0.1); m.add_argument("--remanence-T", type=float, default=0.4); m.add_argument("--no-plots", action="store_true"); m.set_defaults(func=cmd_magnetics)

    def add_model_args(q):
        q.add_argument("--config", help="model JSON; defaults to configs/best_model.json")
        q.add_argument(
            "--magnetostatic-vtu",
            default=str(ROOT / "inputs/comsol_reference/magnetostatic_converged_55iter.vtu"),
            help="Lorentz-force magnetostatic VTU; blocked impedance uses blocked_coil.magnetostatic_vtu",
        )
        q.add_argument("--drive", choices=["current", "voltage"], default="current")
        q.add_argument("--current", type=float, default=1.0)
        q.add_argument("--voltage", type=float, default=1.0)
        q.add_argument("--blocked-impedance-csv")
        q.add_argument("--outdir")
        q.add_argument("--blas-threads", type=int, default=1, help="BLAS/OpenMP threads used by a single solve")
        q.add_argument("--without-nra", action="store_true", help="disable native thermoviscous Narrow Region Acoustics")

    s = sp.add_parser("solve"); add_model_args(s); s.add_argument("--freq", type=float, required=True); s.add_argument("--render", action="store_true"); s.add_argument("--no-exterior-grid", action="store_true"); s.set_defaults(func=cmd_solve)
    s = sp.add_parser("sweep"); add_model_args(s); s.add_argument("--freqs", default="diagnostic", help="preset, comma list, file, or log:start:stop:n"); s.add_argument("--save-each", action="store_true"); s.add_argument("--render", action="store_true"); s.add_argument("--jobs", type=int, default=1, help="frequency worker processes; 0 selects a conservative automatic value"); s.add_argument("--single-profile", action="store_true", help="disable the validated P1/P2 hybrid sweep and use only --config"); s.set_defaults(func=cmd_sweep)

    v = sp.add_parser("visualize"); add_model_args(v); v.add_argument("--solution-npz", required=True); v.add_argument("--no-exterior-grid", action="store_true"); v.set_defaults(func=cmd_visualize)

    c = sp.add_parser("compare"); add_model_args(c); c.add_argument("--req5-raw", required=True); c.add_argument("--req6-raw"); c.add_argument("--solution-npz"); c.add_argument("--sweep-csv"); c.set_defaults(func=cmd_compare)
    pe = sp.add_parser("eigen", help="P2 structural eigenfrequency solve and optional COMSOL MAC pairing")
    pe.add_argument("--config", help="model JSON; defaults to configs/best_model.json")
    pe.add_argument("--n-modes", type=int, default=40)
    pe.add_argument("--sigma-Hz", type=float, default=0.0)
    pe.add_argument("--tol", type=float, default=1e-10)
    pe.add_argument("--maxiter", type=int, default=8000)
    pe.add_argument("--comsol-shapes")
    pe.add_argument("--comsol-frequencies")
    pe.add_argument("--outdir")
    pe.set_defaults(func=cmd_eigen)
    f3 = sp.add_parser(
        "fr10-full360",
        help="FR10 four-sector cyclic/Bloch 3-D P2/local-ASB FEM",
    )
    f3.add_argument("--freq", type=float, nargs="+", default=[90.0, 500.0, 1000.0, 2000.0])
    f3.add_argument("--scale", type=float)
    f3.add_argument("--diagnostic-phases", type=int, nargs="+")
    f3.add_argument("--outdir")
    f3.set_defaults(func=cmd_fr10_full360)
    f3a = sp.add_parser("fr10-animate", help="animate FR10 full-360 complex FEM fields")
    f3a.add_argument("--results-root", required=True)
    f3a.add_argument("--frequency", type=float, default=2000.0)
    f3a.add_argument("--frames", type=int, default=24)
    f3a.add_argument("--fps", type=int, default=12)
    f3a.add_argument(
        "--surface-suite",
        action="store_true",
        help="render continuous moving diaphragm surfaces at 90 and 2000 Hz",
    )
    f3a.set_defaults(func=cmd_fr10_animate)
    f3r = sp.add_parser("fr10-response", help="FR10 full-360 1 m frequency response")
    f3r.add_argument("--frequencies", type=float, nargs="+")
    f3r.add_argument("--output")
    f3r.add_argument("--reuse-summary")
    f3r.add_argument("--scale", type=float)
    f3r.set_defaults(func=cmd_fr10_response)
    e = sp.add_parser("comsol-export-info"); e.set_defaults(func=cmd_export_info)
    return p


def main():
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
