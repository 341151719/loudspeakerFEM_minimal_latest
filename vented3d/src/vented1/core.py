from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

FREQUENCIES = np.array([
    20, 22.4, 25, 28, 31.5, 35.5, 40, 45, 50, 56, 63, 71,
    80, 90, 100, 112, 125, 140, 160, 180, 200, 224, 250, 280,
    315, 355, 400, 450, 500, 560, 630, 710, 800, 900, 1000,
    1120, 1250, 1400, 1600, 1800, 2000, 2240, 2500, 2800,
    3150, 3550
], dtype=float)
SELECTED = FREQUENCIES[np.array([1, 7, 15, 25, 31, 46]) - 1]
AIR_DOMAINS = {17, 22, 25, 28, 29}
SOLID_DOMAINS = set(range(1, 43)) - AIR_DOMAINS
SHELL_GROUPS = {
    "composite": {199, 200, 201, 202},
    "cloth": {173, 174, 175, 176, 177, 178, 179, 180, 181,
              182, 183, 184, 187, 188, 191, 192, 193, 194},
    "foam": {271, 272, 273, 274},
    "glass": {115, 116, 121, 122, 141, 142, 163, 164,
              171, 172, 195, 196},
    "basket": {129, 130, 155, 156, 189, 190, 239, 240, 269, 270}
}
SHELL_TAGS = set().union(*SHELL_GROUPS.values())
SYMMETRY = {
    33, 34, 35, 36, 37, 70, 71, 72, 73, 85, 86, 87, 94, 95,
    111, 112, 113, 114, 123, 124, 147, 148, 149, 150, 151, 152,
    167, 168, 185, 186, 198, 248, 249, 250, 251, 252, 253,
    254, 255, 256, 257, 258
}
INPUT_FILES = [
    "vertices.csv",
    *[f"elements_{s}.csv" for s in ("tet", "pyr", "hex", "tri",
                                   "quad", "edg", "vtx")],
    "selections.csv", "materials.csv", "parameters.csv",
    "boundary_adjacency.csv", "live_settings.csv",
    "int1_source.csv", "int2_source.csv"
]
APPROXIMATIONS = {
    "volumes": "conforming center conversion; straight P2 tetrahedra",
    "shell": "faceted P2 Reissner-Mindlin; mean-shear/residual stabilization",
    "creases": "shared global axial rotations; welded facets, not native MITC",
    "connection": "hard finite-thickness, topology-selected mortar",
    "exterior": "P0 planar collocation with CHIEF constrained least squares",
    "aperture": "P2-P0 work-adjoint mean-pressure mortar",
    "geometry": "fixed exported corner geometry; refinement does not recover CAD",
    "mass": "consistent; no mass or stiffness calibration",
    "motor": "absolute coil velocity feedback; moving reaction assembly"
}


def rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, complex):
        return [clean(value.real), clean(value.imag)]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(clean(value), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_json(path: Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(clean(data), f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez_compressed(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_csv(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    keys = list(dict.fromkeys(k for r in records for k in r))
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in records:
            w.writerow({k: clean(v) for k, v in r.items()})


def configuration(path: Path, root: str | None = None,
                  level: int | None = None) -> tuple[dict, Path]:
    path = Path(path).resolve()
    cfg = json.loads(path.read_text(encoding="utf-8"))
    r = Path(root) if root else Path(cfg["benchmark_root"])
    if not r.is_absolute():
        r = path.parent / r
    if level is not None:
        cfg["level"] = level
    if cfg["shell"]["fold_policy"] != "welded_facets":
        raise ValueError("Only the documented welded_facets shell policy exists")
    if not cfg["bem"]["mirror"]:
        raise ValueError("Benchmark production route requires half-model mirroring")
    backend = cfg["bem"].get("backend", "iterative")
    if backend not in ("iterative", "dense"):
        raise ValueError("Choose iterative or dense BEM to retain all CHIEF equations")
    if backend == "iterative" and cfg["solver"].get("core_backend") == "ilu_gmres":
        raise ValueError("Matrix-free LSMR requires pardiso or direct core inverses")
    if cfg["bem"]["chief_count"] < 1:
        raise ValueError("At least one CHIEF equation is required")
    return cfg, r.resolve()


def missing_inputs(root: Path) -> list[str]:
    return [name for name in INPUT_FILES if not (root / name).is_file()]


def provenance(root: Path, cfg: dict) -> dict:
    missing = missing_inputs(root)
    if missing:
        raise FileNotFoundError("Missing model inputs:\n" + "\n".join(missing))
    code = {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    inputs = {name: sha256(root / name) for name in INPUT_FILES}
    approximations = dict(APPROXIMATIONS)
    if cfg["connection"].get("method") == "nodal_spin":
        approximations["connection"] = "diagnostic nodal displacement/incident-solid spin tie"
    approximations["exterior"] = (
        "P0 planar collocation and CHIEF weighted least squares; "
        + cfg["bem"].get("backend", "iterative"))
    data = {
        "inputs": inputs, "code": code, "configuration": cfg,
        "approximations": approximations,
        "convention": "exp(+i omega t); peak phasors; SI; forward +X; symmetry Y=0",
        "python": sys.version, "platform": platform.platform()
    }
    data["compatibility_hash"] = digest({
        "inputs": inputs, "code": code, "configuration": cfg,
        "approximations": approximations
    })
    return data


def safe_output(root: Path, output: Path) -> Path:
    root, output = root.resolve(), output.resolve()
    if output == root or root in output.parents:
        raise ValueError("Output must not be the benchmark bundle or its descendant")
    output.mkdir(parents=True, exist_ok=True)
    return output


def guard(cfg: dict, byte_estimate: float, purpose: str) -> None:
    limit = cfg["memory_limit_GiB"] * 2**30
    if byte_estimate > limit:
        raise MemoryError(
            f"{purpose}: estimated {byte_estimate / 2**30:.3g} GiB exceeds "
            f"{cfg['memory_limit_GiB']:.3g} GiB. No automatic coarsening is performed."
        )


def peak_memory_bytes() -> int | None:
    try:
        import resource
        x = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(x if sys.platform == "darwin" else 1024 * x)
    except ImportError:
        return None


def frequency_key(f: float) -> str:
    return f"f_{f:.12g}".replace(".", "p")


def parse_frequencies(text: str) -> np.ndarray:
    if text == "benchmark":
        return FREQUENCIES.copy()
    if text == "selected":
        return SELECTED.copy()
    a = np.array([float(v) for v in text.split(",")])
    if len(a) == 0 or np.any(a <= 0) or len(np.unique(a)) != len(a):
        raise ValueError("Frequencies must be distinct positive numbers")
    return a


def normmax(a) -> float:
    if hasattr(a, "data") and not isinstance(a, np.ndarray):
        return float(np.max(np.abs(a.data), initial=0.0))
    return float(np.max(np.abs(a), initial=0.0))
