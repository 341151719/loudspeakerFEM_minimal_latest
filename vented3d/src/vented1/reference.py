"""Offline reference readers and comparisons. Never imported by production assembly."""
from __future__ import annotations

import gzip
import json
import struct
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from .core import (
    rows, atomic_json, atomic_npz, write_csv, FREQUENCIES,
    frequency_key, normmax
)
from .mesh import integer_list, restore_geometry, Locator
from .elements import tet_shape, tri_shape, shell_frame


def field_file(path: Path):
    with gzip.open(path, "rb") as f:
        raw = f.read()
    if len(raw) < 16:
        raise ValueError(f"{path}: truncated field header")
    version, components, solutions, samples = struct.unpack(">4i", raw[:16])
    if version != 1 or min(components, solutions, samples) < 1:
        raise ValueError(f"{path}: invalid field header")
    expected = 16 + 8*(3*samples+2*components*solutions*samples)
    if len(raw) != expected:
        raise ValueError(f"{path}: byte count {len(raw)}, expected {expected}")
    data = np.frombuffer(raw, dtype=">f8", offset=16).astype(float)
    coordinates = data[:3*samples].reshape(3, samples).T
    a = data[3*samples:].reshape(components, solutions, samples, 2)
    values = a[..., 0]+1j*a[..., 1]
    if not np.isfinite(coordinates).all() or not np.isfinite(values).all():
        raise ValueError(f"{path}: nonfinite coordinates or physical values")
    return coordinates, values


def native_vector(path: Path, dofs: Path, allowed_components: set[str] | None = None):
    with gzip.open(path, "rb") as f:
        raw = f.read()
    if len(raw) < 8:
        raise ValueError("Truncated native vector")
    version, length = struct.unpack(">2i", raw[:8])
    if version != 1 or length < 1 or len(raw) != 8+16*length:
        raise ValueError("Native vector header/payload mismatch")
    a = np.frombuffer(raw, dtype=">f8", offset=8).astype(float).reshape(length, 2)
    vector = a[:, 0]+1j*a[:, 1]
    if not np.isfinite(vector).all():
        raise ValueError("Nonfinite native vector")
    groups = {}
    for r in rows(dofs):
        name = r["component"]
        short = name.split(".")[-1]
        if allowed_components is not None and short not in allowed_components:
            continue
        index = int(r["solution_vector_index"])
        if not 0 <= index < length:
            raise ValueError(f"Native DOF {r['dof_index']}: vector index outside payload")
        g = groups.setdefault(name, {"coordinates_m": [], "values": [],
                                     "dof_index": [], "node_index": []})
        g["coordinates_m"].append([float(r[k]) for k in ("x_native", "y_native", "z_native")])
        g["values"].append(vector[index])
        g["dof_index"].append(int(r["dof_index"]))
        g["node_index"].append(int(r["node_index"]))
    return {name: {k: np.asarray(v) for k, v in group.items()}
            for name, group in groups.items()}


def expression_table(path: Path):
    result = {}
    for r in rows(path):
        key = (r["expression"], int(r["solution_index"]))
        if key in result:
            raise ValueError(f"Duplicate explicit reference key {key}")
        result[key] = complex(float(r["real"]), float(r["imag"]))
    return result


def scalar_error(z: complex, r: complex, floor: float):
    meaningful = abs(r) > floor
    phase_ok = meaningful and abs(z) > floor
    return {
        "prediction_real": z.real, "prediction_imag": z.imag,
        "reference_real": r.real, "reference_imag": r.imag,
        "absolute_error": abs(z-r),
        "complex_relative_error": abs(z-r)/abs(r) if meaningful else None,
        "amplitude_difference_dB": 20*np.log10(abs(z)/abs(r)) if phase_ok else None,
        "phase_difference_deg": np.angle(z/r, deg=True) if phase_ok else None,
        "small_reference": not meaningful, "phase_defined": phase_ok
    }


def norm_error(prediction, reference, weights=None):
    if weights is None:
        weights = np.ones(reference.shape[0])
    w = np.asarray(weights)
    while w.ndim < np.asarray(reference).ndim:
        w = w[..., None]
    numerator = np.sum(w*np.abs(prediction-reference)**2)
    denominator = np.sum(w*np.abs(reference)**2)
    return float(np.sqrt(numerator/denominator)) if denominator > 0 else None


def field_kind(expressions: list[str]):
    short = [s.split(".")[-1] for s in expressions]
    if "p_t" in short:
        return "acoustic"
    if all(k in short for k in ("u2", "v2", "w2")):
        return "shell"
    if all(k in short for k in ("u", "v", "w")):
        return "solid"
    raise ValueError(f"Unrecognized field physical expressions {expressions}")


def manifest_fields(root: Path):
    result = []
    for r in rows(root / "field_manifest.csv"):
        path = (root / r["file"]).resolve()
        if root.resolve() not in path.parents:
            raise ValueError("Field manifest path escapes reference root")
        coordinates, values = field_file(path)
        expressions = r["expressions"].split(";")
        if len(expressions) == values.shape[0]+3:
            expressions = expressions[3:]
        if len(expressions) != values.shape[0]:
            raise ValueError(f"{path}: expression/header component mismatch")
        indices = integer_list(r["solution_indices"])
        if len(indices) != values.shape[1] or len(set(indices)) != len(indices):
            raise ValueError(f"{path}: solution index mismatch")
        modal = indices == list(range(1, 21))
        result.append({
            "kind": field_kind(expressions), "modal": modal,
            "entity": int(r["entity"]), "coordinates": coordinates,
            "values": values * np.asarray([0.001 if e in ("u", "v", "w", "u2", "v2", "w2") else 1.0 for e in expressions])[:, None, None], "expressions": expressions,
            "indices": indices, "file": str(path)
        })
    return result


def evaluate(m, s, U, pressure, kind: str, entity: int,
             points: np.ndarray, tolerance: float):
    values, distances = [], []
    if kind in ("solid", "acoustic"):
        cells = np.flatnonzero(m.domain == entity)
        if len(cells) == 0:
            raise ValueError(f"No Python {kind} domain {entity}")
        locator = Locator(m, cells)
        for point in points:
            ci, L, distance = locator.locate(point, tolerance)
            N = tet_shape(L)
            if kind == "solid":
                v = U[:s.offset].reshape(-1, 3)[s.scell[ci]]
            else:
                v = pressure[s.pcell[ci]][:, None]
            values.append(N@v)
            distances.append(distance)
    else:
        faces = np.flatnonzero(m.tag == entity)
        if len(faces) == 0:
            raise ValueError(f"No Python shell entity {entity}")
        centers = m.x[m.btri[faces]].mean(axis=1)
        tree = cKDTree(centers)
        radius = np.linalg.norm(
            m.x[m.btri[faces]]-centers[:, None], axis=2).max()
        uh = U[s.offset:].reshape(-1, 6)
        for point in points:
            best = None
            for local in tree.query_ball_point(point, radius+tolerance):
                bi = int(faces[local])
                tri = m.x[m.btri[bi]]
                _, t1, t2, n, _ = shell_frame(tri)
                xy = (tri-tri[0])@np.column_stack((t1, t2))
                xq = (point-tri[0])@np.column_stack((t1, t2))
                uv = np.linalg.solve(np.column_stack((xy[1], xy[2])), xq)
                L = np.r_[1-uv.sum(), uv]
                distance = abs((point-tri[0])@n)
                if L.min() >= -1e-9 and distance <= tolerance:
                    if best is None or distance < best[0]:
                        nodal = uh[s.hcell[bi]]
                        v = tri_shape(L)@nodal
                        best = (distance, np.r_[v[:3], np.cross(v[3:], n)])
            if best is None:
                raise ValueError(f"Shell {entity}: no bounded projection for {point.tolist()}")
            distances.append(best[0])
            values.append(best[1])
    return np.asarray(values), np.asarray(distances)


def mls_values(points, values, queries, maximum_distance):
    """Entity-restricted affine reconstruction for comparison only, never solving."""
    tree = cKDTree(points)
    k = min(24, len(points))
    d, ids = tree.query(queries, k=k)
    if k == 1:
        d, ids = d[:, None], ids[:, None]
    result, ranks = [], []
    for q, distance, index in zip(queries, d, ids):
        if distance[0] > maximum_distance:
            raise ValueError(f"Reference interpolation distance {distance[0]:g} m exceeds limit")
        if distance[0] < 1e-11:
            result.append(values[index[0]])
            ranks.append(0)
            continue
        scale = max(distance[-1], 1e-12)
        X = np.column_stack((np.ones(k), (points[index]-q)/scale))
        weight = 1/np.maximum(distance/scale, .05)
        Y = values[index].reshape(k, -1)
        coefficient, _, rank, _ = np.linalg.lstsq(
            weight[:, None]*X, weight[:, None]*Y, rcond=1e-10)
        result.append(coefficient[0].reshape(values.shape[1:]))
        ranks.append(int(rank))
    return np.asarray(result), {
        "maximum_nearest_distance_m": float(np.max(d[:, 0])),
        "affine_ranks": sorted(set(ranks)),
        "method": "entity-restricted local affine sampled reconstruction"
    }


def modal_compare(output: Path, root: Path, cfg: dict, fields: list, destination: Path):
    if not (output / "modes.npz").exists():
        return {"available": False}
    m, s = restore_geometry(output / "geometry.npz")
    z = np.load(output / "modes.npz", allow_pickle=False)
    phi = z["U"].copy()
    M = sparse.load_npz(output / "full_M.npz")
    reference = np.zeros((s.nm, 20), complex)
    weights = np.zeros(s.nm)
    reports = []
    director = cfg["compare"]["director"]
    if director not in ("unresolved", "director_perturbation", "axial_rotation"):
        raise ValueError("Unknown reference director interpretation")
    for field in fields:
        if not field["modal"] or field["kind"] == "acoustic":
            continue
        kind, entity = field["kind"], field["entity"]
        # sample, mode, component
        samples = field["values"].transpose(2, 1, 0)
        if kind == "solid":
            cells = np.flatnonzero(m.domain == entity)
            nodes = np.unique(s.scell[cells])
            nodes = nodes[nodes >= 0]
            query = m.x[s.snode[nodes]]
            value, report = mls_values(
                field["coordinates"], samples[:, :, :3], query,
                cfg["compare"]["mls_max_distance_m"])
            for j, node in enumerate(nodes):
                reference[3*node:3*node+3] += value[j].T
                weights[3*node:3*node+3] += 1
        else:
            faces = np.flatnonzero(m.tag == entity)
            nodes = np.unique(s.hcell[faces])
            nodes = nodes[nodes >= 0]
            query = m.x[s.hnode[nodes]]
            value, report = mls_values(
                field["coordinates"], samples, query,
                cfg["compare"]["mls_max_distance_m"])
            normals = {}
            for bi in faces:
                A, _, _, normal, _ = shell_frame(m.x[m.btri[bi]])
                for node in s.hcell[bi]:
                    normals[node] = normals.get(node, np.zeros(3))+A*normal
            sign = float(cfg["compare"]["director_normal_sign"].get(str(entity), 1))
            for j, node in enumerate(nodes):
                base = s.offset+6*node
                reference[base:base+3] += value[j, :, :3].T
                weights[base:base+3] += 1
                if director != "unresolved":
                    if value.shape[2] != 6:
                        raise ValueError("Full-inertia MAC needs all six shell components")
                    n = sign*normals[node]/np.linalg.norm(normals[node])
                    if director == "director_perturbation":
                        rotation = np.cross(n[None], value[j, :, 3:])
                    else:
                        rotation = value[j, :, 3:]
                    reference[base+3:base+6] += rotation.T
                    weights[base+3:base+6] += 1
        reports.append({"entity": entity, "kind": kind, **report})
    active = weights > 0
    reference[active] /= weights[active, None]
    displacement_dofs = np.r_[
        np.arange(s.offset),
        (s.offset+6*np.arange(len(s.hnode))[:, None]+np.arange(3)).ravel()
    ]
    if np.any(weights[displacement_dofs] == 0):
        raise ValueError("Modal reference reconstruction leaves displacement DOFs uncovered")
    if director == "unresolved":
        mask = np.zeros(s.nm)
        mask[displacement_dofs] = 1
        metric = sparse.diags(mask)@M@sparse.diags(mask)
        metric_name = "translational_mass_only; shell rotary inertia excluded"
    else:
        metric, metric_name = M, "full_Python_consistent_mass_with_explicit_director_adapter"
    a = phi.conj().T@(metric@reference)
    pnorm = np.real(np.sum(phi.conj()*(metric@phi), axis=0))
    rnorm = np.real(np.sum(reference.conj()*(metric@reference), axis=0))
    if np.any(pnorm <= 1e-25) or np.any(rnorm <= 1e-25):
        raise ValueError("Near-zero modal mass norm in comparison")
    MAC = abs(a)**2/(pnorm[:, None]*rnorm[None])
    pi, ri = linear_sum_assignment(-MAC)
    eig = expression_table(root / "eigen_reference.csv")
    records = []
    for i, j in zip(pi, ri):
        rf = eig[("freq", j+1)]
        pf = z["frequency_Hz"][i]
        alpha = a[i, j]/pnorm[i]
        difference = alpha*phi[:, i]-reference[:, j]
        error = np.sqrt(max(np.real(np.vdot(difference, metric@difference)), 0)/rnorm[j])
        records.append({
            "python_mode": int(i+1), "reference_mode": int(j+1),
            "MAC": float(MAC[i, j]), "weak_match": bool(MAC[i, j] < cfg["compare"]["mac_min"]),
            "prediction_real_Hz": pf.real, "prediction_imag_Hz": pf.imag,
            "reference_real_Hz": rf.real, "reference_imag_Hz": rf.imag,
            "real_frequency_difference_Hz": (pf-rf).real,
            "imag_frequency_difference_Hz": (pf-rf).imag,
            "phase_aligned_mass_error": float(error),
            "metric": metric_name
        })
    write_csv(destination / "modal_pairs.csv", records)
    atomic_npz(destination / "modal_comparison.npz", MAC=MAC, reference_U=reference,
               prediction_U=phi)
    return {
        "available": True, "metric": metric_name,
        "full_inertia_comparison": director != "unresolved",
        "sampled_interpolation_is_an_additional_error": True,
        "native_vectors_not_assumed_to_be_geometric_nodes": True,
        "interpolation_reports": reports, "pairs": records
    }


def compare(output: Path, root: Path, cfg: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = output / "comparison"
    destination.mkdir(exist_ok=True)
    required = ["frequency_reference.csv", "directivity_reference.csv",
                "eigen_reference.csv", "field_manifest.csv"]
    missing = [name for name in required if not (root/name).exists()]
    if missing:
        atomic_json(destination / "summary.json", {
            "reference_available": False, "missing": missing, "benchmark_acceptance": False
        })
        return
    table = expression_table(root / "frequency_reference.csv")
    direct = {}
    for r in rows(root / "directivity_reference.csv"):
        key = (r["plane"], float(r["theta_deg"]), int(r["solution_index"]))
        if key in direct:
            raise ValueError(f"Duplicate directivity reference key {key}")
        direct[key] = complex(float(r["pressure_real_Pa_peak"]),
                              float(r["pressure_imag_Pa_peak"]))
    if len(direct) != 3404:
        raise ValueError(f"Expected 3404 directivity samples, found {len(direct)}")
    m, s = restore_geometry(output / "geometry.npz")
    fields = manifest_fields(root)
    scalar_records, direct_records, field_records = [], [], []
    solved = {}
    for index, f in enumerate(FREQUENCIES, 1):
        if ("freq", index) not in table or abs(table[("freq", index)]-f) > 1e-8:
            raise ValueError(f"Frequency reference explicit index {index} is inconsistent")
        path = output / f"{frequency_key(f)}.npz"
        if not path.exists():
            continue
        z = np.load(path, allow_pickle=False)
        solved[index] = z
        mappings = {
            "v0": (complex(z["coil_velocity"]), cfg["compare"]["velocity_floor_m_s"]),
            "Zb": (complex(z["blocked_impedance"]), 1e-12),
            "Fe": (complex(z["force"]), 1e-12)
        }
        for expression, (prediction, floor) in mappings.items():
            if (expression, index) not in table:
                raise ValueError(f"Missing reference expression/index {(expression, index)}")
            scalar_records.append({
                "frequency_Hz": f, "solution_index": index, "observable": expression,
                **scalar_error(prediction, table[(expression, index)], floor)
            })
        for plane_index, plane in enumerate(("XY", "XZ")):
            predicted = z["directivity"][plane_index]
            reference = np.array([direct[(plane, float(a), index)] for a in z["theta_deg"]])
            for angle, prediction, ref in zip(z["theta_deg"], predicted, reference):
                direct_records.append({
                    "frequency_Hz": f, "solution_index": index, "plane": plane,
                    "theta_deg": angle,
                    **scalar_error(prediction, ref, cfg["compare"]["pressure_floor_Pa"])
                })
            fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
            for values, label in ((reference, "COMSOL offline"), (predicted, "independent")):
                spl = 20*np.log10(np.maximum(abs(values), 1e-300)/(np.sqrt(2)*20e-6))
                axes[0].plot(z["theta_deg"], spl, label=label)
                axes[1].plot(z["theta_deg"], np.angle(values, deg=True), label=label)
            axes[0].set_ylabel("SPL dB re 20 µPa RMS")
            axes[1].set_ylabel("Peak-phasor phase, deg")
            axes[1].set_xlabel(f"{plane}: angle from +X, deg; radius 1 m")
            axes[0].legend()
            fig.suptitle(f"{f:g} Hz; exp(+iωt); absolute, unnormalized pressure")
            fig.tight_layout()
            fig.savefig(destination / f"{frequency_key(f)}_{plane}.png", dpi=140)
            plt.close(fig)
    for field in fields:
        if field["modal"]:
            continue
        for local, index in enumerate(field["indices"]):
            if index not in solved:
                continue
            z = solved[index]
            try:
                prediction, distances = evaluate(
                    m, s, z["U"], z["p_interior"], field["kind"], field["entity"],
                    field["coordinates"], cfg["compare"]["projection_m"])
                reference = field["values"][:, local].T
                # Director components are not compared under an unresolved convention.
                components = 1 if field["kind"] == "acoustic" else 3
                p, r = prediction[:, :components], reference[:, :components]
                field_records.append({
                    "kind": field["kind"], "entity": field["entity"],
                    "solution_index": index, "frequency_Hz": FREQUENCIES[index-1],
                    "sample_count": len(r),
                    "sample_weighting": "uniform within this entity, not volume L2",
                    "normwise_complex_error": norm_error(p, r),
                    "maximum_absolute_error": normmax(p-r),
                    "maximum_projection_m": distances.max(initial=0),
                    "status": "compared"
                })
            except ValueError as exc:
                field_records.append({
                    "kind": field["kind"], "entity": field["entity"],
                    "solution_index": index, "status": "interpolation_failed",
                    "diagnostic": str(exc)
                })
    write_csv(destination / "scalar_errors.csv", scalar_records)
    write_csv(destination / "directivity_errors.csv", direct_records)
    write_csv(destination / "field_errors.csv", field_records)
    modal = modal_compare(output, root, cfg, fields, destination)
    fig, ax = plt.subplots()
    for observable in ("v0", "Zb", "Fe"):
        rr = [r for r in scalar_records if r["observable"] == observable
              and r["complex_relative_error"] is not None]
        ax.loglog([r["frequency_Hz"] for r in rr],
                  [max(r["complex_relative_error"], 1e-16) for r in rr], label=observable)
    ax.set_xlabel("Frequency, Hz")
    ax.set_ylabel("Complex relative error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination / "frequency_errors.png", dpi=140)
    plt.close(fig)
    angular_summary = []
    for index, z in solved.items():
        for j, plane in enumerate(("XY", "XZ")):
            ref = np.array([direct[(plane, float(a), index)] for a in z["theta_deg"]])
            weights = np.ones(len(ref))
            weights[[0, -1]] = .5
            angular_summary.append({
                "solution_index": index, "plane": plane,
                "normwise_complex_error": norm_error(z["directivity"][j], ref, weights),
                "weights": "uniform-angle trapezoid on the one-metre arc"
            })
    acceptance = cfg["compare"]["scalar_relative_acceptance"]
    compared_errors = [r["normwise_complex_error"] for r in angular_summary]
    summary = {
        "reference_available": True, "frequencies_compared": len(solved),
        "frequency_sweep_complete": len(solved) == 46,
        "angular_errors": angular_summary,
        "angular_tolerance": acceptance,
        "angular_tolerance_satisfied": bool(compared_errors) and max(compared_errors) <= acceptance,
        "field_interpolation_failures": sum(r["status"] != "compared" for r in field_records),
        "modal": modal,
        "benchmark_acceptance": False,
        "acceptance_note": "Not automatically granted: requires field/modal tolerances, "
                           "director interpretation, executed verification and refinement evidence",
        "frequency_reference_unmapped_expressions": sorted(
            {e for e, _ in table}-{ "freq", "Rb(freq)", "Lb(freq)", "v0", "Zb", "Fe" }),
        "one_metre_axis_reference_source": "directivity XY 0 deg and XZ 90 deg"
    }
    atomic_json(destination / "summary.json", summary)


def convergence_summary(root: Path, levels: list[int], frequencies: np.ndarray):
    records = []
    for previous, current in zip(levels[:-1], levels[1:]):
        for f in frequencies:
            a = root/f"L{previous}"/f"{frequency_key(f)}.npz"
            b = root/f"L{current}"/f"{frequency_key(f)}.npz"
            if not a.exists() or not b.exists():
                records.append({"from": previous, "to": current, "frequency_Hz": f,
                                "status": "missing_checkpoint"})
                continue
            x, y = np.load(a), np.load(b)
            for name in ("directivity", "current", "coil_velocity"):
                reference = np.atleast_1d(y[name])
                prediction = np.atleast_1d(x[name])
                records.append({
                    "from": previous, "to": current, "frequency_Hz": f,
                    "observable": name, "status": "compared",
                    "relative_change_to_finer": norm_error(prediction, reference),
                    "absolute_change_inf": normmax(prediction-reference)
                })
    write_csv(root/"convergence.csv", records)
    atomic_json(root/"convergence.json", {
        "levels": levels, "records": records,
        "mesh_convergence_acceptance": None,
        "geometry": "fixed faceted geometry; uniform field/panel refinement only"
    })
