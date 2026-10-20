from __future__ import annotations

import json
import time
import threading
from pathlib import Path

import numpy as np
from scipy import sparse, linalg
from scipy.sparse.linalg import splu, spilu, gmres, eigs, LinearOperator, lsqr, lsmr

from .core import (
    atomic_json, atomic_npz, frequency_key, peak_memory_bytes, normmax,
    guard, AIR_DOMAINS
)
from .elements import air, equivalent, Motor
from .mesh import save_geometry
from .model import Model
from .bem import coupled_operators, operators


class ScaledFactor:
    def __init__(self, A: sparse.spmatrix, columns: np.ndarray):
        self.A = A.tocsr()
        self.columns = np.asarray(columns)
        scaled = self.A @ sparse.diags(self.columns)
        magnitude = np.asarray(abs(scaled).max(axis=1).toarray()).ravel()
        if np.any(magnitude == 0):
            raise ValueError("Zero row in physical core")
        self.rows = 1/magnitude
        self.matrix = sparse.diags(self.rows) @ scaled
        self.lu = splu(self.matrix.tocsc())

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        rhs = np.asarray(rhs)
        if rhs.ndim == 1:
            return self.columns*self.lu.solve(self.rows*rhs)
        return self.columns[:, None]*self.lu.solve(self.rows[:, None]*rhs)

    def solve_adj(self, rhs: np.ndarray) -> np.ndarray:
        rhs = np.asarray(rhs)
        if rhs.ndim == 1:
            return self.rows*self.lu.solve(self.columns*rhs, trans="H")
        return self.rows[:, None]*self.lu.solve(self.columns[:, None]*rhs, trans="H")


class PardisoFactor:
    """Scaled complex nonsymmetric core factorization using oneMKL PARDISO."""
    def __init__(self, A: sparse.spmatrix, columns: np.ndarray):
        try:
            from .pardiso_native import PardisoSolver
        except ImportError as exc:
            raise ImportError("Install vented1[mkl] for the pardiso core backend") from exc
        self.columns = np.asarray(columns)
        scaled = A @ sparse.diags(self.columns)
        magnitude = np.asarray(abs(scaled).max(axis=1).toarray()).ravel()
        if np.any(magnitude == 0):
            raise ValueError("Zero row in physical core")
        self.rows = 1/magnitude
        matrix = (sparse.diags(self.rows) @ scaled).tocsr()
        matrix.sum_duplicates()
        matrix.sort_indices()
        self.solver = PardisoSolver(matrix)
        self._lock = threading.RLock()
        self.lu = None

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        return self._solve(rhs, adjoint=False)

    def solve_adj(self, rhs: np.ndarray) -> np.ndarray:
        return self._solve(rhs, adjoint=True)

    def _solve(self, rhs: np.ndarray, adjoint: bool) -> np.ndarray:
        """Reuse the factorization for A or A^H, restoring iparm after errors.

        A_s=R A D: A^-1=D A_s^-1 R and A^-H=R A_s^-H D.
        oneMKL's zero-based iparm[11]=1 selects the Hermitian transpose.
        One solver handle is serialized because its solve mode is mutable.
        """
        rhs = np.asarray(rhs, dtype=np.complex128)
        if rhs.ndim not in (1, 2) or rhs.shape[0] != len(self.columns):
            raise ValueError("PARDISO RHS must have shape (n,) or (n, nrhs)")
        left, right = ((self.rows, self.columns) if adjoint else
                       (self.columns, self.rows))
        b = right*rhs if rhs.ndim == 1 else np.asfortranarray(right[:, None]*rhs)
        with self._lock:
            previous = self.solver.get_iparm_value(11)
            try:
                self.solver.set_iparm(11, 1 if adjoint else 0)
                x = self.solver.solve(b)
            finally:
                self.solver.set_iparm(11, previous)
        return left*x if rhs.ndim == 1 else left[:, None]*x


class IterativeFactor:
    """Memory-bounded alternative for the large coupled physical core."""
    def __init__(self, A: sparse.spmatrix, columns: np.ndarray, cfg: dict):
        self.A = A.tocsr()
        self.columns = np.asarray(columns)
        scaled = (self.A @ sparse.diags(self.columns)).tocsc()
        magnitude = np.asarray(abs(scaled).max(axis=1).toarray()).ravel()
        if np.any(magnitude == 0):
            raise ValueError("Zero row in physical core")
        self.rows = 1/magnitude
        self.matrix = (sparse.diags(self.rows) @ scaled).tocsc()
        self.preconditioner_shift = cfg.get("ilu_shift", 0.01)
        preconditioner_matrix = self.matrix + self.preconditioner_shift*sparse.eye(
            self.matrix.shape[0], dtype=complex, format="csc")
        self.ilu = spilu(preconditioner_matrix,
                         drop_tol=cfg.get("ilu_drop_tol", 1e-4),
                         fill_factor=cfg.get("ilu_fill_factor", 8.))
        self.lu = self.ilu
        self.tol = cfg.get("iterative_core_tol", 1e-9)
        self.maxiter = cfg.get("iterative_core_maxiter", 300)

    def _one(self, rhs: np.ndarray, adjoint: bool) -> np.ndarray:
        if adjoint:
            right = self.columns*rhs
            mat = LinearOperator(self.matrix.shape,
                                 matvec=lambda x: self.matrix.conj().T@x,
                                 dtype=complex)
            pre = LinearOperator(self.matrix.shape,
                                 matvec=lambda x: self.ilu.solve(x, trans="H"),
                                 dtype=complex)
        else:
            right = self.rows*rhs
            mat = self.matrix
            pre = LinearOperator(self.matrix.shape,
                                 matvec=self.ilu.solve, dtype=complex)
        y, info = gmres(mat, right, M=pre, rtol=self.tol, atol=0.,
                        restart=50, maxiter=self.maxiter)
        if info:
            raise ArithmeticError(f"Physical core GMRES failed: info={info}")
        return (self.rows if adjoint else self.columns)*y

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        rhs = np.asarray(rhs)
        return (self._one(rhs, False) if rhs.ndim == 1 else
                np.column_stack([self._one(rhs[:, j], False) for j in range(rhs.shape[1])]))

    def solve_adj(self, rhs: np.ndarray) -> np.ndarray:
        rhs = np.asarray(rhs)
        return (self._one(rhs, True) if rhs.ndim == 1 else
                np.column_stack([self._one(rhs[:, j], True) for j in range(rhs.shape[1])]))


def acoustic_matrix(model: Model, f: float, cfg: dict):
    w = 2*np.pi*f
    a0 = air(cfg["temperature_K"])
    A = sparse.csr_matrix((model.space.np, model.space.np), dtype=complex)
    slit = {}
    for d in AIR_DOMAINS:
        if d in (28, 29):
            properties = equivalent(f, .0002 if d == 28 else .0004, a0)
            aa, bb = properties["a"], properties["b"]
            slit[d] = properties
        else:
            aa, bb = 1/a0["rho"], 1/(a0["rho"]*a0["c"]**2)
        A = A + aa*model.ak[d] - w*w*bb*model.am[d]
    return A.tocsr(), a0, slit


def initialize_output(model: Model, output: Path, provenance: dict):
    manifest = output / "run_manifest.json"
    if manifest.exists():
        old = json.loads(manifest.read_text(encoding="utf-8"))
        if old["compatibility_hash"] != provenance["compatibility_hash"]:
            raise ValueError("Output directory belongs to an incompatible model/configuration/code")
    else:
        atomic_json(manifest, provenance)
    save_geometry(output / "geometry.npz", model.mesh, model.space)
    for name in ("K", "loss", "C", "M"):
        sparse.save_npz(output / f"full_{name}.npz", model.full[name])
    sparse.save_npz(output / "constraints.npz", model.constraints)
    sparse.save_npz(output / "transform.npz", model.T)
    atomic_json(output / "assembly_audit.json", model.report)


def _physical_residual(A, x, b):
    r = A@x-b
    denom = np.asarray(abs(A)@np.abs(x)).ravel()+np.abs(b)
    return {
        "absolute_inf": normmax(r),
        "componentwise_backward": float(np.max(
            np.abs(r)/np.maximum(denom, 1e-300), initial=0.)),
        "normwise_backward": normmax(r)/max(
            normmax(denom), 1e-300)
    }


def _resultant(model: Model, force: np.ndarray) -> dict:
    s, m = model.space, model.mesh
    fs = force[:s.offset].reshape(-1, 3)
    fh = force[s.offset:].reshape(-1, 6)
    total = fs.sum(axis=0)+fh[:, :3].sum(axis=0)
    moment = np.cross(m.x[s.snode], fs).sum(axis=0)
    shell_moment=(np.cross(s.director_normals,fh[:,3:]) if hasattr(s,'director_normals') else fh[:,3:])
    moment += np.cross(m.x[s.hnode], fh[:, :3]).sum(axis=0)+shell_moment.sum(axis=0)
    return {"force_N": total, "moment_Nm": moment}


def boundary_operator(factor, boundary_rhs, Pe, Qe, Ewall, Eap,
                      Lp, Lq, weights, xscale, progress=None):
    """Weighted FEM-condensed operator including every boundary and CHIEF row.

    Conjugate vectors around dense transposes rather than copying conjugated
    dense BEM matrices on every adjoint application.
    """
    if not callable(getattr(factor, "solve_adj", None)):
        raise TypeError("Least-squares BEM requires a core solve_adj interface")
    weights, xscale = np.asarray(weights), np.asarray(xscale)
    if (not np.isrealobj(weights) or not np.isrealobj(xscale)
            or np.any(weights <= 0) or np.any(xscale <= 0)
            or not np.isfinite(weights).all() or not np.isfinite(xscale).all()):
        raise ValueError("BEM row and column scales must be finite, positive and real")
    PeH, QeH = Pe.conj().T, Qe.conj().T
    BH, EwH, EaH = boundary_rhs.conj().T, Ewall.conj().T, Eap.conj().T
    calls = {"forward": 0, "adjoint": 0}

    def update(name):
        calls[name] += 1
        if progress and (sum(calls.values()) == 1 or sum(calls.values()) % 20 == 0):
            progress(dict(calls))

    def matvec(z):
        x = xscale*np.asarray(z).reshape(-1)
        response = factor.solve(boundary_rhs@x)
        result = weights*(Lp@(Pe@response+Ewall@x)
                          + Lq@(Qe@response+Eap@x))
        update("forward")
        return result

    def rmatvec(z):
        v = weights*np.asarray(z).reshape(-1)
        lp = np.conj(Lp.T@np.conj(v))
        lq = np.conj(Lq.T@np.conj(v))
        adj = factor.solve_adj(PeH@lp+QeH@lq)
        result = xscale*(EwH@lp+EaH@lq+BH@adj)
        update("adjoint")
        return result

    op = LinearOperator((Lp.shape[0], len(xscale)), matvec=matvec,
                        rmatvec=rmatvec, dtype=complex)
    return op, calls


def solve_frequency(model: Model, motor: Motor, f: float, voltage: complex,
                    cfg: dict, output: Path, signature: str, resume=False):
    if f <= 0:
        raise ValueError("Harmonic frequency must be positive")
    bem_backend = cfg["bem"].get("backend", "iterative")
    if bem_backend not in ("iterative", "dense"):
        raise ValueError("BEM backend must be iterative or dense: both retain CHIEF rows. "
                         "The former collocation_gmres route omitted CHIEF stabilization.")
    if bem_backend == "iterative" and cfg["solver"].get("core_backend") == "ilu_gmres":
        raise ValueError("LSMR requires consistent forward/adjoint core inverses; "
                         "select pardiso or direct, not tolerance-stopped inner GMRES")
    key = frequency_key(f)
    status_path, array_path = output / f"{key}.json", output / f"{key}.npz"
    if status_path.exists():
        old = json.loads(status_path.read_text(encoding="utf-8"))
        matches = (old["compatibility_hash"] == signature
                   and old["voltage_peak_V"] == [voltage.real, voltage.imag])
        if resume and matches and old.get("solver_success") and array_path.exists():
            return old
        if old.get("solver_success"):
            raise FileExistsError(f"Refusing to replace successful checkpoint {status_path}")
    start = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    phase_times = {}

    def stage(name, **details):
        elapsed = time.perf_counter()-start
        phase_times[name] = elapsed
        event = {"compatibility_hash": signature, "frequency_Hz": f,
                 "stage": name, "elapsed_s": elapsed,
                 "peak_process_memory_bytes": peak_memory_bytes(), **details}
        atomic_json(output / f"{key}_progress.json", event)
        print(f"[{f:g} Hz] {name} ({elapsed:.2f} s)", flush=True)

    stage("core_assembly_started")
    w = 2*np.pi*f
    r = model.reduced
    nr, npf, nb = model.T.shape[1], model.space.np, len(model.ext)
    nw, na = len(model.wall), len(model.aperture)
    H = r["K"]+1j*r["loss"]+1j*w*r["C"]-w*w*r["M"]
    A, air0, slit = acoustic_matrix(model, f, cfg)
    Rb, Lb, Zb = motor.values(f)
    S = sparse.bmat([
        [H, -r["Bi"], sparse.csr_matrix(-r["f"][:, None])],
        [-w*w*r["Bi"].T, A, sparse.csr_matrix((npf, 1))],
        [sparse.csr_matrix((1j*w*motor.BL*r["gc"])[None]),
         sparse.csr_matrix((1, npf)), sparse.csr_matrix([[Zb]])]
    ], format="csr")
    ncore = S.shape[0]
    col = np.full(nr, cfg["solver"]["displacement_scale_m"])
    shell_rotation = ((model.free >= model.space.offset)
                      & ((model.free-model.space.offset) % 6 >= 3))
    col[shell_rotation] = cfg["solver"]["rotation_scale"]
    columns = np.r_[col, np.full(npf, cfg["solver"]["pressure_scale_Pa"]),
                    cfg["solver"]["current_scale_A"]]
    core_backend = cfg["solver"].get("core_backend", "direct")
    stage("core_factorization_started", backend=core_backend,
          unknowns=ncore, nnz=S.nnz)
    if core_backend == "pardiso":
        factor = PardisoFactor(S, columns)
    elif core_backend == "ilu_gmres":
        factor = IterativeFactor(S, columns, cfg["solver"])
    elif core_backend == "direct":
        factor = ScaledFactor(S, columns)
    else:
        raise ValueError(f"Unknown physical core backend {core_backend}")
    try:
        b = np.zeros(ncore, complex)
        b[-1] = voltage
        y0 = factor.solve(b)
        stage("core_solve_finished")
        wall, ap = model.wall, model.aperture
        Ewall = sparse.coo_matrix(
            (np.ones(nw), (wall, np.arange(nw))), shape=(nb, nb)).tocsr()
        Eap = sparse.coo_matrix(
            (np.ones(na), (ap, nw+np.arange(na))), shape=(nb, nb)).tocsr()
        pressure_trace = sparse.diags(1/model.areas) @ model.Cp.T
        Pe = sparse.hstack((sparse.csr_matrix((nb, nr)), pressure_trace,
                            sparse.csr_matrix((nb, 1))), format="csr")
        Qe = sparse.hstack((
            air0["rho"]*w*w*sparse.diags(1/model.areas)@r["Be"].T,
            sparse.csr_matrix((nb, npf+1))), format="csr")
        boundary_rhs = sparse.bmat([
            [-r["Be"][:, wall], sparse.csr_matrix((nr, na))],
            [sparse.csr_matrix((npf, nw)), model.Cp[:, ap]/air0["rho"]],
            [sparse.csr_matrix((1, nw)), sparse.csr_matrix((1, na))]
        ], format="csr")
        rows_count = nb+cfg["bem"]["chief_count"]
        guard(cfg, 16*(10*nb*nb+4*rows_count*nb+ncore*cfg["bem"]["schur_chunk"]),
              "BEM Schur/least-squares live allocation estimate")
        stage("bem_assembly_started")
        Lp, Lq, weights, chiefs = coupled_operators(
            model, w/air0["c"], cfg["bem"],
            progress=lambda info: stage("bem_assembly_progress", **info))
        rows_count = Lp.shape[0]
        stage("bem_assembly_finished", boundary_rows=nb, chief_rows=len(chiefs))
        rhs = -(Lp@(Pe@y0)+Lq@(Qe@y0))
        xscale = np.r_[
            np.full(nw, cfg["solver"]["pressure_scale_Pa"]),
            np.full(na, cfg["solver"]["pressure_scale_Pa"]/cfg["solver"]["length_scale_m"])
        ]
        if cfg["bem"].get("column_equilibration") == "geometric":
            # Dimensionally balance pressure and normal-derivative columns.
            # These positive variable scales do not change the physical equations.
            xscale[:nw] /= weights[wall]
            xscale[nw:] = (cfg["solver"]["pressure_scale_Pa"]
                           / (np.sqrt(model.areas[ap])*weights[ap]))
        scaled_rhs = weights*rhs
        stage("schur_started", backend=bem_backend, rows=rows_count, columns=nb)
        operator_calls = None
        if bem_backend == "iterative":
            op, operator_calls = boundary_operator(
                factor, boundary_rhs, Pe, Qe, Ewall, Eap, Lp, Lq, weights, xscale,
                progress=lambda counts: stage("schur_progress", operator_calls=counts))
            iterative = lsmr(
                op, scaled_rhs, atol=cfg["bem"].get("iterative_tol", 1e-8),
                btol=cfg["bem"].get("iterative_tol", 1e-8),
                conlim=cfg["bem"].get("iterative_conlim", 1e8),
                maxiter=cfg["bem"].get("iterative_maxiter", 500),
            show=cfg["bem"].get("iterative_show", False))
            z = iterative[0]
            rank, condition_estimate = None, iterative[6]
            rank_ok = iterative[1] in (0, 1, 2, 4, 5)
            ls_residual = op@z-scaled_rhs
            normal_residual = np.linalg.norm(op.rmatvec(ls_residual)) / max(
                iterative[5]*np.linalg.norm(scaled_rhs), 1e-300)
        elif bem_backend == "dense":
            schur = np.empty((rows_count, nb), complex)
            chunk = cfg["bem"]["schur_chunk"]
            for begin in range(0, nb, chunk):
                end = min(nb, begin+chunk)
                response = factor.solve(boundary_rhs[:, begin:end].toarray())
                pe = Pe@response+Ewall[:, begin:end].toarray()
                qe = Qe@response+Eap[:, begin:end].toarray()
                schur[:, begin:end] = Lp@pe+Lq@qe
            scaled = weights[:, None]*schur*xscale[None]
            z, _, rank, singular = linalg.lstsq(
                scaled, scaled_rhs, cond=cfg["bem"]["rcond"], lapack_driver="gelsd")
            rank_ok = rank == nb
            condition_estimate = singular[0]/singular[-1]
            ls_residual = scaled@z-scaled_rhs
            normal_residual = np.linalg.norm(np.conj(scaled.T@np.conj(ls_residual))) / max(
                np.linalg.norm(scaled, "fro")*np.linalg.norm(scaled_rhs), 1e-300)
        stage("schur_finished", normal_residual=normal_residual,
              condition_estimate=condition_estimate, operator_calls=operator_calls)
        x = xscale*z
        y = factor.solve(b+boundary_rhs@x)
    finally:
        if isinstance(factor, PardisoFactor):
            factor.solver.release()
    Ured, pi, current = y[:nr], y[nr:nr+npf], y[-1]
    U = model.T@Ured
    pe, qe = Pe@y+Ewall@x, Qe@y+Eap@x
    bem_residual = Lp@pe+Lq@qe
    bem_denominator = max(np.linalg.norm(Lp@pe)+np.linalg.norm(Lq@qe), 1e-300)
    bem_relative = np.linalg.norm(bem_residual)/bem_denominator
    chief_denominator = max(np.linalg.norm(Lp[nb:]@pe)+np.linalg.norm(Lq[nb:]@qe),
                            1e-300)
    chief_relative = np.linalg.norm(bem_residual[nb:])/chief_denominator
    boundary_denominator = max(np.linalg.norm(Lp[:nb]@pe)+np.linalg.norm(Lq[:nb]@qe),
                               1e-300)
    boundary_relative = np.linalg.norm(bem_residual[:nb])/boundary_denominator
    # Preserve the condensed boundary solve before potentially costly diagnostics.
    # This is a diagnostic checkpoint, never a successful frequency result.
    atomic_npz(output / f"{key}_boundary.npz",
               compatibility_hash=np.array(signature), frequency_Hz=np.array(f),
               voltage=np.array(voltage), core_solution=y, boundary_unknowns=x,
               p_exterior=pe, q_exterior=qe, chief_points_m=chiefs,
               bem_residual=bem_residual)
    stage("boundary_checkpoint_saved", boundary_relative_residual=boundary_relative,
          chief_relative_residual=chief_relative)
    physical = _physical_residual(S, y, b+boundary_rhs@x)
    block_residuals = {}
    for name, sl in (("mechanical", slice(0, nr)),
                     ("interior_acoustic", slice(nr, nr+npf)),
                     ("electrical", slice(nr+npf, ncore))):
        block_residuals[name] = _physical_residual(
            S[sl], y, (b+boundary_rhs@x)[sl])
    trace_pressure_error = pe[ap]-(pressure_trace@pi)[ap]
    wall_error = qe[wall]-(Qe@y)[wall]
    observation_angles = np.arange(0., 181., 5.)
    points = []
    for plane in ("XY", "XZ"):
        for angle in np.deg2rad(observation_angles):
            points.append([np.cos(angle), np.sin(angle) if plane == "XY" else 0.,
                           np.sin(angle) if plane == "XZ" else 0.])
    stage("observation_assembly_started")
    Do, So = operators(np.array(points), model.panels, model.normals,
                       w/air0["c"], cfg["bem"], True)
    directivity = (Do@pe-So@qe).reshape(2, -1)
    stage("reaction_recovery_started")
    coil_displacement = model.full["gc"]@U
    vcoil = 1j*w*coil_displacement
    vmagnet = 1j*w*(model.full["gm"]@U)
    force = motor.BL*current
    F = model.full
    Hfull = F["K"]+1j*F["loss"]+1j*w*F["C"]-w*w*F["M"]
    full_residual = Hfull@U-F["Bi"]@pi+F["Be"]@pe-F["f"]*current
    # Multipliers are diagnostics, not a second correction to the solution.
    multiplier = lsqr(model.constraints.T, -full_residual,
                      atol=1e-10, btol=1e-10, iter_lim=5000)
    reaction = model.constraints.T@multiplier[0]
    loss_struct = .5*w*np.real(np.vdot(U, F["loss"]@U))
    loss_struct += .5*w*w*np.real(np.vdot(U, F["C"]@U))
    loss_air = np.imag(np.vdot(pi, A@pi))/(2*w)
    radiation = .5*np.real(np.sum(
        model.areas*pe*np.conj(1j*qe/(w*air0["rho"]))))
    mechanical_half = .5*np.real(np.dot(F["f"]*current, np.conj(1j*w*U)))
    input_power = .5*np.real(voltage*np.conj(current))
    blocked_power = .5*Rb*abs(current)**2
    emf_power = .5*np.real(motor.BL*vcoil*np.conj(current))
    solid_u = U[:model.space.offset].reshape(-1, 3)
    shell_u = U[model.space.offset:].reshape(-1, 6)
    success = (
        rank_ok
        and physical["normwise_backward"] < cfg["solver"]["backward_tol"]
        and all(v["normwise_backward"] < cfg["solver"]["backward_tol"]
                for v in block_residuals.values())
        and normal_residual < 1e-7
        and bem_relative <= cfg["bem"]["max_relative_residual"]
        and boundary_relative <= cfg["bem"]["max_relative_residual"]
        and chief_relative <= cfg["bem"].get("chief_max_relative_residual", 0.05)
        and np.isfinite(U).all() and np.isfinite(pi).all()
    )
    status = {
        "compatibility_hash": signature, "frequency_Hz": f,
        "voltage_peak_V": voltage, "solver_success": bool(success),
        "benchmark_acceptance": None, "mesh_convergence_acceptance": None,
        "current_peak_A": current, "blocked_impedance_ohm": Zb,
        "total_impedance_ohm": voltage/current if current != 0 else complex(np.nan),
        "Rb_ohm": Rb, "Lb_H": Lb, "coil_average_displacement_m": coil_displacement,
        "coil_average_velocity_m_s": vcoil, "magnetic_average_velocity_m_s": vmagnet,
        "force_full_peak_N": force,
        "max_solid_displacement_m": np.linalg.norm(solid_u, axis=1).max(initial=0),
        "max_shell_displacement_m": np.linalg.norm(shell_u[:, :3], axis=1).max(initial=0),
        "forward_X_pressure_Pa_peak": directivity[0, 0],
        "positive_Z_pressure_Pa_peak": directivity[1, 18],
        "physical_core_residual": physical, "block_residuals": block_residuals,
        "aperture_mean_pressure_error_Pa": normmax(trace_pressure_error),
        "wall_derivative_error_Pa_m": normmax(wall_error),
        "bem_chief_relative_residual": bem_relative,
        "bem_boundary_relative_residual": boundary_relative,
        "chief_relative_residual": chief_relative,
        "chief_equations_in_solve": True,
        "bem_boundary_absolute_inf_Pa": normmax(bem_residual[:nb]),
        "chief_absolute_inf_Pa": normmax(bem_residual[nb:]),
        "least_squares_normal_residual": normal_residual,
        "normal_residual_normalization": "norm(A^H r)/(norm(A)*norm(rhs)); LSMR norm estimate or dense Frobenius norm",
        "schur_algebraic_relative_residual": normal_residual,
        "schur_rank": rank, "schur_columns": nb,
        "schur_condition_estimate": condition_estimate,
        "schur_backend": cfg["bem"].get("backend", "dense"),
        "iterative_stop_code": iterative[1] if bem_backend == "iterative" else None,
        "iterative_iterations": iterative[2] if bem_backend == "iterative" else None,
        "operator_calls": operator_calls,
        "phase_times_s": phase_times,
        "constraint_violation": normmax(model.constraints@U),
        "reaction_recovery_residual": normmax(full_residual+reaction),
        "reaction_recovery_lsqr_status": multiplier[1],
        "interior_interface_resultant": _resultant(model, F["Bi"]@pi),
        "exterior_interface_resultant": _resultant(model, -F["Be"]@pe),
        "constraint_reaction_resultant": _resultant(model, reaction),
        "power_W": {
            "electrical_input_full": input_power,
            "blocked_electrical_loss_full": blocked_power,
            "absolute_coil_back_emf_work_full": emf_power,
            "electrical_identity_error": input_power-blocked_power-emf_power,
            "mechanical_motor_half": mechanical_half,
            "mechanical_motor_reconstructed_full": 2*mechanical_half,
            "structural_dissipation_half": loss_struct,
            "interior_fluid_dissipation_half": loss_air,
            "exterior_outgoing_power_half": radiation,
            "mechanical_balance_half": mechanical_half-loss_struct-loss_air-radiation,
            "absolute_feedback_minus_relative_work_full": emf_power-2*mechanical_half,
            "predicted_feedback_difference_full":
                .5*np.real(force*np.conj(vmagnet))
        },
        "drilling_energy_fraction": float(
            abs(np.vdot(U, F["drill"]@U))/max(abs(np.vdot(U, F["K"]@U)), 1e-300)),
        "slits": slit, "air": air0, "chief_points_m": chiefs,
        "elapsed_s": time.perf_counter()-start, "peak_process_memory_bytes": peak_memory_bytes(),
        "physical_core_nnz": S.nnz,
        "core_backend": cfg["solver"].get("core_backend", "direct"),
        "preconditioner_shift": getattr(factor, "preconditioner_shift", None),
        "factor_L_nnz": factor.lu.L.nnz if factor.lu is not None else None,
        "factor_U_nnz": factor.lu.U.nnz if factor.lu is not None else None,
        "units": {"U": "m and dimensionless axial rotations", "p": "Pa peak",
                  "q": "Pa/m peak", "I": "A peak"},
        "harmonic_convention": "exp(+i omega t)",
        "bem_residual_note": "CHIEF and collocation are least-squares equations; FEM, "
                             "mechanical, aperture, wall and electrical constraints are exact"
    }
    destination = output if success else output / "failed"
    destination.mkdir(parents=True, exist_ok=True)
    atomic_npz(
        destination / f"{key}.npz",
        frequency_Hz=np.array(f), voltage=np.array(voltage), current=np.array(current),
        U=U, U_reduced=Ured, p_interior=pi, p_exterior=pe, q_exterior=qe,
        directivity=directivity, theta_deg=observation_angles,
        blocked_impedance=np.array(Zb), force=np.array(force),
        coil_velocity=np.array(vcoil), coil_displacement=np.array(coil_displacement),
        ext_triangles=model.ext, ext_normals=model.normals,
        ext_areas=model.areas, chiefs=chiefs,
        constraint_multipliers=multiplier[0],
        bem_residual=bem_residual, physical_residual=S@y-b-boundary_rhs@x
    )
    atomic_json(destination / f"{key}.json", status)
    stage("finished" if success else "numerical_acceptance_failed",
          solver_success=bool(success), results_directory=str(destination))
    if not success:
        raise ArithmeticError(f"{f:g} Hz failed acceptance of numerical solve; "
                              f"raw results preserved in {destination}")
    return status


def structural_modes(model: Model, cfg: dict, output: Path, count: int = 20):
    start = time.perf_counter()
    r = model.reduced
    K, C, M = r["K"]+1j*r["loss"], r["C"], r["M"]
    n = K.shape[0]
    shift_f = complex(*cfg["modes"]["shift_Hz"])
    sigma = 2j*np.pi*shift_f
    dynamic = K+sigma*C+sigma*sigma*M
    modal_columns = np.full(n, cfg["solver"]["displacement_scale_m"])
    rotation = ((model.free >= model.space.offset)
                & ((model.free-model.space.offset) % 6 >= 3))
    modal_columns[rotation] = cfg["solver"]["rotation_scale"]
    core_backend = cfg["solver"].get("core_backend", "direct")
    lu = (PardisoFactor(dynamic, modal_columns) if core_backend == "pardiso"
          else ScaledFactor(dynamic, modal_columns))

    def apply(state):
        u0, v0 = state[:n], state[n:]
        u = -lu.solve(M@v0+(C+sigma*M)@u0)
        return np.r_[u, u0+sigma*u]

    operator = LinearOperator((2*n, 2*n), matvec=apply, dtype=np.complex128)
    candidates = min(max(cfg["modes"]["candidates"], 2*count+6), 2*n-2)
    try:
        mu, vectors = eigs(
            operator, k=candidates, which="LM", tol=cfg["modes"]["tol"],
            maxiter=cfg["modes"]["maxiter"],
            v0=np.random.default_rng(1841).normal(size=2*n).astype(complex)
        )
    finally:
        if isinstance(lu, PardisoFactor):
            lu.solver.release()
    lambdas = sigma+1/mu
    frequencies = -1j*lambdas/(2*np.pi)
    valid = np.flatnonzero(
        (frequencies.real > cfg["modes"]["rigid_Hz"])
        & (frequencies.imag >= -1e-7)
    )
    valid = valid[np.argsort(frequencies[valid].real)]
    if len(valid) < count:
        atomic_json(output / "modes_failure.json", {
            "returned_candidates": frequencies, "usable": len(valid),
            "requested": count,
            "message": "Increase candidates/change explicit search shift; no artificial supports added"
        })
        raise ArithmeticError("Insufficient positive-frequency damped structural modes")
    selected = valid[:count]
    full_modes, residuals, mass_norms = [], [], []
    for j in selected:
        u = vectors[:n, j]
        mass = np.real(np.vdot(u, M@u))
        if mass <= 1e-30:
            raise ArithmeticError("Near-zero-mass mode; cannot normalize it as a structural mode")
        u /= np.sqrt(mass)
        lam = lambdas[j]
        residual = K@u+lam*(C@u)+lam*lam*(M@u)
        denominator = np.linalg.norm(K@u)+abs(lam)*np.linalg.norm(C@u)
        denominator += abs(lam)**2*np.linalg.norm(M@u)
        residuals.append(np.linalg.norm(residual)/max(denominator, 1e-300))
        full_modes.append(model.T@u)
        mass_norms.append(float(np.real(np.vdot(u, M@u))))
    full_modes = np.column_stack(full_modes)
    atomic_npz(output / "modes.npz", frequency_Hz=frequencies[selected],
               lambda_s=lambdas[selected], U=full_modes,
               candidate_frequencies_Hz=frequencies, residuals=np.array(residuals))
    summary = {
        "solver_success": max(residuals) < 1e-6,
        "frequency_Hz": frequencies[selected], "polynomial_residuals": residuals,
        "mass_norms": mass_norms,
        "constraint_violation": normmax(model.constraints@full_modes),
        "search": "shift-invert QEP, LM in inverse pencil; not COMSOL eigwhich=lr",
        "source_search_provenance": {"shift_Hz": 1, "eigwhich": "lr"},
        "damping": "K+i K_loss + lambda C + lambda^2 M; lambda=i omega",
        "acoustics_active": False, "elapsed_s": time.perf_counter()-start,
        "peak_process_memory_bytes": peak_memory_bytes(),
        "benchmark_acceptance": None
    }
    atomic_json(output / "modes.json", summary)
    if not summary["solver_success"]:
        raise ArithmeticError("Dry modal polynomial residual failed")
    return summary


def export_fields(output: Path, frequency: float | None = None, mode: int | None = None):
    import meshio
    from .mesh import restore_geometry
    m, s = restore_geometry(output / "geometry.npz")
    if mode is not None:
        z = np.load(output / "modes.npz", allow_pickle=False)
        U = z["U"][:, mode-1]
        p, prefix = None, f"mode_{mode:03d}"
    else:
        prefix = frequency_key(float(frequency))
        z = np.load(output / f"{prefix}.npz", allow_pickle=False)
        U, p = z["U"], z["p_interior"]
    dest = output / "vtk"
    dest.mkdir(exist_ok=True)
    solid = np.flatnonzero(s.scell[:, 0] >= 0)
    perm10 = [0, 1, 2, 3, 4, 7, 5, 6, 8, 9]
    us = U[:s.offset].reshape(-1, 3)
    meshio.write(dest / f"{prefix}_solid.vtu", meshio.Mesh(
        m.x[s.snode], [("tetra10", s.scell[solid][:, perm10])],
        point_data={"u_real_m": us.real, "u_imag_m": us.imag},
        cell_data={"domain": [m.domain[solid]]}))
    shell = np.flatnonzero(s.hcell[:, 0] >= 0)
    uh = U[s.offset:].reshape(-1, 6)
    meshio.write(dest / f"{prefix}_shell.vtu", meshio.Mesh(
        m.x[s.hnode], [("triangle6", s.hcell[shell][:, [0, 1, 2, 3, 5, 4]])],
        point_data={"u_real_m": uh[:, :3].real, "u_imag_m": uh[:, :3].imag,
                    "theta_real": uh[:, 3:].real, "theta_imag": uh[:, 3:].imag},
        cell_data={"boundary": [m.tag[shell]]}))
    if p is not None:
        acoustic = np.flatnonzero(s.pcell[:, 0] >= 0)
        meshio.write(dest / f"{prefix}_air.vtu", meshio.Mesh(
            m.x[s.pnode], [("tetra10", s.pcell[acoustic][:, perm10])],
            point_data={"p_real_Pa_peak": p.real, "p_imag_Pa_peak": p.imag},
            cell_data={"domain": [m.domain[acoustic]]}))
        ext = z["ext_triangles"]
        meshio.write(dest / f"{prefix}_exterior.vtu", meshio.Mesh(
            m.x, [("triangle", m.btri[ext])],
            cell_data={
                "boundary": [m.tag[ext]],
                "p_real_Pa_peak": [z["p_exterior"].real],
                "p_imag_Pa_peak": [z["p_exterior"].imag],
                "q_real_Pa_per_m": [z["q_exterior"].real],
                "q_imag_Pa_per_m": [z["q_exterior"].imag]
            }))
