from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

from .core import AIR_DOMAINS, SOLID_DOMAINS, SHELL_GROUPS, rows
from .mesh import EDGES, TRI_EDGES


@dataclass(frozen=True)
class Material:
    E: float
    nu: float
    rho: float
    loss: float = 0.0
    beta: float = 0.0
    thickness: float = 0.0


def steel(T: float) -> tuple[float, float]:
    return (200e9 + (T - 293.15) * (166.6e9 - 200e9) / 500,
            0.30 + (T - 293.15) * 0.015 / 500)


def solid_material(domain: int, T: float = 293.15) -> Material:
    if domain not in SOLID_DOMAINS:
        raise ValueError(f"No solid material for domain {domain}")
    if domain == 26:
        return Material(110e9, 0.35, 4500, 0.05)
    if domain in (23, 27):
        E, nu = steel(T)
        return Material(E, nu, 7850, 0.01)
    if domain == 24:
        return Material(200e9, 0.30, 5000, 0.01)
    return Material(4e9, 0.30, 900, 0.07)


def shell_material(tag: int, T: float = 293.15) -> Material:
    name = next((k for k, v in SHELL_GROUPS.items() if tag in v), None)
    omega_loss = 2 * np.pi * 40
    values = {
        "composite": Material(2e9, .42, 1200, .04, 0, .001),
        "cloth": Material(.58e9, .30, 650, 0, .14 / omega_loss, .0004),
        "foam": Material(5e6, .40, 67, 0, .46 / omega_loss, .0015),
        "glass": Material(70e9, .33, 2000, .04, 0, .0002),
        "basket": Material(*steel(T), 7850, .01, 0, .0008)
    }
    if name is None:
        raise ValueError(f"No shell material for boundary {tag}")
    return values[name]


def air(T: float = 293.15, pressure: float = 101325.0) -> dict:
    R, molar, gamma = 8.31446261815324, .02897, 1.4
    rho = pressure * molar / (R * T)
    c = np.sqrt(gamma * R * T / molar)
    eta = np.polynomial.polynomial.polyval(
        T, [-8.38278e-7, 8.35717342e-8, -7.69429583e-11,
            4.6437266e-14, -1.06585607e-17])
    Cp = np.polynomial.polynomial.polyval(
        T, [1047.63657, -.372589265, 9.45304214e-4,
            -6.02409443e-7, 1.2858961e-10])
    kappa = np.polynomial.polynomial.polyval(
        T, [-.00227583562, 1.15480022e-4, -7.90252856e-8,
            4.11702505e-11, -7.43864331e-15])
    if min(rho, c, eta, Cp, kappa) <= 0:
        raise ValueError("Invalid air coefficients")
    return dict(rho=rho, c=c, eta=eta, Cp=Cp, kappa=kappa,
                gamma=gamma, pressure=pressure, Pr=eta * Cp / kappa)


def tanhc(z: complex) -> tuple[complex, complex]:
    """Return tanh(z)/z and 1-tanh(z)/z without low-frequency cancellation."""
    if abs(z) < .02:
        z2 = z * z
        deficit = z2 / 3 - 2 * z2**2 / 15 + 17 * z2**3 / 315 - 62 * z2**4 / 2835
        return 1 - deficit, deficit
    if z.real > 25:
        e = np.exp(-2 * z)
        ratio = (1 - e) / ((1 + e) * z)
    else:
        ratio = np.tanh(z) / z
    return ratio, 1 - ratio


def equivalent(f: float, height: float, a: dict) -> dict:
    if f <= 0 or height <= 0:
        raise ValueError("Slit coefficients require positive frequency and height")
    w = 2 * np.pi * f
    half = height / 2
    zv = half * np.sqrt(1j * w * a["rho"] / a["eta"])
    zt = half * np.sqrt(1j * w * a["rho"] * a["Cp"] / a["kappa"])
    _, deficit = tanhc(zv)
    thermal, _ = tanhc(zt)
    rho = a["rho"] / deficit
    compressibility = (1 + (a["gamma"] - 1) * thermal) / (
        a["gamma"] * a["pressure"])
    result = dict(
        rho=rho, bulk=1 / compressibility,
        a=1 / rho, b=compressibility,
        viscous_depth_over_half=np.sqrt(2 * a["eta"] / (w * a["rho"])) / half,
        thermal_depth_over_half=np.sqrt(
            2 * a["kappa"] / (w * a["rho"] * a["Cp"])) / half
    )
    if not all(np.isfinite(result[k]) for k in ("rho", "bulk", "a", "b")):
        raise FloatingPointError(f"Nonfinite slit coefficients at {f:g} Hz")
    if rho.imag > 1e-10 * abs(rho) or compressibility.imag > 1e-10 * abs(compressibility):
        raise ValueError("Thermoviscous passivity sign violation")
    return result


class Motor:
    BL = 10.48

    def __init__(self, resistance: np.ndarray, inductance: np.ndarray):
        self.r = np.asarray(resistance, float)
        self.l = np.asarray(inductance, float)
        for name, table in (("Rb", self.r), ("Lb", self.l)):
            if table.ndim != 2 or table.shape[1] != 2:
                raise ValueError(f"{name}: expected frequency,value table")
            if not np.isfinite(table).all() or np.any(np.diff(table[:, 0]) <= 0):
                raise ValueError(f"{name}: nonfinite or unordered input table")
            if np.any(table[:, 1] < 0):
                raise ValueError(f"{name}: negative blocked component")

    @classmethod
    def load(cls, root: Path, strict: bool = True):
        tables = []
        for name in ("int1_source.csv", "int2_source.csv"):
            rr = rows(root / name)
            if strict and len(rr) != 126:
                raise ValueError(f"{name}: expected 126 source rows")
            tables.append(np.array([[float(r["frequency_Hz"]), float(r["value_SI"])]
                                    for r in rr]))
        return cls(*tables)

    def values(self, f: float) -> tuple[float, float, complex]:
        R = float(np.interp(f, self.r[:, 0], self.r[:, 1]))
        L = float(np.interp(f, self.l[:, 0], self.l[:, 1]))
        return R, L, R + 2j * np.pi * f * L


def cross(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


@lru_cache(None)
def unit_gauss(n: int) -> tuple[np.ndarray, np.ndarray]:
    x, w = leggauss(n)
    return (x + 1) / 2, w / 2


@lru_cache(None)
def tri_rule(n: int = 4) -> tuple[np.ndarray, np.ndarray]:
    x, w = unit_gauss(n)
    L, weights = [], []
    for i, u in enumerate(x):
        for j, v in enumerate(x):
            a, b = u, (1 - u) * v
            L.append([1 - a - b, a, b])
            weights.append(w[i] * w[j] * (1 - u))
    return np.array(L), np.array(weights)


@lru_cache(None)
def tet_rule(n: int = 4) -> tuple[np.ndarray, np.ndarray]:
    x, w = unit_gauss(n)
    L, weights = [], []
    for i, u in enumerate(x):
        for j, v in enumerate(x):
            for k, z in enumerate(x):
                a, b, c = u, (1 - u) * v, (1 - u) * (1 - v) * z
                L.append([1 - a - b - c, a, b, c])
                weights.append(w[i] * w[j] * w[k] * (1 - u)**2 * (1 - v))
    return np.array(L), np.array(weights)


def shape(L: np.ndarray, edges: tuple, gradients: np.ndarray | None = None):
    N = np.r_[L * (2 * L - 1), [4 * L[i] * L[j] for i, j in edges]]
    if gradients is None:
        return N
    dN = np.vstack((
        (4 * L - 1)[:, None] * gradients,
        [4 * (L[i] * gradients[j] + L[j] * gradients[i]) for i, j in edges]
    ))
    return N, dN


def tet_shape(L, gradients=None):
    return shape(np.asarray(L), EDGES, gradients)


def tri_shape(L, gradients=None):
    return shape(np.asarray(L), TRI_EDGES, gradients)


def tet_geometry(p: np.ndarray) -> tuple[float, np.ndarray]:
    J = np.column_stack((p[1] - p[0], p[2] - p[0], p[3] - p[0]))
    det = np.linalg.det(J)
    if det <= 0:
        raise ValueError("tet_geometry received a nonpositive orientation")
    g = np.linalg.inv(J)
    return det / 6, np.vstack((-g.sum(axis=0), g))


@lru_cache(None)
def tetra_mass_unit() -> np.ndarray:
    """Integral N_i N_j / V, by exact barycentric polynomial moments."""
    polynomials = []
    for i in range(4):
        a, b = [0] * 4, [0] * 4
        a[i], b[i] = 2, 1
        polynomials.append([(2., a), (-1., b)])
    for i, j in EDGES:
        a = [0] * 4
        a[i] = a[j] = 1
        polynomials.append([(4., a)])
    M = np.zeros((10, 10))
    for i, pi in enumerate(polynomials):
        for j, pj in enumerate(polynomials):
            for ci, ai in pi:
                for cj, aj in pj:
                    alpha = np.array(ai) + aj
                    M[i, j] += ci * cj * 6 * math.prod(
                        math.factorial(int(a)) for a in alpha
                    ) / math.factorial(int(alpha.sum()) + 3)
    return M


def tetra_integrals(V: float) -> np.ndarray:
    return V * np.r_[np.full(4, -1 / 20), np.full(6, 1 / 5)]


def constitutive(E: float, nu: float) -> np.ndarray:
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[np.arange(3), np.arange(3)] += 2 * mu
    D[3:, 3:] = mu * np.eye(3)
    return D


def strain_matrix(g: np.ndarray) -> np.ndarray:
    # Engineering strain order: xx, yy, zz, xy, yz, xz.
    B = np.zeros((6, 3 * len(g)))
    for j, (x, y, z) in enumerate(g):
        B[:, 3*j:3*j+3] = [
            [x, 0, 0], [0, y, 0], [0, 0, z],
            [y, x, 0], [0, z, y], [z, 0, x]
        ]
    return B


def scalar_element(p: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    V, gL = tet_geometry(p)
    a, b = .5854101966249685, .1381966011250105
    K = np.zeros((10, 10))
    for j in range(4):
        L = np.full(4, b)
        L[j] = a
        _, g = tet_shape(L, gL)
        K += V / 4 * (g @ g.T)
    return K, V * tetra_mass_unit(), V


def solid_element(p: np.ndarray, mat: Material):
    V, gL = tet_geometry(p)
    D = constitutive(mat.E, mat.nu)
    K = np.zeros((30, 30))
    a, b = .5854101966249685, .1381966011250105
    for j in range(4):
        L = np.full(4, b)
        L[j] = a
        _, g = tet_shape(L, gL)
        B = strain_matrix(g)
        K += V / 4 * (B.T @ D @ B)
    M = mat.rho * V * np.kron(tetra_mass_unit(), np.eye(3))
    return K, M, V


def shell_frame(p: np.ndarray):
    t1 = p[1] - p[0]
    t1 /= np.linalg.norm(t1)
    n = np.cross(p[1] - p[0], p[2] - p[0])
    area = np.linalg.norm(n) / 2
    if area <= 0:
        raise ValueError("Degenerate shell triangle")
    n /= 2 * area
    t2 = np.cross(n, t1)
    xy = (p - p[0]) @ np.column_stack((t1, t2))
    J = np.column_stack((xy[1], xy[2]))
    inv = np.linalg.inv(J)
    gL = np.vstack((-inv.sum(axis=0), inv))
    return area, t1, t2, n, gL


def shell_element(p: np.ndarray, mat: Material, options: dict):
    """P2 u and P2 axial theta, ordered [ux,uy,uz,tx,ty,tz] per node."""
    A, t1, t2, n, gL = shell_frame(p)
    h = max(np.linalg.norm(a - b) for a, b in ((p[0], p[1]),
                                              (p[0], p[2]), (p[1], p[2])))
    t = mat.thickness
    D = mat.E / (1 - mat.nu**2) * np.array([
        [1, mat.nu, 0], [mat.nu, 1, 0], [0, 0, (1 - mat.nu) / 2]
    ])
    mu = mat.E / (2 * (1 + mat.nu))
    cap = min(1., options["fluctuation_cap"] * (t / h)**2)
    Ls, ws = tri_rule(4)
    samples = []
    mean_shear = np.zeros((2, 36))
    K, M, drill = (np.zeros((36, 36)) for _ in range(3))
    P = np.eye(3) - np.outer(n, n)
    for L, wr in zip(Ls, ws):
        N, g = tri_shape(L, gL)
        weight = 2 * A * wr
        Bm, Bb, Bs, Bd = (np.zeros((r, 36)) for r in (3, 3, 2, 1))
        Nu, Nr = np.zeros((3, 36)), np.zeros((3, 36))
        for j, (Nj, (gx, gy)) in enumerate(zip(N, g)):
            u, r = slice(6*j, 6*j+3), slice(6*j+3, 6*j+6)
            Bm[:, u] = [gx*t1, gy*t2, gy*t1 + gx*t2]
            # a=theta cross n; a_1=theta.t2; a_2=-theta.t1.
            Bb[:, r] = [gx*t2, -gy*t1, gy*t2 - gx*t1]
            Bs[:, u] = [gx*n, gy*n]
            Bs[:, r] = [Nj*t2, -Nj*t1]
            Bd[0, u] = -.5 * (gx*t2 - gy*t1)
            Bd[0, r] = Nj*n
            Nu[:, u] = Nj*np.eye(3)
            Nr[:, r] = Nj*P
        K += weight * (t * Bm.T @ D @ Bm + t**3 / 12 * Bb.T @ D @ Bb)
        kd = weight * options["drilling"] * mu * t * Bd.T @ Bd
        K += kd
        drill += kd
        M += weight * mat.rho * (t * Nu.T @ Nu + t**3 / 12 * Nr.T @ Nr)
        mean_shear += weight / A * Bs
        samples.append((weight, Bs))
    shear_factor = options["shear_correction"] * mu * t
    K += shear_factor * A * mean_shear.T @ mean_shear
    for weight, Bs in samples:
        residual = Bs - mean_shear
        K += shear_factor * cap * weight * residual.T @ residual
    return K, M, drill, cap
