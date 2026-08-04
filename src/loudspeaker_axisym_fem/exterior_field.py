from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Sequence
import math
import numpy as np
from skfem import MeshTri, FacetBasis


@dataclass(frozen=True)
class ExteriorMetrics:
    frequency_Hz: float
    axis_spl_1m_dB: float
    halfspace_power_W: float
    directivity_index_dB: float
    beamwidth_6dB_deg: float | None
    reference_pressure_rms_Pa: float


def _green_and_dgdn(
    robs: np.ndarray,
    zobs: np.ndarray,
    rs: np.ndarray,
    zs: np.ndarray,
    nr: np.ndarray,
    nz: np.ndarray,
    phi: np.ndarray,
    k: float,
    mirror: bool = False,
):
    """Axisymmetric Green kernel and normal derivative for exp(-i k R)/(4 pi R).

    `robs`, `zobs` are column vectors.  Boundary meridian samples are rotated by
    azimuth `phi`.  The surface normal is `(nr cos(phi), nr sin(phi), nz)`.
    If `mirror` is true, add a rigid-baffle image across z=0; this matches the
    half-space formulation used for the front radiation of a baffled driver.
    """
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    xs = rs * cphi
    ys = rs * sphi
    dx = xs - robs
    dy = ys
    dz = zs - zobs
    R = np.maximum(np.sqrt(dx * dx + dy * dy + dz * dz), 1e-14)
    e = np.exp(-1j * k * R)
    G = e / (4.0 * np.pi * R)
    dGdR = e * (-1j * k * R - 1.0) / (4.0 * np.pi * R * R)
    dot = (dx * (nr * cphi) + dy * (nr * sphi) + dz * nz) / R
    dGdn = dGdR * dot
    if mirror:
        dz2 = -zs - zobs
        R2 = np.maximum(np.sqrt(dx * dx + dy * dy + dz2 * dz2), 1e-14)
        e2 = np.exp(-1j * k * R2)
        G2 = e2 / (4.0 * np.pi * R2)
        dGdR2 = e2 * (-1j * k * R2 - 1.0) / (4.0 * np.pi * R2 * R2)
        # mirrored source point has z -> -z and normal z component -> -nz
        dot2 = (dx * (nr * cphi) + dy * (nr * sphi) + dz2 * (-nz)) / R2
        G = G + G2
        dGdn = dGdn + dGdR2 * dot2
    return G, dGdn


def facet_samples_from_fe(
    mesh: MeshTri,
    element,
    facets: np.ndarray,
    field: np.ndarray,
    intorder: int = 6,
    prefer_inside_radius: float | None = None,
    force_radial_normals: bool = False,
):
    """Return boundary/interface quadrature samples using FE values and FE gradients.

    Parameters
    ----------
    mesh, element, facets, field:
        scikit-fem mesh, element and global FE vector.
    prefer_inside_radius:
        For internal HK interfaces between physical air and PML, evaluate the FE
        gradient on the side whose adjacent element centroid radius is smaller
        than this radius.  This avoids using the PML-side gradient.
    force_radial_normals:
        Replace facet normals by the exact radial normal `(r,z)/sqrt(r^2+z^2)`;
        useful for spherical/circular HK surfaces where a polygonal facet normal
        is only a geometric approximation.  The gradient is still the FE gradient.
    """
    facets = np.asarray(facets, dtype=int)
    if facets.size == 0:
        raise ValueError("No facets supplied for exterior-field integration.")

    def make(side: int):
        fb = FacetBasis(mesh, element, facets=facets, intorder=intorder, side=side)
        df = fb.interpolate(field)
        x = np.asarray(fb.global_coordinates())  # (2, nfacet, nq)
        n = np.asarray(fb.normals)
        if force_radial_normals:
            rr = x[0]
            zz = x[1]
            rad = np.maximum(np.sqrt(rr * rr + zz * zz), 1e-14)
            n = np.stack([rr / rad, zz / rad], axis=0)
        grad = np.asarray(df.grad)
        dpdn = grad[0] * n[0] + grad[1] * n[1]
        return {
            "fb": fb,
            "tind": np.asarray(fb.tind),
            "x": x,
            "n": n,
            "dx": np.asarray(fb.dx),
            "p": np.asarray(df),
            "dpdn": dpdn,
        }

    # Boundary facets only have one valid side.  Internal facets have two.
    s0 = make(0)
    if prefer_inside_radius is None:
        chosen = s0
    else:
        try:
            s1 = make(1)
        except Exception:
            s1 = None
        if s1 is None:
            chosen = s0
        else:
            centers = mesh.p[:, mesh.t].mean(axis=1)
            rad0 = np.sqrt(centers[0, s0["tind"]] ** 2 + centers[1, s0["tind"]] ** 2)
            rad1 = np.sqrt(centers[0, s1["tind"]] ** 2 + centers[1, s1["tind"]] ** 2)
            # choose per facet the side closer to the physical-domain side
            use0 = np.abs(rad0 - prefer_inside_radius) <= np.abs(rad1 - prefer_inside_radius)
            # If one side is clearly inside and the other outside, prefer inside.
            use0 = np.where((rad0 <= prefer_inside_radius) & (rad1 > prefer_inside_radius), True, use0)
            use0 = np.where((rad1 <= prefer_inside_radius) & (rad0 > prefer_inside_radius), False, use0)
            chosen = {}
            for key in ("x", "n", "dx", "p", "dpdn"):
                a = s0[key].copy()
                b = s1[key]
                if a.ndim == 3:
                    a[:, ~use0, :] = b[:, ~use0, :]
                elif a.ndim == 2:
                    a[~use0, :] = b[~use0, :]
                chosen[key] = a
            chosen["tind"] = np.where(use0, s0["tind"], s1["tind"])

    x = chosen["x"].reshape(2, -1)
    n = chosen["n"].reshape(2, -1)
    ds_w = chosen["dx"].reshape(-1)
    pvals = chosen["p"].reshape(-1)
    dpdn = chosen["dpdn"].reshape(-1)
    return x[0], x[1], n[0], n[1], ds_w, pvals, dpdn


def hk_pressure_from_samples(
    frequency_Hz: float,
    c0: float,
    rs: np.ndarray,
    zs: np.ndarray,
    nr: np.ndarray,
    nz: np.ndarray,
    ds_w: np.ndarray,
    p_boundary: np.ndarray,
    dpdn_boundary: np.ndarray,
    obs_r: np.ndarray | float,
    obs_z: np.ndarray | float,
    nphi: int = 64,
    mirror: bool = False,
    sign: int = -1,
):
    """Evaluate exterior pressure from meridian-surface HK facet samples.

    The default `sign=-1` uses ∫(p dG/dn - G dp/dn)dS for exterior
    reconstruction with G=exp(-ikR)/(4πR), validated by the included monopole
    benchmark.  `sign=1` reproduces the older project convention; it has the same
    magnitude for many symmetric checks but the opposite complex phase.
    """
    obs_r = np.atleast_1d(np.asarray(obs_r, dtype=float))
    obs_z = np.atleast_1d(np.asarray(obs_z, dtype=float))
    if obs_r.shape != obs_z.shape:
        obs_r, obs_z = np.broadcast_arrays(obs_r, obs_z)
    xphi, wphi0 = np.polynomial.legendre.leggauss(int(nphi))
    phi = np.pi * (xphi + 1.0)  # 0..2pi
    wphi = np.pi * wphi0
    rr = np.repeat(rs, nphi)[None, :]
    zz = np.repeat(zs, nphi)[None, :]
    nnr = np.repeat(nr, nphi)[None, :]
    nnz = np.repeat(nz, nphi)[None, :]
    ph = np.tile(phi, len(rs))[None, :]
    weights = (np.repeat(ds_w, nphi) * np.repeat(rs, nphi) * np.tile(wphi, len(rs)))[None, :]
    pp = np.repeat(p_boundary, nphi)[None, :]
    dd = np.repeat(dpdn_boundary, nphi)[None, :]
    ro = obs_r.reshape(-1, 1)
    zo = obs_z.reshape(-1, 1)
    k = 2.0 * np.pi * float(frequency_Hz) / float(c0)
    out = np.zeros(ro.shape[0], dtype=complex)
    for s in range(0, ro.shape[0], 32):
        e = min(s + 32, ro.shape[0])
        G, dGdn = _green_and_dgdn(ro[s:e], zo[s:e], rr, zz, nnr, nnz, ph, k, mirror=mirror)
        out[s:e] = sign * np.sum((G * dd - pp * dGdn) * weights, axis=1)
    return out.reshape(obs_r.shape)


def hk_pressure_from_fe(
    frequency_Hz: float,
    c0: float,
    mesh: MeshTri,
    element,
    facets: np.ndarray,
    field: np.ndarray,
    obs_r: np.ndarray | float,
    obs_z: np.ndarray | float,
    nphi: int = 64,
    intorder: int = 6,
    mirror: bool = False,
    prefer_inside_radius: float | None = None,
    force_radial_normals: bool = False,
    sign: int = -1,
):
    samples = facet_samples_from_fe(
        mesh,
        element,
        facets,
        field,
        intorder=intorder,
        prefer_inside_radius=prefer_inside_radius,
        force_radial_normals=force_radial_normals,
    )
    return hk_pressure_from_samples(
        frequency_Hz,
        c0,
        *samples,
        obs_r=obs_r,
        obs_z=obs_z,
        nphi=nphi,
        mirror=mirror,
        sign=sign,
    )


def intensity_power_from_samples(
    frequency_Hz: float,
    rho0: float,
    rs: np.ndarray,
    ds_w: np.ndarray,
    p_boundary: np.ndarray,
    dpdn_boundary: np.ndarray,
):
    """Acoustic power through a rotational surface from FE pressure flux.

    Uses v_n = -(∂p/∂n)/(i omega rho0), compatible with the existing solver
    convention, and integrates 0.5 Re{p v_n*} over 2πr ds.
    """
    omega = 2.0 * np.pi * float(frequency_Hz)
    vn = -dpdn_boundary / (1j * omega * float(rho0))
    return float(np.real(np.sum(0.5 * p_boundary * np.conj(vn) * 2.0 * np.pi * rs * ds_w)))


def halfspace_power_from_directivity(
    pressure: np.ndarray,
    theta_rad: np.ndarray,
    rho0: float,
    c0: float,
    radius_m: float = 1.0,
):
    """Integrate far-field half-space power from pressure samples vs polar angle."""
    p = np.asarray(pressure)
    th = np.asarray(theta_rad, dtype=float)
    prms2 = 0.5 * np.abs(p) ** 2
    # I = p_rms^2 / (rho c), dS = R^2 2π sinθ dθ.
    integrand = prms2 / (rho0 * c0) * (radius_m ** 2) * 2.0 * np.pi * np.sin(th)
    return float(np.trapz(integrand, th))


def directivity_index_halfspace(pressure: np.ndarray, theta_rad: np.ndarray):
    """Half-space DI using axis pressure divided by hemispherical mean pressure^2.

    DI_hs = 10 log10( |p(0)|^2 / <|p|^2>_hemisphere ), where
    <|p|^2> = ∫|p|^2 sinθ dθ / ∫sinθ dθ over θ in [0, π/2].
    """
    p = np.asarray(pressure)
    th = np.asarray(theta_rad, dtype=float)
    denom = float(np.trapz(np.abs(p) ** 2 * np.sin(th), th) / max(np.trapz(np.sin(th), th), 1e-300))
    axis = float(np.abs(p[0]) ** 2)
    return 10.0 * math.log10(max(axis, 1e-300) / max(denom, 1e-300))


def beamwidth_6db(theta_deg: Sequence[float], pressure: Sequence[complex]) -> float | None:
    spl_rel = 20.0 * np.log10(np.maximum(np.abs(pressure) / max(abs(pressure[0]), 1e-300), 1e-300))
    th = np.asarray(theta_deg, dtype=float)
    below = np.nonzero(spl_rel <= -6.0)[0]
    if below.size == 0:
        return None
    i = int(below[0])
    if i == 0:
        return float(th[0])
    # linear interpolation in dB between samples i-1 and i
    x0, x1 = spl_rel[i - 1], spl_rel[i]
    t0, t1 = th[i - 1], th[i]
    if abs(x1 - x0) < 1e-12:
        return float(t1)
    return float(t0 + (-6.0 - x0) * (t1 - t0) / (x1 - x0))


def spl_db_from_pressure_peak(p_peak: np.ndarray | complex, p_ref: float = 20e-6):
    return 20.0 * np.log10(np.maximum(np.abs(p_peak) / math.sqrt(2.0), 1e-300) / float(p_ref))
