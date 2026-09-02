r"""Reference-only axisymmetric acoustics for the A/B/C enclosure demonstrators.

This module is the deliberately scoped ``Stage 3B2 reference A/B validation
layer`` on top of the Stage 3B1 open-back and sealed compatibility core; it is
not a complete Stage 3 implementation.
It loads the audited reference mesh, keeps only ``air_*`` triangles in the pressure space,
and assembles the P1 weak form

.. math::

   \int 2\pi r\left[\rho^{-1}\nabla p\cdot\nabla q
   -\omega^2 K^{-1}pq\right]dA
   = \int 2\pi r q\,\rho^{-1}\partial_n p\,d\Gamma.

The convention is ``exp(+i omega t)``.  A reference planar piston has global
velocity ``v_z = +1 m/s`` and the load is generated from the pressure-domain
outward normal, not from a front/back sign lookup.  This is not a production
moving-interface implementation and it intentionally does not import the old
``fem_solver`` or any production CLI path.

The PML is a spherical radial coordinate stretch.  For ``R=sqrt(r^2+z^2)``
and ``eta=(R-R0)/thickness`` in the PML,

``s_R = 1 - i alpha eta**m``

and ``tilde_R`` is obtained by integrating this same stretch from ``R0``.
``s_t = tilde_R/R``.  The axisymmetric meridian tensor uses radial coefficient
``s_t**2/s_R``, tangential coefficient ``s_R`` and mass coefficient
``s_R*s_t**2``.  At the interface all three are exactly one.  The PML is
applied by physical domain name, never by a scalar sponge or by radius alone.
"""

from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import meshio
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from skfem import (
    Basis,
    BilinearForm,
    ElementTriP1,
    ElementTriP2,
    FacetBasis,
    LinearForm,
    MeshTri,
    asm,
)
from skfem.helpers import grad

from .enclosure_geometry import (
    BOUNDARY_PHYSICAL_TAGS,
    DOMAIN_PHYSICAL_TAGS,
    case_id_for_config,
    expected_domain_names,
)
from .enclosure_schema import EnclosureConfig, load_enclosure_config
from .enclosure_topology import audit_mesh
from .thermoviscous_boundaries import (
    ThermoviscousAirProperties,
    assess_bli_applicability,
    bli_bilinear_coefficients,
    bli_dissipation,
)
from .exterior_field import (
    facet_samples_from_fe,
    hk_pressure_from_samples,
    intensity_power_from_samples,
)


REFERENCE_PLANAR_PISTON_IDENTITY = "reference planar piston"
REFERENCE_PLANAR_PISTON_FRONT = "reference_planar_piston_front"
REFERENCE_PLANAR_PISTON_BACK = "reference_planar_piston_back"
OUTER_PML_BOUNDARY = "outer_pml_boundary"
HK_BOUNDARIES = ("hk_front", "hk_rear")
PML_DOMAINS = frozenset(("air_pml_front", "air_pml_rear"))

DEFAULT_REFERENCE_VELOCITY_M_S = 1.0
DEFAULT_PML_ALPHA = 4.0
DEFAULT_PML_EXPONENT = 2
DEFAULT_PML_TARGET_ATTENUATION_NEPERS = 8.0
EXPLICIT_PML_MODE = "explicit_alpha"
TARGET_PML_MODE = "target_nepers"
DEFAULT_PML_MODE = EXPLICIT_PML_MODE
SUPPORTED_PML_MODES = frozenset((TARGET_PML_MODE, EXPLICIT_PML_MODE))
SUPPORTED_ACOUSTIC_CASES = frozenset(("A", "B", "C"))
PML_RADIUS_TOLERANCE_M = 2.0e-10
PML_GEOMETRY_TOLERANCE_M = 2.0e-8
CAVITY_VOLUME_RELATIVE_TOLERANCE = 5.0e-3
FAR_FIELD_MIN_KR = 20.0
FAR_FIELD_REFERENCE_DISTANCE_M = 1.0


def sha256_file(path: str | Path) -> str:
    """Hash an input without changing it."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_points_rz(points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("axisymmetric mesh points must contain r,z coordinates")
    return np.asarray(values[:, :2], dtype=float)


def axisymmetric_triangle_volumes(
    points_rz: np.ndarray,
    triangles: np.ndarray,
) -> np.ndarray:
    """Return ``2*pi*r_centroid*dA`` for each meridian triangle."""

    points = _as_points_rz(points_rz)
    cells = np.asarray(triangles, dtype=np.int64)
    if cells.ndim != 2 or cells.shape[1] != 3:
        raise ValueError("triangles must have shape (n, 3)")
    vertices = points[cells]
    edge_a = vertices[:, 1] - vertices[:, 0]
    edge_b = vertices[:, 2] - vertices[:, 0]
    area = 0.5 * np.abs(edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0])
    centroid_r = np.mean(vertices[:, :, 0], axis=1)
    return 2.0 * math.pi * centroid_r * area


def axisymmetric_volume(points_rz: np.ndarray, triangles: np.ndarray) -> float:
    """Return the rotated volume of a collection of triangles."""

    return float(np.sum(axisymmetric_triangle_volumes(points_rz, triangles)))


@dataclass
class AxisymmetricP1Operators:
    """Ordinary no-PML P1 operators for independent gradient verification.

    ``stiffness`` is
    ``K = integral(2*pi*r/rho * grad(u).grad(v) dA)`` and
    ``compressibility_mass`` is
    ``C = integral(2*pi*r/K_bulk * u*v dA)``.  No boundary condition is
    applied here; a closed Neumann cavity therefore retains its constant zero
    mode, as required for an independent generalized-eigenmode check.
    """

    mesh: MeshTri
    stiffness: csr_matrix
    compressibility_mass: csr_matrix
    rho0_kg_m3: float
    bulk_modulus_Pa: float

    @property
    def dof_count(self) -> int:
        return int(self.mesh.p.shape[1])

    @property
    def mass(self) -> csr_matrix:
        """Short alias for the compressibility matrix ``C``."""

        return self.compressibility_mass


def assemble_axisymmetric_operators(
    mesh_or_points: MeshTri | np.ndarray,
    triangles: np.ndarray | None = None,
    rho0_kg_m3: float = 1.0,
    bulk_modulus_Pa: float = 1.0,
    *,
    intorder: int = 4,
) -> AxisymmetricP1Operators:
    """Assemble ordinary axisymmetric P1 stiffness and compressibility mass.

    ``mesh_or_points`` may be an existing scikit-fem ``MeshTri`` or an
    ``(n,2)`` ``(r,z)`` point array paired with ``triangles``.  This low-level
    function has no physical labels, PML, prescribed drive, or solver state;
    it exists specifically so a separately generated closed cavity can test
    the gradient operator through its Neumann eigenmodes.
    """

    rho = float(rho0_kg_m3)
    bulk = float(bulk_modulus_Pa)
    if rho <= 0.0 or bulk <= 0.0:
        raise ValueError("rho0 and bulk modulus must be positive")
    if isinstance(mesh_or_points, MeshTri):
        if triangles is not None:
            raise ValueError("triangles must be omitted when a MeshTri is supplied")
        mesh = mesh_or_points
    else:
        if triangles is None:
            raise ValueError("triangles are required with point coordinates")
        points = _as_points_rz(np.asarray(mesh_or_points, dtype=float))
        cells = np.asarray(triangles, dtype=np.int64)
        if cells.ndim != 2 or cells.shape[1] != 3:
            raise ValueError("triangles must have shape (n, 3)")
        mesh = MeshTri(points.T, cells.T, sort_t=False, validate=True)
    basis = Basis(mesh, ElementTriP1(), intorder=intorder)

    @BilinearForm(dtype=np.float64)
    def stiffness_form(u, v, w):
        gu = grad(u)
        gv = grad(v)
        return 2.0 * math.pi * w.x[0] / rho * (gu[0] * gv[0] + gu[1] * gv[1])

    @BilinearForm(dtype=np.float64)
    def compressibility_form(u, v, w):
        return 2.0 * math.pi * w.x[0] / bulk * u * v

    return AxisymmetricP1Operators(
        mesh=mesh,
        stiffness=asm(stiffness_form, basis).tocsr(),
        compressibility_mass=asm(compressibility_form, basis).tocsr(),
        rho0_kg_m3=rho,
        bulk_modulus_Pa=bulk,
    )


def _integral_r_times_linear_field(
    points_rz: np.ndarray,
    triangles: np.ndarray,
    values: np.ndarray,
) -> complex:
    """Integrate ``2*pi*r*p`` exactly for a P1 field on triangles.

    The product of the linear radius and the linear P1 field is quadratic, so
    the barycentric second moments give a deterministic exact cell integral.
    """

    points = _as_points_rz(points_rz)
    cells = np.asarray(triangles, dtype=np.int64)
    field = np.asarray(values, dtype=complex)
    vertices = points[cells]
    p_vertices = field[cells]
    edge_a = vertices[:, 1] - vertices[:, 0]
    edge_b = vertices[:, 2] - vertices[:, 0]
    area = 0.5 * np.abs(edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0])
    radii = vertices[:, :, 0]
    diagonal = np.sum(radii * p_vertices, axis=1)
    off_diagonal = np.sum(radii, axis=1) * np.sum(p_vertices, axis=1) - diagonal
    cell_integrals = 2.0 * math.pi * area * (diagonal / 6.0 + off_diagonal / 12.0)
    return complex(np.sum(cell_integrals))


def pml_coefficients(
    r: np.ndarray | float,
    z: np.ndarray | float,
    inner_radius_m: float,
    thickness_m: float,
    *,
    alpha: float = DEFAULT_PML_ALPHA,
    exponent: int = DEFAULT_PML_EXPONENT,
) -> dict[str, np.ndarray]:
    """Evaluate the spherical radial-stretch PML coefficients.

    The returned arrays include ``R``, ``eta``, ``s_R``, ``tilde_R``, ``s_t``,
    ``gradient_radial``, ``gradient_tangential`` and ``mass``.  The stretch is
    identity for ``R <= inner_radius_m``.  For a point beyond the configured
    outer radius the last PML stretch is continued consistently; reference
    meshes do not contain such points.
    """

    inner = float(inner_radius_m)
    thickness = float(thickness_m)
    if inner <= 0.0 or thickness <= 0.0:
        raise ValueError("PML inner radius and thickness must be positive")
    if alpha <= 0.0 or exponent < 1:
        raise ValueError("PML alpha must be positive and exponent >= 1")

    rr, zz = np.broadcast_arrays(np.asarray(r, dtype=float), np.asarray(z, dtype=float))
    radius = np.sqrt(rr * rr + zz * zz)
    raw_eta = (radius - inner) / thickness
    eta = np.clip(raw_eta, 0.0, 1.0)

    # Integral of -i*alpha*eta(r)^m from the inner interface.  The second
    # branch is only relevant outside the reference outer PML radius.
    integral = np.zeros_like(radius, dtype=complex)
    inside = raw_eta > 0.0
    integral[inside] = -1j * float(alpha) * thickness * eta[inside] ** (exponent + 1) / (exponent + 1)
    beyond = raw_eta > 1.0
    integral[beyond] += -1j * float(alpha) * (radius[beyond] - (inner + thickness))
    tilde_radius = radius.astype(complex) + integral

    s_r = np.ones_like(radius, dtype=complex)
    s_r[inside] = 1.0 - 1j * float(alpha) * eta[inside] ** exponent
    safe_radius = np.where(radius > 0.0, radius, 1.0)
    s_t = tilde_radius / safe_radius
    s_t = np.where(radius > 0.0, s_t, 1.0 + 0.0j)
    return {
        "R": radius,
        "eta": eta,
        "s_R": s_r,
        "tilde_R": tilde_radius,
        "s_t": s_t,
        "gradient_radial": s_t * s_t / s_r,
        "gradient_tangential": s_r,
        "mass": s_r * s_t * s_t,
    }


def pml_alpha_for_frequency(
    frequency_Hz: float,
    c0_m_s: float,
    thickness_m: float,
    target_attenuation_nepers: float = DEFAULT_PML_TARGET_ATTENUATION_NEPERS,
    *,
    exponent: int = DEFAULT_PML_EXPONENT,
) -> float:
    r"""Return the frequency-scaled spherical-stretch ``alpha``.

    For ``s_R = 1 - i alpha eta**m`` and ``k = omega / c``, the imaginary
    stretch integral is ``k * alpha * thickness / (m + 1)``.  Choosing

    ``alpha(f) = target * (m + 1) / (k * thickness)``

    therefore gives the theoretical outer-boundary amplitude factor
    ``exp(-target)``.  The public low-level :func:`pml_coefficients` remains
    explicit-alpha only; the reference solver calls this function when its
    ``target_nepers`` mode is selected.
    """

    frequency = float(frequency_Hz)
    c0 = float(c0_m_s)
    thickness = float(thickness_m)
    target = float(target_attenuation_nepers)
    order = int(exponent)
    if frequency <= 0.0 or c0 <= 0.0 or thickness <= 0.0:
        raise ValueError("frequency, sound speed, and PML thickness must be positive")
    if target <= 0.0 or order < 1:
        raise ValueError("PML target attenuation must be positive and exponent >= 1")
    wavenumber = 2.0 * math.pi * frequency / c0
    return float(target * (order + 1) / (wavenumber * thickness))


def far_field_evaluation_radius(
    frequency_Hz: float,
    c0_m_s: float,
    hk_radius_m: float,
    *,
    minimum_kR: float = FAR_FIELD_MIN_KR,
    strict_margin: float = 1.0e-6,
) -> float:
    """Choose an axis observation radius outside HK with ``k R >= minimum_kR``.

    The returned radius is an evaluation point for the closed HK kernel, not
    a physical 1 m microphone location.  The small strict margin keeps the
    point outside HK even when the minimum ``kR`` radius happens to coincide
    with the configured HK radius.
    """

    frequency = float(frequency_Hz)
    c0 = float(c0_m_s)
    hk_radius = float(hk_radius_m)
    min_kr = float(minimum_kR)
    margin = float(strict_margin)
    if frequency <= 0.0 or c0 <= 0.0 or hk_radius <= 0.0:
        raise ValueError("frequency, sound speed, and HK radius must be positive")
    if min_kr <= 0.0 or margin <= 0.0:
        raise ValueError("minimum_kR and strict margin must be positive")
    wavenumber = 2.0 * math.pi * frequency / c0
    return float(max(min_kr / wavenumber, hk_radius * (1.0 + margin)))


def pml_operator_coefficients(
    r: np.ndarray | float,
    z: np.ndarray | float,
    inner_radius_m: float,
    thickness_m: float,
    rho0_kg_m3: float,
    bulk_modulus_Pa: float,
    *,
    alpha: float = DEFAULT_PML_ALPHA,
    exponent: int = DEFAULT_PML_EXPONENT,
) -> dict[str, np.ndarray]:
    """Return complete weak-form PML coefficients including material factors.

    At the HK interface these are exactly the ordinary coefficients
    ``gradient_radial = gradient_tangential = 1/rho`` and
    ``mass = 1/K``.  Keeping this public makes the operator-continuity check
    independent of the solver assembly path.
    """

    rho = float(rho0_kg_m3)
    bulk = float(bulk_modulus_Pa)
    if rho <= 0.0 or bulk <= 0.0:
        raise ValueError("rho0 and bulk modulus must be positive")
    coeff = pml_coefficients(
        r,
        z,
        inner_radius_m,
        thickness_m,
        alpha=alpha,
        exponent=exponent,
    )
    return {
        **coeff,
        "operator_gradient_radial": coeff["gradient_radial"] / rho,
        "operator_gradient_tangential": coeff["gradient_tangential"] / rho,
        "operator_mass": coeff["mass"] / bulk,
    }


def validate_pml_geometry(
    points_rz: np.ndarray,
    triangles: np.ndarray,
    triangle_domain_names: Iterable[str],
    boundary_edges: Mapping[str, Iterable[tuple[int, int]]],
    inner_radius_m: float,
    thickness_m: float,
    *,
    tolerance_m: float = PML_GEOMETRY_TOLERANCE_M,
    strict: bool = True,
) -> dict[str, Any]:
    """Check PML placement using physical-domain names and mesh coordinates.

    The hard check is intentionally geometric but local: it checks triangle
    vertices/centroids and named HK/outer boundary nodes.  It is not a general
    CAD intersection detector; such a detector remains a 3B risk.  With
    ``strict=True`` any mismatch raises ``ValueError``; ``strict=False`` is
    provided for deterministic fault-report tests and diagnostics.
    """

    points = _as_points_rz(points_rz)
    cells = np.asarray(triangles, dtype=np.int64)
    names = tuple(str(name) for name in triangle_domain_names)
    if cells.ndim != 2 or cells.shape[1] != 3 or len(names) != len(cells):
        raise ValueError("PML geometry check needs one domain name per triangular cell")
    inner = float(inner_radius_m)
    thickness = float(thickness_m)
    outer = inner + thickness
    tolerance = float(tolerance_m)
    if inner <= 0.0 or thickness <= 0.0 or tolerance < 0.0:
        raise ValueError("PML geometry radii and tolerance are invalid")

    node_radius = np.sqrt(np.sum(points * points, axis=1))
    pml_mask = np.asarray([name in PML_DOMAINS for name in names], dtype=bool)
    non_pml_mask = ~pml_mask
    failures: list[str] = []
    if not np.any(pml_mask):
        failures.append("missing_pml_triangles")

    def cell_radius_range(mask: np.ndarray) -> tuple[float | None, float | None]:
        if not np.any(mask):
            return None, None
        vertex_values = node_radius[cells[mask].reshape(-1)]
        centroid = np.mean(points[cells[mask]], axis=1)
        centroid_values = np.sqrt(np.sum(centroid * centroid, axis=1))
        return float(min(np.min(vertex_values), np.min(centroid_values))), float(
            max(np.max(vertex_values), np.max(centroid_values))
        )

    pml_min, pml_max = cell_radius_range(pml_mask)
    non_pml_min, non_pml_max = cell_radius_range(non_pml_mask)
    if pml_min is None or pml_min < inner - tolerance:
        failures.append("pml_triangle_below_inner")
    if pml_max is None or pml_max > outer + tolerance:
        failures.append("pml_triangle_above_outer")
    if non_pml_max is not None and non_pml_max > inner + tolerance:
        failures.append("non_pml_domain_crosses_inner")

    def boundary_node_radii(name: str) -> np.ndarray:
        edges = tuple(boundary_edges.get(name, ()))
        if not edges:
            return np.empty(0, dtype=float)
        node_ids = sorted({int(node) for edge in edges for node in edge})
        return node_radius[np.asarray(node_ids, dtype=np.int64)]

    hk_ranges: dict[str, list[float | None]] = {}
    for name in HK_BOUNDARIES:
        values = boundary_node_radii(name)
        if not len(values):
            failures.append(f"missing_{name}_nodes")
            hk_ranges[name] = [None, None]
        else:
            hk_ranges[name] = [float(np.min(values)), float(np.max(values))]
            if np.max(np.abs(values - inner)) > tolerance:
                failures.append(f"{name}_not_on_inner_radius")
    outer_values = boundary_node_radii(OUTER_PML_BOUNDARY)
    if not len(outer_values):
        failures.append("missing_outer_pml_boundary_nodes")
        outer_range: list[float | None] = [None, None]
    else:
        outer_range = [float(np.min(outer_values)), float(np.max(outer_values))]
        if np.max(np.abs(outer_values - outer)) > tolerance:
            failures.append("outer_pml_boundary_not_on_outer_radius")

    report = {
        "passed": not failures,
        "failures": sorted(set(failures)),
        "inner_radius_m": inner,
        "outer_radius_m": outer,
        "tolerance_m": tolerance,
        "pml_domain_names": sorted(PML_DOMAINS),
        "pml_triangle_count": int(np.count_nonzero(pml_mask)),
        "non_pml_triangle_count": int(np.count_nonzero(non_pml_mask)),
        "pml_triangle_R_range_m": [pml_min, pml_max],
        "pml_triangle_vertex_centroid_R_range_m": [pml_min, pml_max],
        "non_pml_triangle_R_range_m": [non_pml_min, non_pml_max],
        "hk_node_R_ranges_m": hk_ranges,
        "outer_node_R_range_m": outer_range,
        "non_pml_crosses_inner": bool(
            non_pml_max is not None and non_pml_max > inner + tolerance
        ),
    }
    if failures and strict:
        raise ValueError("PML geometry contract failed: " + ", ".join(report["failures"]))
    return report


def validate_closed_hk_geometry(
    points_rz: np.ndarray,
    facet_edges: Mapping[str, Iterable[tuple[int, int]]],
    inner_radius_m: float,
    *,
    tolerance_m: float = PML_GEOMETRY_TOLERANCE_M,
    strict: bool = True,
) -> dict[str, Any]:
    """Validate that ``hk_front`` plus ``hk_rear`` is a complete sphere.

    The meridian representation of a closed spherical Huygens--Kirchhoff
    surface is a connected polyline from the positive axis to the negative
    axis.  This check deliberately stays at the named-mesh level: it verifies
    both named halves, radial node placement, endpoint/degree topology, and
    angular coverage, without attempting a general CAD intersection test.
    """

    points = _as_points_rz(points_rz)
    radius = float(inner_radius_m)
    tolerance = float(tolerance_m)
    if radius <= 0.0 or tolerance < 0.0:
        raise ValueError("HK radius must be positive and tolerance non-negative")
    failures: list[str] = []
    front_edges = tuple(_edge_key(int(a), int(b)) for a, b in facet_edges.get("hk_front", ()))
    rear_edges = tuple(_edge_key(int(a), int(b)) for a, b in facet_edges.get("hk_rear", ()))
    if not front_edges:
        failures.append("missing_hk_front_edges")
    if not rear_edges:
        failures.append("missing_hk_rear_edges")
    overlap = set(front_edges) & set(rear_edges)
    if overlap:
        failures.append("duplicate_hk_front_rear_edges")
    all_edges = tuple(sorted(set(front_edges) | set(rear_edges)))
    nodes = sorted({node for edge in all_edges for node in edge})
    if not nodes:
        failures.append("missing_hk_nodes")

    node_radius = np.sqrt(np.sum(points * points, axis=1)) if nodes else np.empty(0)
    if nodes:
        selected_r = node_radius[np.asarray(nodes, dtype=np.int64)]
        if np.max(np.abs(selected_r - radius)) > tolerance:
            failures.append("hk_nodes_not_on_inner_radius")
    else:
        selected_r = np.empty(0)

    degrees: dict[int, int] = defaultdict(int)
    graph: dict[int, set[int]] = {node: set() for node in nodes}
    for first, second in all_edges:
        degrees[first] += 1
        degrees[second] += 1
        graph[first].add(second)
        graph[second].add(first)
    endpoints = sorted(node for node, degree in degrees.items() if degree == 1)
    if len(endpoints) != 2:
        failures.append("hk_meridian_endpoint_count")
    if any(degree != 2 for degree in degrees.values() if degree != 1):
        failures.append("hk_meridian_branch_or_duplicate")
    if nodes:
        seen = {nodes[0]}
        queue: deque[int] = deque([nodes[0]])
        while queue:
            current = queue.popleft()
            for neighbor in sorted(graph[current]):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        if len(seen) != len(nodes):
            failures.append("hk_meridian_disconnected")

    angles_by_name: dict[str, list[float | None]] = {}
    for name, edges in (("hk_front", front_edges), ("hk_rear", rear_edges)):
        named_nodes = sorted({node for edge in edges for node in edge})
        if not named_nodes:
            angles_by_name[name] = [None, None]
            continue
        coords = points[np.asarray(named_nodes, dtype=np.int64)]
        angles = np.arctan2(coords[:, 0], coords[:, 1])
        angles = np.clip(angles, 0.0, math.pi)
        angles_by_name[name] = [float(np.min(angles)), float(np.max(angles))]
    angle_tolerance = max(1.0e-8, tolerance / radius)
    front_range = angles_by_name["hk_front"]
    rear_range = angles_by_name["hk_rear"]
    if front_range[0] is None or front_range[0] > angle_tolerance:
        failures.append("hk_front_does_not_reach_positive_axis")
    if front_range[1] is None or front_range[1] < math.pi / 2.0 - angle_tolerance:
        failures.append("hk_front_does_not_reach_equator")
    if rear_range[0] is None or rear_range[0] > math.pi / 2.0 + angle_tolerance:
        failures.append("hk_rear_does_not_start_at_equator")
    if rear_range[1] is None or rear_range[1] < math.pi - angle_tolerance:
        failures.append("hk_rear_does_not_reach_negative_axis")
    if front_range[1] is not None and front_range[1] > math.pi / 2.0 + angle_tolerance:
        failures.append("hk_front_crosses_equator")
    if rear_range[0] is not None and rear_range[0] < math.pi / 2.0 - angle_tolerance:
        failures.append("hk_rear_crosses_equator")
    if endpoints:
        endpoint_coords = points[np.asarray(endpoints, dtype=np.int64)]
        endpoint_r = endpoint_coords[:, 0]
        endpoint_z = endpoint_coords[:, 1]
        if np.max(endpoint_r) > tolerance:
            failures.append("hk_endpoints_not_on_axis")
        if not (np.any(endpoint_z > 0.0) and np.any(endpoint_z < 0.0)):
            failures.append("hk_endpoints_do_not_span_both_axes")

    report = {
        "passed": not failures,
        "failures": sorted(set(failures)),
        "surface": "complete closed spherical HK after rotation",
        "mirror": False,
        "inner_radius_m": radius,
        "tolerance_m": tolerance,
        "front_edge_count": int(len(front_edges)),
        "rear_edge_count": int(len(rear_edges)),
        "combined_edge_count": int(len(all_edges)),
        "unique_node_count": int(len(nodes)),
        "endpoint_count": int(len(endpoints)),
        "endpoint_nodes": [int(node) for node in endpoints],
        "front_theta_range_rad": angles_by_name["hk_front"],
        "rear_theta_range_rad": angles_by_name["hk_rear"],
        "angular_coverage_rad": [0.0, math.pi],
        "node_R_range_m": (
            [float(np.min(selected_r)), float(np.max(selected_r))]
            if len(selected_r)
            else [None, None]
        ),
    }
    if failures and strict:
        raise ValueError("closed HK geometry contract failed: " + ", ".join(report["failures"]))
    return report


@dataclass(frozen=True)
class AcousticPhysicalParameters:
    """Validated material, drive, and frequency-scaled PML parameters."""

    rho0_kg_m3: float
    c0_m_s: float
    bulk_modulus_Pa: float
    pml_inner_radius_m: float
    pml_thickness_m: float
    reference_velocity_m_s: float = DEFAULT_REFERENCE_VELOCITY_M_S
    pml_mode: str = DEFAULT_PML_MODE
    pml_target_attenuation_nepers: float = DEFAULT_PML_TARGET_ATTENUATION_NEPERS
    pml_alpha: float | None = None
    pml_exponent: int = DEFAULT_PML_EXPONENT
    thermoviscous_loss_scale: float = 0.0
    viscous_loss_scale: float = 1.0
    thermal_loss_scale: float = 1.0
    pressure_element_order: int = 1


@dataclass
class ReferencePressureMesh:
    """Pressure-only P1 mesh data and deterministic physical trace maps."""

    path: Path
    source_sha256: str
    case_id: str
    config_path: Path
    config: EnclosureConfig
    audit_report: dict[str, Any]
    meshio_mesh: meshio.Mesh
    mesh: MeshTri
    points_rz: np.ndarray
    triangle_domain_names: tuple[str, ...]
    triangle_domain_tags: np.ndarray
    original_point_indices: np.ndarray
    line_facets: dict[str, np.ndarray]
    facet_edges: dict[str, tuple[tuple[int, int], ...]]
    facet_normals_rz: dict[str, np.ndarray]
    component_by_triangle: np.ndarray
    component_triangles: dict[int, np.ndarray]
    component_dofs: dict[int, np.ndarray]
    component_domains: dict[int, tuple[str, ...]]
    cavity_component: int
    exterior_components: tuple[int, ...]
    cavity_triangle_indices: np.ndarray
    pml_triangle_indices: np.ndarray
    non_pml_triangle_indices: np.ndarray
    outer_dirichlet_dofs: np.ndarray
    cavity_volume_m3: float
    pml_geometry_report: dict[str, Any]
    hk_geometry_report: dict[str, Any]
    trace_metrics: dict[str, dict[str, Any]]
    cavity_bli_facets: np.ndarray
    cavity_bli_group_facets: dict[str, np.ndarray]
    cavity_bli_boundary_report: dict[str, Any]
    pressure_uniform_refinement_levels: int = 0
    pressure_boundary_local_refinement_levels: int = 0

    @property
    def pressure_dof_count(self) -> int:
        return int(self.mesh.p.shape[1])

    @property
    def pressure_triangle_count(self) -> int:
        return int(self.mesh.t.shape[1])

    @property
    def component_count(self) -> int:
        return int(len(self.component_triangles))


@dataclass
class AcousticAssemblyResult:
    """Deterministic output of one frequency-domain matrix assembly."""

    frequency_Hz: float
    omega_rad_s: float
    matrix: csr_matrix
    pml_matrix: csr_matrix
    bli_matrix: csr_matrix
    bli_viscous_matrix: csr_matrix
    bli_thermal_matrix: csr_matrix
    bli_tangential_gradient_matrix: csr_matrix
    bli_boundary_mass_matrix: csr_matrix
    rhs: np.ndarray
    rhs_front: np.ndarray
    rhs_back: np.ndarray
    dirichlet_dofs: np.ndarray
    free_dofs: np.ndarray
    dof_component: np.ndarray
    component_dofs: dict[int, np.ndarray]
    pml_diagnostics: dict[str, Any]
    thermoviscous_diagnostics: dict[str, Any]
    matrix_symmetry_error: float


@dataclass
class SealedBAnalyticLimit:
    """Analytic uniform-cavity result under the same ``exp(+i omega t)`` sign."""

    frequency_Hz: float
    cavity_volume_m3: float
    rho0_kg_m3: float
    c0_m_s: float
    compliance_m3_Pa: float
    impedance_Pa_s_m3: complex
    q_into_cavity_m3_s: complex
    mean_pressure_Pa: complex

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-safe analytic B diagnostics."""

        return {
            "frequency_Hz": float(self.frequency_Hz),
            "cavity_volume_m3": float(self.cavity_volume_m3),
            "rho0_kg_m3": float(self.rho0_kg_m3),
            "c0_m_s": float(self.c0_m_s),
            "compliance_m3_Pa": float(self.compliance_m3_Pa),
            "impedance_Pa_s_m3": {
                "real": float(np.real(self.impedance_Pa_s_m3)),
                "imag": float(np.imag(self.impedance_Pa_s_m3)),
            },
            "q_into_cavity_m3_s": {
                "real": float(np.real(self.q_into_cavity_m3_s)),
                "imag": float(np.imag(self.q_into_cavity_m3_s)),
            },
            "mean_pressure_Pa": {
                "real": float(np.real(self.mean_pressure_Pa)),
                "imag": float(np.imag(self.mean_pressure_Pa)),
            },
        }


@dataclass
class AcousticSolveResult:
    """One prescribed-velocity reference solution and its audit metrics."""

    frequency_Hz: float
    pressure: np.ndarray
    assembly: AcousticAssemblyResult
    mesh: ReferencePressureMesh
    parameters: AcousticPhysicalParameters
    component_mean_pressure_Pa: dict[int, complex]
    cavity_mean_pressure_Pa: complex
    front_back_traces: dict[str, dict[str, Any]]
    q_into_cavity_m3_s: complex
    z_box_Pa_s_m3: complex
    analytic_limit: SealedBAnalyticLimit | None
    relative_impedance_error: float | None
    residual_absolute: float
    residual_relative: float
    cavity_real_impedance_ratio: float | None
    q_out_total_m3_s: float
    q_into_total_m3_s: float
    q_balance_relative_error: float
    drive_power_into_fluid_W: dict[str, float]
    input_power_from_rhs_W: float
    input_power_boundary_cross_error_W: float
    hk_diagnostics: dict[str, Any]
    pml_diagnostics: dict[str, Any]
    thermoviscous_diagnostics: dict[str, Any]

    @property
    def matrix(self) -> csr_matrix:
        return self.assembly.matrix

    @property
    def dof_component(self) -> np.ndarray:
        return self.assembly.dof_component

    @property
    def component_dofs(self) -> dict[int, np.ndarray]:
        return self.assembly.component_dofs

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-safe scalar diagnostics without serializing sparse data."""

        return {
            "frequency_Hz": float(self.frequency_Hz),
            "case_id": self.mesh.case_id,
            "reference_identity": REFERENCE_PLANAR_PISTON_IDENTITY,
            "final_production_interface_ready": False,
            "pressure_dof_count": int(len(self.pressure)),
            "pressure_triangle_count": int(self.mesh.pressure_triangle_count),
            "pressure_element_order": int(self.parameters.pressure_element_order),
            "pressure_uniform_refinement_levels": int(
                self.mesh.pressure_uniform_refinement_levels
            ),
            "pressure_boundary_local_refinement_levels": int(
                self.mesh.pressure_boundary_local_refinement_levels
            ),
            "cavity_volume_m3": float(self.mesh.cavity_volume_m3),
            "cavity_mean_pressure_Pa": {
                "real": float(self.cavity_mean_pressure_Pa.real),
                "imag": float(self.cavity_mean_pressure_Pa.imag),
            },
            "q_into_cavity_m3_s": {
                "real": float(np.real(self.q_into_cavity_m3_s)),
                "imag": float(np.imag(self.q_into_cavity_m3_s)),
            },
            "z_box_Pa_s_m3": {
                "real": float(np.real(self.z_box_Pa_s_m3)),
                "imag": float(np.imag(self.z_box_Pa_s_m3)),
            },
            "analytic_z_box_Pa_s_m3": {
                "real": float(np.real(self.analytic_limit.impedance_Pa_s_m3)),
                "imag": float(np.imag(self.analytic_limit.impedance_Pa_s_m3)),
            } if self.analytic_limit is not None else "N/A",
            "analytic_limit": self.analytic_limit.as_dict() if self.analytic_limit is not None else "N/A",
            "relative_impedance_error": (
                float(self.relative_impedance_error)
                if self.relative_impedance_error is not None
                else "N/A"
            ),
            "residual_absolute": float(self.residual_absolute),
            "residual_relative": float(self.residual_relative),
            "cavity_real_impedance_ratio": (
                float(self.cavity_real_impedance_ratio)
                if self.cavity_real_impedance_ratio is not None
                else "N/A"
            ),
            "q_out_total_m3_s": float(self.q_out_total_m3_s),
            "q_into_total_m3_s": float(self.q_into_total_m3_s),
            "q_balance_relative_error": float(self.q_balance_relative_error),
            "drive_power_into_fluid_W": self.drive_power_into_fluid_W,
            "input_power_from_rhs_W": float(self.input_power_from_rhs_W),
            "input_power_boundary_cross_error_W": float(
                self.input_power_boundary_cross_error_W
            ),
            "front_back_traces": self.front_back_traces,
            "hk_diagnostics": self.hk_diagnostics,
            "pml_diagnostics": self.pml_diagnostics,
            "thermoviscous_diagnostics": self.thermoviscous_diagnostics,
        }


def _edge_key(first: int, second: int) -> tuple[int, int]:
    return (first, second) if first < second else (second, first)


def _meshio_cells(mesh: meshio.Mesh, cell_type: str, width: int) -> tuple[np.ndarray, np.ndarray]:
    cells = np.asarray(mesh.cells_dict.get(cell_type, np.empty((0, width), dtype=np.int64)), dtype=np.int64)
    if cells.size == 0:
        cells = np.empty((0, width), dtype=np.int64)
    if cells.ndim != 2 or cells.shape[1] != width:
        raise ValueError(f"mesh {cell_type} cells must have shape (n, {width})")
    try:
        tags = np.asarray(mesh.cell_data_dict["gmsh:physical"][cell_type], dtype=np.int64).reshape(-1)
    except KeyError as exc:
        raise ValueError(f"mesh has no gmsh:physical tags for {cell_type}") from exc
    if len(tags) != len(cells):
        raise ValueError(f"mesh {cell_type} cell/tag count mismatch")
    return cells, tags


def _field_data_map(mesh: meshio.Mesh) -> dict[str, tuple[int, int]]:
    result: dict[str, tuple[int, int]] = {}
    for name, raw in mesh.field_data.items():
        values = np.asarray(raw, dtype=np.int64).reshape(-1)
        if len(values) < 2:
            raise ValueError(f"malformed physical field_data for {name!r}")
        result[str(name)] = (int(values[0]), int(values[1]))
    return result


def _pressure_triangles(
    mesh: meshio.Mesh,
    triangles: np.ndarray,
    triangle_tags: np.ndarray,
    field_data: Mapping[str, tuple[int, int]],
    case_id: str,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], dict[str, int]]:
    expected = expected_domain_names(case_id)
    pressure_tags = {
        int(tag): name
        for name, (tag, dimension) in field_data.items()
        if dimension == 2 and name.startswith("air_")
    }
    expected_pressure = {
        name: int(DOMAIN_PHYSICAL_TAGS[name])
        for name in expected
        if name.startswith("air_")
    }
    if pressure_tags != {tag: name for name, tag in expected_pressure.items()}:
        raise ValueError("pressure physical tags do not match the exact audited contract")
    selected = np.asarray([int(tag) in pressure_tags for tag in triangle_tags], dtype=bool)
    if not np.any(selected):
        raise ValueError("reference mesh contains no air_* pressure triangles")
    pressure_triangles = triangles[selected]
    pressure_tags_selected = triangle_tags[selected]
    names = tuple(pressure_tags[int(tag)] for tag in pressure_tags_selected)
    return pressure_triangles, pressure_tags_selected, names, expected_pressure


def _pressure_edge_adjacency(mesh: MeshTri) -> dict[tuple[int, int], list[int]]:
    result: dict[tuple[int, int], list[int]] = defaultdict(list)
    for triangle_index, triangle in enumerate(mesh.t.T):
        a, b, c = (int(value) for value in triangle)
        for first, second in ((a, b), (b, c), (c, a)):
            result[_edge_key(first, second)].append(int(triangle_index))
    return dict(result)


def _facet_adjacency(mesh: MeshTri) -> dict[int, list[int]]:
    edge_to_triangles = _pressure_edge_adjacency(mesh)
    result: dict[int, list[int]] = {}
    for facet_index, edge in enumerate(mesh.facets.T):
        result[int(facet_index)] = edge_to_triangles[_edge_key(int(edge[0]), int(edge[1]))]
    return result


def _facet_outward_normal(mesh: MeshTri, facet_index: int, adjacent: list[int]) -> np.ndarray:
    if len(adjacent) != 1:
        raise ValueError(f"trace facet {facet_index} has {len(adjacent)} pressure neighbors")
    edge = mesh.facets[:, int(facet_index)]
    triangle = mesh.t[:, int(adjacent[0])]
    edge_nodes = {int(edge[0]), int(edge[1])}
    opposite = next(int(node) for node in triangle if int(node) not in edge_nodes)
    p0 = np.asarray(mesh.p[:, int(edge[0])], dtype=float)
    p1 = np.asarray(mesh.p[:, int(edge[1])], dtype=float)
    po = np.asarray(mesh.p[:, opposite], dtype=float)
    tangent = p1 - p0
    length = float(np.linalg.norm(tangent))
    if length <= 0.0:
        raise ValueError(f"trace facet {facet_index} is degenerate")
    candidate = np.asarray([tangent[1], -tangent[0]], dtype=float) / length
    midpoint = 0.5 * (p0 + p1)
    interior = po - midpoint
    if float(np.dot(candidate, interior)) > 0.0:
        candidate = -candidate
    return candidate


def _connected_components(
    mesh: MeshTri,
    triangle_domain_names: tuple[str, ...],
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, np.ndarray], dict[int, tuple[str, ...]]]:
    edge_to_triangles = _pressure_edge_adjacency(mesh)
    adjacency: dict[int, set[int]] = {index: set() for index in range(mesh.t.shape[1])}
    for indices in edge_to_triangles.values():
        if len(indices) == 2:
            left, right = indices
            adjacency[left].add(right)
            adjacency[right].add(left)
    remaining = set(adjacency)
    component_by_triangle = np.full(mesh.t.shape[1], -1, dtype=np.int64)
    component_triangles: dict[int, np.ndarray] = {}
    component_dofs: dict[int, np.ndarray] = {}
    component_domains: dict[int, tuple[str, ...]] = {}
    component_id = 0
    while remaining:
        start = min(remaining)
        queue: deque[int] = deque([start])
        remaining.remove(start)
        members: list[int] = []
        while queue:
            current = queue.popleft()
            members.append(current)
            component_by_triangle[current] = component_id
            for neighbor in sorted(adjacency[current]):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
        triangles = np.asarray(sorted(members), dtype=np.int64)
        dofs = np.unique(mesh.t[:, triangles].reshape(-1)).astype(np.int64)
        domains = tuple(sorted(set(triangle_domain_names[index] for index in triangles)))
        component_triangles[component_id] = triangles
        component_dofs[component_id] = dofs
        component_domains[component_id] = domains
        component_id += 1
    return component_by_triangle, component_triangles, component_dofs, component_domains


def _trace_map(
    meshio_mesh: meshio.Mesh,
    mesh: MeshTri,
    original_point_indices: np.ndarray,
    field_data: Mapping[str, tuple[int, int]],
    edge_to_facet: Mapping[tuple[int, int], int],
    facet_adjacency: Mapping[int, list[int]],
    line_name: str,
    velocity_m_s: float,
    *,
    require_boundary: bool = True,
) -> tuple[np.ndarray, tuple[tuple[int, int], ...], np.ndarray, dict[str, Any]]:
    if line_name not in field_data:
        raise ValueError(f"required pressure trace group {line_name!r} is missing")
    lines, line_tags = _meshio_cells(meshio_mesh, "line", 2)
    line_tag = field_data[line_name][0]
    inverse_points = {int(old): int(new) for new, old in enumerate(original_point_indices)}
    facets: set[int] = set()
    edges: set[tuple[int, int]] = set()
    for line, tag in zip(lines, line_tags):
        if int(tag) != int(line_tag):
            continue
        old_first, old_second = (int(line[0]), int(line[1]))
        if old_first not in inverse_points or old_second not in inverse_points:
            continue
        edge = _edge_key(inverse_points[old_first], inverse_points[old_second])
        facet = edge_to_facet.get(edge)
        if facet is None:
            continue
        facets.add(int(facet))
        edges.add(edge)
    if not facets:
        raise ValueError(f"pressure trace group {line_name!r} has no pressure facets")

    normals: list[np.ndarray] = []
    q_out = 0.0
    trace_rows: list[dict[str, Any]] = []
    for facet in sorted(facets):
        edge = mesh.facets[:, facet]
        p0 = mesh.p[:, int(edge[0])]
        p1 = mesh.p[:, int(edge[1])]
        length = float(np.linalg.norm(p1 - p0))
        r_mid = float(0.5 * (p0[0] + p1[0]))
        if require_boundary:
            normal = _facet_outward_normal(mesh, facet, facet_adjacency[facet])
            contribution = 2.0 * math.pi * r_mid * length * float(velocity_m_s) * float(normal[1])
        else:
            normal = np.asarray([np.nan, np.nan], dtype=float)
            contribution = 0.0
        q_out += contribution
        normals.append(normal)
        trace_rows.append(
            {
                "facet": int(facet),
                "edge": [int(edge[0]), int(edge[1])],
                "length_m": length,
                "r_mid_m": r_mid,
                "normal_r": float(normal[0]),
                "normal_z": float(normal[1]),
                "q_out_contribution_m3_s": contribution,
            }
        )
    node_set = sorted({node for edge in edges for node in edge})
    normals_array = np.asarray(normals, dtype=float)
    return (
        np.asarray(sorted(facets), dtype=np.int64),
        tuple(sorted(edges)),
        normals_array,
        {
            "name": line_name,
            "identity": REFERENCE_PLANAR_PISTON_IDENTITY,
            "facet_count": int(len(facets)),
            "node_count": int(len(node_set)),
            "node_indices": node_set,
            "q_out_m3_s": float(q_out),
            "q_into_m3_s": float(-q_out),
            "velocity_z_m_s": float(velocity_m_s),
            "normal_z_range": (
                [float(np.min(normals_array[:, 1])), float(np.max(normals_array[:, 1]))]
                if require_boundary
                else [None, None]
            ),
            "segments": trace_rows,
        },
    )


def _select_cavity_bli_facets(
    meshio_mesh: meshio.Mesh,
    pressure_mesh: MeshTri,
    original_point_indices: np.ndarray,
    triangle_domain_names: tuple[str, ...],
    field_data: Mapping[str, tuple[int, int]],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    """Select C BLI facets from exact full-mesh edge/domain adjacency.

    A physical name is not sufficient because a cabinet physical group can
    contain separate cavity- and exterior-facing segments.  Each selected
    pressure facet must instead have exactly one pressure neighbor and that
    neighbor must be ``air_cavity``; its full-mesh geometric edge must have
    exactly one physical line owner and no second pressure-domain neighbor.
    """

    lines, line_tags = _meshio_cells(meshio_mesh, "line", 2)
    triangles, triangle_tags = _meshio_cells(meshio_mesh, "triangle", 3)
    domain_name_by_tag = {
        int(tag): name
        for name, (tag, dimension) in field_data.items()
        if int(dimension) == 2
    }
    boundary_name_by_tag = {
        int(tag): name
        for name, (tag, dimension) in field_data.items()
        if int(dimension) == 1
    }
    full_edge_triangles: dict[tuple[int, int], list[int]] = defaultdict(list)
    for triangle_index, triangle in enumerate(triangles):
        a, b, c = (int(value) for value in triangle)
        for first, second in ((a, b), (b, c), (c, a)):
            full_edge_triangles[_edge_key(first, second)].append(int(triangle_index))
    full_edge_owners: dict[tuple[int, int], list[tuple[str, int]]] = defaultdict(list)
    for line, raw_tag in zip(lines, line_tags, strict=True):
        tag = int(raw_tag)
        if tag not in boundary_name_by_tag:
            raise ValueError(f"line edge has unknown physical tag {tag}")
        name = boundary_name_by_tag[tag]
        if int(BOUNDARY_PHYSICAL_TAGS.get(name, -1)) != tag:
            raise ValueError(f"line group {name!r} does not own its exact physical tag")
        full_edge_owners[_edge_key(int(line[0]), int(line[1]))].append((name, tag))

    pressure_adjacency = _facet_adjacency(pressure_mesh)
    selected_by_group: dict[str, list[int]] = defaultdict(list)
    group_rows: dict[str, dict[str, Any]] = {}
    excluded = {
        "axis": {"facet_count": 0, "axisymmetric_area_m2": 0.0},
        "pressure_pressure": {"facet_count": 0, "axisymmetric_area_m2": 0.0},
        "non_cavity": {"facet_count": 0, "axisymmetric_area_m2": 0.0},
    }
    forbidden_groups = {
        OUTER_PML_BOUNDARY,
        *HK_BOUNDARIES,
        "pml_mid_interface",
        "freefield_front_side_interface",
        "freefield_rear_side_interface",
        REFERENCE_PLANAR_PISTON_FRONT,
    }
    candidate_cavity_facets = 0
    selected_facets: list[int] = []
    for facet, adjacent_pressure in sorted(pressure_adjacency.items()):
        edge = pressure_mesh.facets[:, int(facet)]
        p0 = pressure_mesh.p[:, int(edge[0])]
        p1 = pressure_mesh.p[:, int(edge[1])]
        length = float(np.linalg.norm(p1 - p0))
        area = 2.0 * math.pi * float(0.5 * (p0[0] + p1[0])) * length
        if len(adjacent_pressure) != 1:
            continue
        pressure_domain = triangle_domain_names[int(adjacent_pressure[0])]
        if pressure_domain != "air_cavity":
            excluded["non_cavity"]["facet_count"] += 1
            excluded["non_cavity"]["axisymmetric_area_m2"] += area
            continue
        candidate_cavity_facets += 1
        old_edge = _edge_key(
            int(original_point_indices[int(edge[0])]),
            int(original_point_indices[int(edge[1])]),
        )
        owners = full_edge_owners.get(old_edge, [])
        if len(owners) != 1:
            raise ValueError(
                f"cavity BLI edge {old_edge} must have exactly one line physical owner; "
                f"found {owners}"
            )
        owner, physical_tag = owners[0]
        adjacent_full = full_edge_triangles.get(old_edge, [])
        adjacent_domains = tuple(
            sorted(
                {
                    domain_name_by_tag.get(int(triangle_tags[index]), f"tag_{int(triangle_tags[index])}")
                    for index in adjacent_full
                }
            )
        )
        if "air_cavity" not in adjacent_domains:
            raise ValueError(
                f"pressure cavity facet {facet} owner {owner!r} has no full-mesh air_cavity adjacency"
            )
        other_pressure = [
            name for name in adjacent_domains if name.startswith("air_") and name != "air_cavity"
        ]
        if other_pressure:
            excluded["pressure_pressure"]["facet_count"] += 1
            excluded["pressure_pressure"]["axisymmetric_area_m2"] += area
            continue
        if owner == "axis":
            excluded["axis"]["facet_count"] += 1
            excluded["axis"]["axisymmetric_area_m2"] += area
            continue
        if owner in forbidden_groups:
            raise ValueError(
                f"forbidden exterior/interface group {owner!r} unexpectedly borders air_cavity"
            )
        unexpected_domains = [
            name
            for name in adjacent_domains
            if name != "air_cavity" and not name.startswith("rigid_")
        ]
        if unexpected_domains:
            raise ValueError(
                f"cavity BLI facet {facet} has non-rigid ownership {unexpected_domains}"
            )
        selected_facets.append(int(facet))
        selected_by_group[owner].append(int(facet))
        row = group_rows.setdefault(
            owner,
            {
                "physical_tag": physical_tag,
                "physical_dimension": 1,
                "facet_count": 0,
                "meridian_length_m": 0.0,
                "axisymmetric_area_m2": 0.0,
                "adjacent_domain_sets": set(),
                "wall_condition": "isothermal_no_slip_BLI",
            },
        )
        row["facet_count"] += 1
        row["meridian_length_m"] += length
        row["axisymmetric_area_m2"] += area
        row["adjacent_domain_sets"].add(adjacent_domains)

    if not selected_facets:
        raise ValueError("sealed C cavity has no automatically selected BLI wall facets")
    if "reference_planar_piston_back" not in selected_by_group:
        raise ValueError("sealed C BLI selection must include reference planar piston back")
    if len(selected_facets) != len(set(selected_facets)):
        raise ValueError("sealed C BLI selection contains duplicate pressure facets")
    selected_set = set(selected_facets)
    for name, facets in selected_by_group.items():
        if len(facets) != len(set(facets)) or not set(facets) <= selected_set:
            raise ValueError(f"BLI group {name!r} has duplicate or foreign facets")

    serial_groups: dict[str, dict[str, Any]] = {}
    for name, row in sorted(group_rows.items()):
        serial_groups[name] = {
            **{key: value for key, value in row.items() if key != "adjacent_domain_sets"},
            "adjacent_domain_sets": [
                list(values) for values in sorted(row["adjacent_domain_sets"])
            ],
        }
    total_area = float(
        sum(float(row["axisymmetric_area_m2"]) for row in serial_groups.values())
    )
    report = {
        "status": "pass",
        "selection_rule": (
            "unique physical line owner; exactly one pressure neighbor air_cavity; "
            "no second air_* neighbor; axis and exterior/PML/HK interfaces excluded"
        ),
        "wall_condition": "fixed/isothermal/no-slip BLI with prescribed normal piston source retained",
        "physical_groups": serial_groups,
        "selected_group_names": sorted(serial_groups),
        "selected_facet_count": int(len(selected_facets)),
        "candidate_cavity_boundary_facet_count": int(candidate_cavity_facets),
        "selected_axisymmetric_area_m2": total_area,
        "excluded": excluded,
        "duplicate_selected_facet_count": 0,
        "reference_planar_piston_back_included": True,
        "axis_excluded": True,
        "pressure_pressure_excluded": True,
        "pml_hk_exterior_excluded": True,
    }
    return (
        np.asarray(sorted(selected_facets), dtype=np.int64),
        {
            name: np.asarray(sorted(facets), dtype=np.int64)
            for name, facets in sorted(selected_by_group.items())
        },
        report,
    )


def load_reference_pressure_mesh(
    mesh_path: str | Path,
    config_path: str | Path,
    *,
    case_id: str | None = None,
    reference_velocity_m_s: float = DEFAULT_REFERENCE_VELOCITY_M_S,
) -> ReferencePressureMesh:
    """Load an audited reference A/B/C mesh and build a pressure-only P1 map.

    The case is inferred from the validated config unless an explicit case is
    supplied for a consistency check.  A is one connected pressure component
    through its rear opening; sealed B/C deliberately have two pressure
    components.  A rear opening is never converted into a pressure-release
    boundary.
    """

    path = Path(mesh_path)
    config_file = Path(config_path)
    config = load_enclosure_config(config_file)
    inferred_case = case_id_for_config(config.case)
    if inferred_case not in SUPPORTED_ACOUSTIC_CASES:
        raise ValueError(
            f"reference open/sealed/thermoviscous core supports cases A/B/C, not config {config.case!r}"
        )
    requested_case = inferred_case if case_id is None else str(case_id).upper()
    if requested_case != inferred_case:
        raise ValueError(
            f"case mismatch: config case {config.case!r} maps to {inferred_case}; "
            f"sealed case {inferred_case} only for this config, not requested {requested_case}"
        )
    audit = audit_mesh(path, case_id=requested_case, config_path=config_file)
    if audit["status"] != "pass":
        raise ValueError(
            "reference mesh failed exact enclosure audit: "
            + ", ".join(audit.get("failures", []))
        )
    meshio_mesh = meshio.read(path)
    field_data = _field_data_map(meshio_mesh)
    expected_names = set(expected_domain_names(requested_case))
    for name in expected_names:
        expected = (int(DOMAIN_PHYSICAL_TAGS[name]), 2)
        if field_data.get(name) != expected:
            raise ValueError(f"physical group {name!r} is not the exact pressure contract")

    triangles, triangle_tags = _meshio_cells(meshio_mesh, "triangle", 3)
    pressure_triangles_old, pressure_tags, pressure_names, _ = _pressure_triangles(
        meshio_mesh,
        triangles,
        triangle_tags,
        field_data,
        requested_case,
    )
    points_rz_old = _as_points_rz(meshio_mesh.points)
    original_point_indices = np.unique(pressure_triangles_old.reshape(-1)).astype(np.int64)
    remap = np.full(len(points_rz_old), -1, dtype=np.int64)
    remap[original_point_indices] = np.arange(len(original_point_indices), dtype=np.int64)
    pressure_triangles = remap[pressure_triangles_old]
    points_rz = points_rz_old[original_point_indices]
    if np.any(pressure_triangles < 0):
        raise ValueError("pressure triangle remapping encountered a non-pressure point")
    pressure_mesh = MeshTri(points_rz.T, pressure_triangles.T, sort_t=False, validate=True)
    if pressure_mesh.t.shape[1] != len(pressure_names):
        raise ValueError("scikit-fem pressure cell count changed during mesh construction")

    edge_to_facet = {
        _edge_key(int(edge[0]), int(edge[1])): int(index)
        for index, edge in enumerate(pressure_mesh.facets.T)
    }
    facet_adjacency = _facet_adjacency(pressure_mesh)
    line_facets: dict[str, np.ndarray] = {}
    facet_edges: dict[str, tuple[tuple[int, int], ...]] = {}
    facet_normals: dict[str, np.ndarray] = {}
    trace_metrics: dict[str, dict[str, Any]] = {}
    for name in (REFERENCE_PLANAR_PISTON_FRONT, REFERENCE_PLANAR_PISTON_BACK):
        facets, edges, normals, metrics = _trace_map(
            meshio_mesh,
            pressure_mesh,
            original_point_indices,
            field_data,
            edge_to_facet,
            facet_adjacency,
            name,
            reference_velocity_m_s,
        )
        line_facets[name] = facets
        facet_edges[name] = edges
        facet_normals[name] = normals
        trace_metrics[name] = metrics

    # HK is an internal pressure-pressure interface, so it is mapped for PML
    # diagnostics but deliberately does not receive an outward boundary normal.
    for name in HK_BOUNDARIES:
        facets, edges, normals, metrics = _trace_map(
            meshio_mesh,
            pressure_mesh,
            original_point_indices,
            field_data,
            edge_to_facet,
            facet_adjacency,
            name,
            0.0,
            require_boundary=False,
        )
        if any(len(facet_adjacency[int(facet)]) != 2 for facet in facets):
            raise ValueError(f"{name} must be an internal pressure-pressure interface")
        line_facets[name] = facets
        facet_edges[name] = edges
        facet_normals[name] = normals
        trace_metrics[name] = metrics

    front_nodes = {node for edge in facet_edges[REFERENCE_PLANAR_PISTON_FRONT] for node in edge}
    back_nodes = {node for edge in facet_edges[REFERENCE_PLANAR_PISTON_BACK] for node in edge}
    if front_nodes & back_nodes:
        raise ValueError("reference front/back pressure traces share pressure DOFs")

    outer_facets, outer_edges, outer_normals, outer_metrics = _trace_map(
        meshio_mesh,
        pressure_mesh,
        original_point_indices,
        field_data,
        edge_to_facet,
        facet_adjacency,
        OUTER_PML_BOUNDARY,
        0.0,
    )
    line_facets[OUTER_PML_BOUNDARY] = outer_facets
    facet_edges[OUTER_PML_BOUNDARY] = outer_edges
    facet_normals[OUTER_PML_BOUNDARY] = outer_normals
    _ = outer_metrics
    outer_dofs = np.unique(pressure_mesh.facets[:, outer_facets].reshape(-1)).astype(np.int64)

    pml_geometry_report = validate_pml_geometry(
        pressure_mesh.p.T,
        pressure_mesh.t.T,
        pressure_names,
        {
            name: facet_edges[name]
            for name in (*HK_BOUNDARIES, OUTER_PML_BOUNDARY)
        },
        float(config.raw["geometry"]["pml_inner_radius_m"]),
        float(config.raw["geometry"]["pml_thickness_m"]),
    )
    hk_geometry_report = validate_closed_hk_geometry(
        pressure_mesh.p.T,
        facet_edges,
        float(config.raw["geometry"]["pml_inner_radius_m"]),
    )

    component_by_triangle, component_triangles, component_dofs, component_domains = _connected_components(
        pressure_mesh,
        pressure_names,
    )
    cavity_components = [
        component
        for component, domains in component_domains.items()
        if "air_cavity" in domains
    ]
    if len(cavity_components) != 1:
        raise ValueError("reference pressure mesh must have one air_cavity component")
    cavity_component = int(cavity_components[0])
    exterior_domain_names = {
        "air_front_free",
        "air_side_free",
        "air_rear_free",
        "air_pml_front",
        "air_pml_rear",
    }
    exterior_components = tuple(
        sorted(
            component
            for component, domains in component_domains.items()
            if set(domains) & exterior_domain_names
        )
    )
    if requested_case == "A":
        if len(component_triangles) != 1 or component_domains[cavity_component] == ("air_cavity",):
            raise ValueError("open A pressure domains must be one component through rear opening")
        if not exterior_components or exterior_components != (cavity_component,):
            raise ValueError("open A pressure mesh must connect cavity and exterior in one component")
    else:
        if component_domains[cavity_component] != ("air_cavity",):
            raise ValueError("sealed B/C air_cavity is not pressure-disconnected from exterior")
        if len(exterior_components) != 1 or exterior_components[0] == cavity_component:
            raise ValueError("sealed B/C exterior pressure field must be one connected component")

    cavity_triangle_indices = np.asarray(
        [index for index, name in enumerate(pressure_names) if name == "air_cavity"],
        dtype=np.int64,
    )
    if not len(cavity_triangle_indices):
        raise ValueError("reference pressure mesh contains no air_cavity triangles")
    pml_triangle_indices = np.asarray(
        [index for index, name in enumerate(pressure_names) if name in PML_DOMAINS],
        dtype=np.int64,
    )
    non_pml_triangle_indices = np.asarray(
        [index for index, name in enumerate(pressure_names) if name not in PML_DOMAINS],
        dtype=np.int64,
    )
    cavity_volume = axisymmetric_volume(pressure_mesh.p.T, pressure_mesh.t[:, cavity_triangle_indices].T)
    target = float(config.net_volume_target_m3)
    if not math.isclose(
        cavity_volume,
        target,
        rel_tol=CAVITY_VOLUME_RELATIVE_TOLERANCE,
        abs_tol=1.0e-12,
    ):
        raise ValueError("pressure-only cavity volume fails the A/B/C volume contract")

    if requested_case in {"B", "C"}:
        if requested_case == "C" and not bool(config.raw["thermoviscous"]["enabled"]):
            raise ValueError("case C requires enabled thermoviscous configuration")
        cavity_bli_facets, cavity_bli_group_facets, cavity_bli_boundary_report = (
            _select_cavity_bli_facets(
                meshio_mesh,
                pressure_mesh,
                original_point_indices,
                tuple(pressure_names),
                field_data,
            )
        )
    else:
        cavity_bli_facets = np.empty((0,), dtype=np.int64)
        cavity_bli_group_facets = {}
        cavity_bli_boundary_report = {
            "status": "not_applicable",
            "case": requested_case,
            "selected_facet_count": 0,
            "selected_axisymmetric_area_m2": 0.0,
        }

    dof_component = np.full(pressure_mesh.p.shape[1], -1, dtype=np.int64)
    for component, dofs in component_dofs.items():
        dof_component[dofs] = int(component)
    if np.any(dof_component < 0):
        raise ValueError("pressure mesh contains an unassigned pressure DOF")

    return ReferencePressureMesh(
        path=path,
        source_sha256=sha256_file(path),
        case_id=requested_case,
        config_path=config_file,
        config=config,
        audit_report=audit,
        meshio_mesh=meshio_mesh,
        mesh=pressure_mesh,
        points_rz=points_rz,
        triangle_domain_names=tuple(pressure_names),
        triangle_domain_tags=np.asarray(pressure_tags, dtype=np.int64),
        original_point_indices=original_point_indices,
        line_facets=line_facets,
        facet_edges=facet_edges,
        facet_normals_rz=facet_normals,
        component_by_triangle=component_by_triangle,
        component_triangles=component_triangles,
        component_dofs=component_dofs,
        component_domains=component_domains,
        cavity_component=cavity_component,
        exterior_components=exterior_components,
        cavity_triangle_indices=cavity_triangle_indices,
        pml_triangle_indices=pml_triangle_indices,
        non_pml_triangle_indices=non_pml_triangle_indices,
        outer_dirichlet_dofs=outer_dofs,
        cavity_volume_m3=cavity_volume,
        pml_geometry_report=pml_geometry_report,
        hk_geometry_report=hk_geometry_report,
        trace_metrics=trace_metrics,
        cavity_bli_facets=cavity_bli_facets,
        cavity_bli_group_facets=cavity_bli_group_facets,
        cavity_bli_boundary_report=cavity_bli_boundary_report,
    )


def _refined_trace_metrics(
    mesh: MeshTri,
    facets: np.ndarray,
    name: str,
    velocity_m_s: float,
    *,
    require_boundary: bool,
) -> tuple[tuple[tuple[int, int], ...], np.ndarray, dict[str, Any]]:
    adjacency = _facet_adjacency(mesh)
    edges: list[tuple[int, int]] = []
    normals: list[np.ndarray] = []
    segments: list[dict[str, Any]] = []
    q_out = 0.0
    for facet in np.asarray(facets, dtype=np.int64):
        edge = mesh.facets[:, int(facet)]
        edge_key = _edge_key(int(edge[0]), int(edge[1]))
        p0 = mesh.p[:, int(edge[0])]
        p1 = mesh.p[:, int(edge[1])]
        length = float(np.linalg.norm(p1 - p0))
        r_mid = float(0.5 * (p0[0] + p1[0]))
        if require_boundary:
            normal = _facet_outward_normal(mesh, int(facet), adjacency[int(facet)])
            contribution = (
                2.0
                * math.pi
                * r_mid
                * length
                * float(velocity_m_s)
                * float(normal[1])
            )
        else:
            if len(adjacency[int(facet)]) != 2:
                raise ValueError(f"refined internal trace {name!r} is not pressure-pressure")
            normal = np.asarray([np.nan, np.nan], dtype=float)
            contribution = 0.0
        edges.append(edge_key)
        normals.append(normal)
        q_out += contribution
        segments.append(
            {
                "facet": int(facet),
                "edge": [int(edge_key[0]), int(edge_key[1])],
                "length_m": length,
                "r_mid_m": r_mid,
                "normal_r": float(normal[0]),
                "normal_z": float(normal[1]),
                "q_out_contribution_m3_s": contribution,
            }
        )
    normals_array = np.asarray(normals, dtype=float)
    node_indices = sorted({node for edge in edges for node in edge})
    metrics = {
        "name": name,
        "identity": REFERENCE_PLANAR_PISTON_IDENTITY,
        "facet_count": int(len(facets)),
        "node_count": int(len(node_indices)),
        "node_indices": node_indices,
        "q_out_m3_s": float(q_out),
        "q_into_m3_s": float(-q_out),
        "velocity_z_m_s": float(velocity_m_s),
        "normal_z_range": (
            [
                float(np.min(normals_array[:, 1])),
                float(np.max(normals_array[:, 1])),
            ]
            if require_boundary
            else [None, None]
        ),
        "segments": segments,
    }
    return tuple(sorted(edges)), normals_array, metrics


def uniformly_refine_reference_pressure_mesh(
    pressure_mesh: ReferencePressureMesh,
    levels: int = 1,
) -> ReferencePressureMesh:
    """Deterministically refine every pressure triangle and tagged facet.

    This is an in-memory pressure-discretization operation.  It neither edits
    nor rewrites the audited phase-2 source ``.msh``.  Every parent triangle
    produces four children with the same physical domain tag, and every named
    parent facet produces two named child facets.
    """

    count = int(levels)
    if count < 0:
        raise ValueError("pressure uniform refinement levels must be nonnegative")
    result = pressure_mesh
    for _ in range(count):
        old_mesh = result.mesh
        old_points = old_mesh.p.T
        old_triangles = old_mesh.t.T
        old_edges = sorted(
            _edge_key(int(edge[0]), int(edge[1])) for edge in old_mesh.facets.T
        )
        hk_edge_set = {
            _edge_key(int(edge[0]), int(edge[1]))
            for name in HK_BOUNDARIES
            for edge in result.facet_edges[name]
        }
        outer_edge_set = {
            _edge_key(int(edge[0]), int(edge[1]))
            for edge in result.facet_edges[OUTER_PML_BOUNDARY]
        }
        pml_inner_radius = float(
            result.config.raw["geometry"]["pml_inner_radius_m"]
        )
        pml_outer_radius = pml_inner_radius + float(
            result.config.raw["geometry"]["pml_thickness_m"]
        )
        midpoint_by_edge: dict[tuple[int, int], int] = {}
        new_points = [np.asarray(point, dtype=float) for point in old_points]
        for edge in old_edges:
            midpoint_by_edge[edge] = len(new_points)
            midpoint = 0.5 * (old_points[edge[0]] + old_points[edge[1]])
            target_radius = (
                pml_inner_radius
                if edge in hk_edge_set
                else pml_outer_radius
                if edge in outer_edge_set
                else None
            )
            if target_radius is not None:
                radius = float(np.linalg.norm(midpoint))
                if radius <= 0.0:
                    raise ValueError("cannot project a circular boundary midpoint at R=0")
                midpoint = midpoint * (target_radius / radius)
            new_points.append(midpoint)
        new_triangles: list[tuple[int, int, int]] = []
        for triangle in old_triangles:
            a, b, c = (int(value) for value in triangle)
            ab = midpoint_by_edge[_edge_key(a, b)]
            bc = midpoint_by_edge[_edge_key(b, c)]
            ca = midpoint_by_edge[_edge_key(c, a)]
            new_triangles.extend(
                ((a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca))
            )
        refined_mesh = MeshTri(
            np.asarray(new_points, dtype=float).T,
            np.asarray(new_triangles, dtype=np.int64).T,
            sort_t=False,
            validate=True,
        )
        if refined_mesh.t.shape[1] != 4 * old_mesh.t.shape[1]:
            raise ValueError("uniform refinement did not create four child triangles")
        refined_edge_to_facet = {
            _edge_key(int(edge[0]), int(edge[1])): int(index)
            for index, edge in enumerate(refined_mesh.facets.T)
        }

        def child_facets(parent_facets: np.ndarray) -> np.ndarray:
            children: list[int] = []
            for parent in np.asarray(parent_facets, dtype=np.int64):
                edge = old_mesh.facets[:, int(parent)]
                key = _edge_key(int(edge[0]), int(edge[1]))
                middle = midpoint_by_edge[key]
                for child_edge in (
                    _edge_key(key[0], middle),
                    _edge_key(middle, key[1]),
                ):
                    if child_edge not in refined_edge_to_facet:
                        raise ValueError("refined physical facet child is missing")
                    children.append(refined_edge_to_facet[child_edge])
            if len(children) != 2 * len(parent_facets) or len(children) != len(
                set(children)
            ):
                raise ValueError("refined physical facet propagation is not one-to-two")
            return np.asarray(sorted(children), dtype=np.int64)

        refined_line_facets = {
            name: child_facets(facets)
            for name, facets in result.line_facets.items()
        }
        refined_facet_edges: dict[str, tuple[tuple[int, int], ...]] = {}
        refined_normals: dict[str, np.ndarray] = {}
        refined_trace_metrics: dict[str, dict[str, Any]] = {}
        for name, facets in refined_line_facets.items():
            require_boundary = name not in HK_BOUNDARIES
            velocity = (
                float(result.trace_metrics[name]["velocity_z_m_s"])
                if name in result.trace_metrics
                else 0.0
            )
            edges, normals, metrics = _refined_trace_metrics(
                refined_mesh,
                facets,
                name,
                velocity,
                require_boundary=require_boundary,
            )
            refined_facet_edges[name] = edges
            refined_normals[name] = normals
            if name in result.trace_metrics:
                refined_trace_metrics[name] = metrics

        refined_bli_group_facets = {
            name: child_facets(facets)
            for name, facets in result.cavity_bli_group_facets.items()
        }
        refined_bli_facets = (
            np.asarray(
                sorted(
                    int(facet)
                    for facets in refined_bli_group_facets.values()
                    for facet in facets
                ),
                dtype=np.int64,
            )
            if refined_bli_group_facets
            else np.empty((0,), dtype=np.int64)
        )
        if len(refined_bli_facets) != len(set(refined_bli_facets.tolist())):
            raise ValueError("refined BLI physical groups share child facets")

        names = tuple(
            name for name in result.triangle_domain_names for _ in range(4)
        )
        tags = np.repeat(result.triangle_domain_tags, 4).astype(np.int64)
        component_by_triangle, component_triangles, component_dofs, component_domains = (
            _connected_components(refined_mesh, names)
        )
        cavity_components = [
            component
            for component, domains in component_domains.items()
            if "air_cavity" in domains
        ]
        if len(cavity_components) != 1:
            raise ValueError("refined pressure mesh lost unique cavity component")
        cavity_component = int(cavity_components[0])
        exterior_names = {
            "air_front_free",
            "air_side_free",
            "air_rear_free",
            "air_pml_front",
            "air_pml_rear",
        }
        exterior_components = tuple(
            sorted(
                component
                for component, domains in component_domains.items()
                if set(domains) & exterior_names
            )
        )
        cavity_triangles = np.asarray(
            [index for index, name in enumerate(names) if name == "air_cavity"],
            dtype=np.int64,
        )
        pml_triangles = np.asarray(
            [index for index, name in enumerate(names) if name in PML_DOMAINS],
            dtype=np.int64,
        )
        non_pml_triangles = np.asarray(
            [index for index, name in enumerate(names) if name not in PML_DOMAINS],
            dtype=np.int64,
        )
        outer_facets = refined_line_facets[OUTER_PML_BOUNDARY]
        outer_dofs = np.unique(
            refined_mesh.facets[:, outer_facets].reshape(-1)
        ).astype(np.int64)
        cavity_volume = axisymmetric_volume(
            refined_mesh.p.T,
            refined_mesh.t[:, cavity_triangles].T,
        )
        if not math.isclose(
            cavity_volume,
            result.cavity_volume_m3,
            rel_tol=2.0e-13,
            abs_tol=1.0e-14,
        ):
            raise ValueError("uniform refinement changed axisymmetric cavity volume")
        pml_report = validate_pml_geometry(
            refined_mesh.p.T,
            refined_mesh.t.T,
            names,
            {
                name: refined_facet_edges[name]
                for name in (*HK_BOUNDARIES, OUTER_PML_BOUNDARY)
            },
            float(result.config.raw["geometry"]["pml_inner_radius_m"]),
            float(result.config.raw["geometry"]["pml_thickness_m"]),
        )
        hk_report = validate_closed_hk_geometry(
            refined_mesh.p.T,
            refined_facet_edges,
            float(result.config.raw["geometry"]["pml_inner_radius_m"]),
        )
        bli_report = deepcopy(result.cavity_bli_boundary_report)
        if refined_bli_group_facets:
            for name, facets in refined_bli_group_facets.items():
                bli_report["physical_groups"][name]["facet_count"] = int(len(facets))
            bli_report["selected_facet_count"] = int(len(refined_bli_facets))
            bli_report["candidate_cavity_boundary_facet_count"] = int(
                2 * result.cavity_bli_boundary_report[
                    "candidate_cavity_boundary_facet_count"
                ]
            )
            for row in bli_report["excluded"].values():
                row["facet_count"] = int(2 * row["facet_count"])
            bli_report["pressure_uniform_refinement_levels"] = int(
                result.pressure_uniform_refinement_levels + 1
            )
            bli_report["tag_propagation"] = (
                "each parent triangle kept its domain tag; each named parent "
                "facet produced two child facets"
            )
        original_indices = np.concatenate(
            (
                result.original_point_indices,
                np.full(len(new_points) - len(old_points), -1, dtype=np.int64),
            )
        )
        result = replace(
            result,
            mesh=refined_mesh,
            points_rz=refined_mesh.p.T,
            triangle_domain_names=names,
            triangle_domain_tags=tags,
            original_point_indices=original_indices,
            line_facets=refined_line_facets,
            facet_edges=refined_facet_edges,
            facet_normals_rz=refined_normals,
            component_by_triangle=component_by_triangle,
            component_triangles=component_triangles,
            component_dofs=component_dofs,
            component_domains=component_domains,
            cavity_component=cavity_component,
            exterior_components=exterior_components,
            cavity_triangle_indices=cavity_triangles,
            pml_triangle_indices=pml_triangles,
            non_pml_triangle_indices=non_pml_triangles,
            outer_dirichlet_dofs=outer_dofs,
            cavity_volume_m3=cavity_volume,
            pml_geometry_report=pml_report,
            hk_geometry_report=hk_report,
            trace_metrics=refined_trace_metrics,
            cavity_bli_facets=refined_bli_facets,
            cavity_bli_group_facets=refined_bli_group_facets,
            cavity_bli_boundary_report=bli_report,
            pressure_uniform_refinement_levels=result.pressure_uniform_refinement_levels
            + 1,
        )
    return result


def locally_refine_cavity_wall_pressure_mesh(
    pressure_mesh: ReferencePressureMesh,
    levels: int = 1,
) -> ReferencePressureMesh:
    """Conformingly refine only triangles touching audited cavity wall facets.

    The source phase-2 mesh is not changed.  scikit-fem's deterministic
    red/green/blue adaptive refinement supplies conformity.  Parent physical
    domains are propagated through named subdomains; every named physical
    facet is then required to survive either one-to-one or split one-to-two.
    This operation is defined identically for sealed B and C so that setting
    C's loss scale to zero leaves the two discrete pressure fields identical.
    """

    count = int(levels)
    if count < 0:
        raise ValueError("pressure boundary-local refinement levels must be nonnegative")
    if count and pressure_mesh.case_id not in {"B", "C"}:
        raise ValueError("cavity-wall local refinement supports sealed B/C only")
    result = pressure_mesh
    for _ in range(count):
        old_mesh = result.mesh
        if not len(result.cavity_bli_facets):
            raise ValueError("cavity-wall local refinement has no audited wall facets")
        adjacency = _facet_adjacency(old_mesh)
        marked = np.asarray(
            sorted(
                {
                    int(adjacency[int(facet)][0])
                    for facet in result.cavity_bli_facets
                    if len(adjacency[int(facet)]) == 1
                }
            ),
            dtype=np.int64,
        )
        if len(marked) != len(result.cavity_bli_facets):
            raise ValueError("each audited cavity wall facet must have one pressure neighbor")

        subdomains = {
            name: np.asarray(
                [
                    index
                    for index, domain_name in enumerate(result.triangle_domain_names)
                    if domain_name == name
                ],
                dtype=np.int64,
            )
            for name in sorted(set(result.triangle_domain_names))
        }
        tagged_old_mesh = old_mesh.with_subdomains(subdomains)
        sorted_old_mesh = replace(
            tagged_old_mesh,
            t=tagged_old_mesh._adaptive_sort_mesh(tagged_old_mesh.p, tagged_old_mesh.t),
            sort_t=False,
        )
        split_flags = tagged_old_mesh._adaptive_find_facets(sorted_old_mesh, marked)
        split_parent_facets = np.flatnonzero(split_flags == 1).astype(np.int64)
        refined_mesh = tagged_old_mesh.refined(marked)
        midpoint_by_edge: dict[tuple[int, int], int] = {}
        old_node_count = old_mesh.p.shape[1]
        for offset, parent_facet in enumerate(split_parent_facets):
            edge = sorted_old_mesh.facets[:, int(parent_facet)]
            key = _edge_key(int(edge[0]), int(edge[1]))
            midpoint = old_node_count + int(offset)
            expected = 0.5 * (old_mesh.p[:, key[0]] + old_mesh.p[:, key[1]])
            if not np.array_equal(refined_mesh.p[:, midpoint], expected):
                raise ValueError("adaptive midpoint ordering is not deterministic")
            midpoint_by_edge[key] = midpoint
        refined_edge_to_facet = {
            _edge_key(int(edge[0]), int(edge[1])): int(index)
            for index, edge in enumerate(refined_mesh.facets.T)
        }

        def propagated_facets(parent_facets: np.ndarray) -> np.ndarray:
            children: list[int] = []
            for parent in np.asarray(parent_facets, dtype=np.int64):
                edge = old_mesh.facets[:, int(parent)]
                key = _edge_key(int(edge[0]), int(edge[1]))
                if key in midpoint_by_edge:
                    middle = midpoint_by_edge[key]
                    child_edges = (
                        _edge_key(key[0], middle),
                        _edge_key(middle, key[1]),
                    )
                else:
                    child_edges = (key,)
                for child_edge in child_edges:
                    if child_edge not in refined_edge_to_facet:
                        raise ValueError(
                            f"adaptive physical facet child {child_edge} is missing"
                        )
                    children.append(refined_edge_to_facet[child_edge])
            if len(children) != len(set(children)):
                raise ValueError("adaptive physical facet propagation duplicated a facet")
            return np.asarray(sorted(children), dtype=np.int64)

        refined_line_facets = {
            name: propagated_facets(facets)
            for name, facets in result.line_facets.items()
        }
        refined_facet_edges: dict[str, tuple[tuple[int, int], ...]] = {}
        refined_normals: dict[str, np.ndarray] = {}
        refined_trace_metrics: dict[str, dict[str, Any]] = {}
        for name, facets in refined_line_facets.items():
            require_boundary = name not in HK_BOUNDARIES
            velocity = (
                float(result.trace_metrics[name]["velocity_z_m_s"])
                if name in result.trace_metrics
                else 0.0
            )
            edges, normals, metrics = _refined_trace_metrics(
                refined_mesh,
                facets,
                name,
                velocity,
                require_boundary=require_boundary,
            )
            refined_facet_edges[name] = edges
            refined_normals[name] = normals
            if name in result.trace_metrics:
                refined_trace_metrics[name] = metrics

        refined_bli_group_facets = {
            name: propagated_facets(facets)
            for name, facets in result.cavity_bli_group_facets.items()
        }
        refined_bli_facets = np.asarray(
            sorted(
                int(facet)
                for facets in refined_bli_group_facets.values()
                for facet in facets
            ),
            dtype=np.int64,
        )
        if len(refined_bli_facets) != len(set(refined_bli_facets.tolist())):
            raise ValueError("locally refined BLI physical groups share facets")

        names_by_triangle = np.empty(refined_mesh.t.shape[1], dtype=object)
        assignment_count = np.zeros(refined_mesh.t.shape[1], dtype=np.int8)
        for name, indices in refined_mesh.subdomains.items():
            names_by_triangle[np.asarray(indices, dtype=np.int64)] = name
            assignment_count[np.asarray(indices, dtype=np.int64)] += 1
        if np.any(assignment_count != 1):
            raise ValueError("adaptive triangle physical-domain propagation is not exact")
        names = tuple(str(value) for value in names_by_triangle.tolist())
        tags = np.asarray([DOMAIN_PHYSICAL_TAGS[name] for name in names], dtype=np.int64)
        component_by_triangle, component_triangles, component_dofs, component_domains = (
            _connected_components(refined_mesh, names)
        )
        cavity_components = [
            component
            for component, domains in component_domains.items()
            if "air_cavity" in domains
        ]
        if len(cavity_components) != 1:
            raise ValueError("local refinement lost the unique cavity component")
        cavity_component = int(cavity_components[0])
        exterior_names = {
            "air_front_free",
            "air_side_free",
            "air_rear_free",
            "air_pml_front",
            "air_pml_rear",
        }
        exterior_components = tuple(
            sorted(
                component
                for component, domains in component_domains.items()
                if set(domains) & exterior_names
            )
        )
        cavity_triangles = np.asarray(
            [index for index, name in enumerate(names) if name == "air_cavity"],
            dtype=np.int64,
        )
        pml_triangles = np.asarray(
            [index for index, name in enumerate(names) if name in PML_DOMAINS],
            dtype=np.int64,
        )
        non_pml_triangles = np.asarray(
            [index for index, name in enumerate(names) if name not in PML_DOMAINS],
            dtype=np.int64,
        )
        outer_dofs = np.unique(
            refined_mesh.facets[:, refined_line_facets[OUTER_PML_BOUNDARY]].reshape(-1)
        ).astype(np.int64)
        cavity_volume = axisymmetric_volume(
            refined_mesh.p.T,
            refined_mesh.t[:, cavity_triangles].T,
        )
        if not math.isclose(
            cavity_volume,
            result.cavity_volume_m3,
            rel_tol=2.0e-13,
            abs_tol=1.0e-14,
        ):
            raise ValueError("local cavity-wall refinement changed cavity volume")
        pml_report = validate_pml_geometry(
            refined_mesh.p.T,
            refined_mesh.t.T,
            names,
            {
                name: refined_facet_edges[name]
                for name in (*HK_BOUNDARIES, OUTER_PML_BOUNDARY)
            },
            float(result.config.raw["geometry"]["pml_inner_radius_m"]),
            float(result.config.raw["geometry"]["pml_thickness_m"]),
        )
        hk_report = validate_closed_hk_geometry(
            refined_mesh.p.T,
            refined_facet_edges,
            float(result.config.raw["geometry"]["pml_inner_radius_m"]),
        )
        bli_report = deepcopy(result.cavity_bli_boundary_report)
        for name, facets in refined_bli_group_facets.items():
            bli_report["physical_groups"][name]["facet_count"] = int(len(facets))
        bli_report["selected_facet_count"] = int(len(refined_bli_facets))
        bli_report["pressure_boundary_local_refinement_levels"] = int(
            result.pressure_boundary_local_refinement_levels + 1
        )
        bli_report["tag_propagation"] = (
            "parent physical domains propagated by conforming subdomains; named "
            "facets required to survive one-to-one or split one-to-two"
        )
        original_indices = np.concatenate(
            (
                result.original_point_indices,
                np.full(
                    refined_mesh.p.shape[1] - old_node_count,
                    -1,
                    dtype=np.int64,
                ),
            )
        )
        result = replace(
            result,
            mesh=refined_mesh,
            points_rz=refined_mesh.p.T,
            triangle_domain_names=names,
            triangle_domain_tags=tags,
            original_point_indices=original_indices,
            line_facets=refined_line_facets,
            facet_edges=refined_facet_edges,
            facet_normals_rz=refined_normals,
            component_by_triangle=component_by_triangle,
            component_triangles=component_triangles,
            component_dofs=component_dofs,
            component_domains=component_domains,
            cavity_component=cavity_component,
            exterior_components=exterior_components,
            cavity_triangle_indices=cavity_triangles,
            pml_triangle_indices=pml_triangles,
            non_pml_triangle_indices=non_pml_triangles,
            outer_dirichlet_dofs=outer_dofs,
            cavity_volume_m3=cavity_volume,
            pml_geometry_report=pml_report,
            hk_geometry_report=hk_report,
            trace_metrics=refined_trace_metrics,
            cavity_bli_facets=refined_bli_facets,
            cavity_bli_group_facets=refined_bli_group_facets,
            cavity_bli_boundary_report=bli_report,
            pressure_boundary_local_refinement_levels=(
                result.pressure_boundary_local_refinement_levels + 1
            ),
        )
    return result


def sealed_b_analytic_limit(
    frequency_Hz: float,
    cavity_volume_m3: float,
    rho0_kg_m3: float,
    c0_m_s: float,
    q_into_cavity_m3_s: complex,
) -> SealedBAnalyticLimit:
    """Return ``C=V/(rho*c^2)`` and ``Z=1/(i*omega*C)`` for sealed B."""

    frequency = float(frequency_Hz)
    if frequency <= 0.0:
        raise ValueError("frequency must be positive")
    volume = float(cavity_volume_m3)
    rho = float(rho0_kg_m3)
    c0 = float(c0_m_s)
    if volume <= 0.0 or rho <= 0.0 or c0 <= 0.0:
        raise ValueError("analytic sealed-B parameters must be positive")
    omega = 2.0 * math.pi * frequency
    compliance = volume / (rho * c0 * c0)
    impedance = 1.0 / (1j * omega * compliance)
    q_into = complex(q_into_cavity_m3_s)
    return SealedBAnalyticLimit(
        frequency_Hz=frequency,
        cavity_volume_m3=volume,
        rho0_kg_m3=rho,
        c0_m_s=c0,
        compliance_m3_Pa=compliance,
        impedance_Pa_s_m3=impedance,
        q_into_cavity_m3_s=q_into,
        mean_pressure_Pa=impedance * q_into,
    )


class ReferencePrescribedVelocityAcoustics:
    """P1 axisymmetric reference solver for open A and sealed B/C."""

    def __init__(
        self,
        pressure_mesh: ReferencePressureMesh,
        *,
        reference_velocity_m_s: float = DEFAULT_REFERENCE_VELOCITY_M_S,
        pml_mode: str = DEFAULT_PML_MODE,
        target_attenuation_nepers: float = DEFAULT_PML_TARGET_ATTENUATION_NEPERS,
        pml_alpha: float | None = None,
        pml_exponent: int = DEFAULT_PML_EXPONENT,
        pml_attenuation_mode: str | None = None,
        loss_scale: float | None = None,
        viscous_loss_scale: float = 1.0,
        thermal_loss_scale: float = 1.0,
        pressure_element_order: int = 1,
    ) -> None:
        if pressure_mesh.case_id not in SUPPORTED_ACOUSTIC_CASES:
            raise ValueError("reference acoustic solver supports cases A, B, and C only")
        if pml_attenuation_mode is not None:
            if pml_mode != DEFAULT_PML_MODE and pml_mode != str(pml_attenuation_mode):
                raise ValueError("pml_mode and pml_attenuation_mode disagree")
            pml_mode = str(pml_attenuation_mode)
        mode = str(pml_mode).strip().lower()
        if mode not in SUPPORTED_PML_MODES:
            raise ValueError(f"unsupported PML mode {pml_mode!r}")
        if int(pml_exponent) < 1:
            raise ValueError("PML exponent must be >= 1")
        target = float(target_attenuation_nepers)
        if target <= 0.0:
            raise ValueError("target attenuation must be positive")
        if mode == EXPLICIT_PML_MODE:
            if pml_alpha is None:
                pml_alpha = DEFAULT_PML_ALPHA
            if float(pml_alpha) <= 0.0:
                raise ValueError("explicit_alpha mode requires positive pml_alpha")
        elif pml_alpha is not None:
            raise ValueError("fixed pml_alpha cannot be mixed with target_nepers mode")
        resolved_loss_scale = (
            (1.0 if pressure_mesh.case_id == "C" else 0.0)
            if loss_scale is None
            else float(loss_scale)
        )
        component_scales = {
            "loss_scale": resolved_loss_scale,
            "viscous_loss_scale": float(viscous_loss_scale),
            "thermal_loss_scale": float(thermal_loss_scale),
        }
        for name, value in component_scales.items():
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if pressure_mesh.case_id != "C" and resolved_loss_scale != 0.0:
            raise ValueError("thermoviscous loss_scale is available only for case C")
        element_order = int(pressure_element_order)
        if element_order not in (1, 2):
            raise ValueError("pressure_element_order must be 1 or 2")
        air = pressure_mesh.config.raw["air"]
        geometry = pressure_mesh.config.raw["geometry"]
        self.mesh_data = pressure_mesh
        self.element = ElementTriP1() if element_order == 1 else ElementTriP2()
        self.basis = Basis(pressure_mesh.mesh, self.element, intorder=6)
        self.parameters = AcousticPhysicalParameters(
            rho0_kg_m3=float(air["rho0_kg_m3"]),
            c0_m_s=float(air["c0_m_s"]),
            bulk_modulus_Pa=float(air["rho0_kg_m3"]) * float(air["c0_m_s"]) ** 2,
            pml_inner_radius_m=float(geometry["pml_inner_radius_m"]),
            pml_thickness_m=float(geometry["pml_thickness_m"]),
            reference_velocity_m_s=float(reference_velocity_m_s),
            pml_mode=mode,
            pml_target_attenuation_nepers=target,
            pml_alpha=None if pml_alpha is None else float(pml_alpha),
            pml_exponent=int(pml_exponent),
            thermoviscous_loss_scale=resolved_loss_scale,
            viscous_loss_scale=component_scales["viscous_loss_scale"],
            thermal_loss_scale=component_scales["thermal_loss_scale"],
            pressure_element_order=element_order,
        )
        self._dof_component = np.full(self.basis.N, -1, dtype=np.int64)
        self._component_dofs: dict[int, np.ndarray] = {}
        for component, triangles in pressure_mesh.component_triangles.items():
            dofs = np.asarray(
                self.basis.get_dofs(elements=triangles).all(),
                dtype=np.int64,
            )
            self._component_dofs[int(component)] = dofs
            occupied = self._dof_component[dofs]
            if np.any((occupied >= 0) & (occupied != int(component))):
                raise ValueError("pressure FE DOF is shared by disconnected components")
            self._dof_component[dofs] = int(component)
        if np.any(self._dof_component < 0):
            raise ValueError("scikit-fem pressure DOFs are not component-mapped")
        outer_dofs = np.asarray(
            self.basis.get_dofs(
                facets=pressure_mesh.line_facets[OUTER_PML_BOUNDARY]
            ).all(),
            dtype=np.int64,
        )
        self._free_dofs = np.setdiff1d(
            np.arange(self.basis.N, dtype=np.int64), outer_dofs
        )
        self._outer_dirichlet_dofs = outer_dofs

    @classmethod
    def from_files(
        cls,
        mesh_path: str | Path,
        config_path: str | Path,
        *,
        case_id: str | None = None,
        reference_velocity_m_s: float = DEFAULT_REFERENCE_VELOCITY_M_S,
        pml_mode: str = DEFAULT_PML_MODE,
        target_attenuation_nepers: float = DEFAULT_PML_TARGET_ATTENUATION_NEPERS,
        pml_alpha: float | None = None,
        pml_exponent: int = DEFAULT_PML_EXPONENT,
        pml_attenuation_mode: str | None = None,
        loss_scale: float | None = None,
        viscous_loss_scale: float = 1.0,
        thermal_loss_scale: float = 1.0,
        pressure_uniform_refinements: int = 0,
        pressure_boundary_local_refinements: int = 0,
        pressure_element_order: int = 1,
    ) -> "ReferencePrescribedVelocityAcoustics":
        mesh_data = load_reference_pressure_mesh(
            mesh_path,
            config_path,
            case_id=case_id,
            reference_velocity_m_s=reference_velocity_m_s,
        )
        mesh_data = uniformly_refine_reference_pressure_mesh(
            mesh_data,
            levels=pressure_uniform_refinements,
        )
        mesh_data = locally_refine_cavity_wall_pressure_mesh(
            mesh_data,
            levels=pressure_boundary_local_refinements,
        )
        return cls(
            mesh_data,
            reference_velocity_m_s=reference_velocity_m_s,
            pml_mode=pml_mode,
            target_attenuation_nepers=target_attenuation_nepers,
            pml_alpha=pml_alpha,
            pml_exponent=pml_exponent,
            pml_attenuation_mode=pml_attenuation_mode,
            loss_scale=loss_scale,
            viscous_loss_scale=viscous_loss_scale,
            thermal_loss_scale=thermal_loss_scale,
            pressure_element_order=pressure_element_order,
        )

    @property
    def pressure_dof_count(self) -> int:
        return int(self.basis.N)

    def _alpha_for_frequency(self, frequency_Hz: float) -> float:
        if self.parameters.pml_mode == EXPLICIT_PML_MODE:
            assert self.parameters.pml_alpha is not None
            return float(self.parameters.pml_alpha)
        return pml_alpha_for_frequency(
            frequency_Hz,
            self.parameters.c0_m_s,
            self.parameters.pml_thickness_m,
            self.parameters.pml_target_attenuation_nepers,
            exponent=self.parameters.pml_exponent,
        )

    def _ordinary_form(self, omega: float):
        rho = self.parameters.rho0_kg_m3
        bulk = self.parameters.bulk_modulus_Pa

        @BilinearForm(dtype=np.complex128)
        def form(u, v, w):
            gu = grad(u)
            gv = grad(v)
            r = w.x[0]
            return 2.0 * math.pi * r * (
                (gu[0] * gv[0] + gu[1] * gv[1]) / rho
                - (omega * omega / bulk) * u * v
            )

        return form

    def _pressure_volume_integral(
        self,
        pressure: np.ndarray,
        triangle_indices: np.ndarray,
    ) -> complex:
        """Integrate a P1/P2 pressure field with the exact ``2*pi*r`` weight."""

        @LinearForm
        def volume_load(v, w):
            return 2.0 * math.pi * w.x[0] * v

        load = np.asarray(
            asm(
                volume_load,
                self.basis.with_elements(
                    np.asarray(triangle_indices, dtype=np.int64)
                ),
            ),
            dtype=float,
        )
        return complex(np.dot(load, pressure))

    def _pml_form(self, omega: float, alpha: float):
        rho = self.parameters.rho0_kg_m3
        bulk = self.parameters.bulk_modulus_Pa
        inner = self.parameters.pml_inner_radius_m
        thickness = self.parameters.pml_thickness_m
        exponent = self.parameters.pml_exponent

        @BilinearForm(dtype=np.complex128)
        def form(u, v, w):
            r = w.x[0]
            z = w.x[1]
            radius = np.sqrt(r * r + z * z)
            safe = np.where(radius > 0.0, radius, 1.0)
            er = r / safe
            ez = z / safe
            tr = ez
            tz = -er
            gu = grad(u)
            gv = grad(v)
            du_radial = gu[0] * er + gu[1] * ez
            dv_radial = gv[0] * er + gv[1] * ez
            du_tangent = gu[0] * tr + gu[1] * tz
            dv_tangent = gv[0] * tr + gv[1] * tz
            coeff = pml_coefficients(
                r,
                z,
                inner,
                thickness,
                alpha=alpha,
                exponent=exponent,
            )
            return 2.0 * math.pi * r * (
                (
                    coeff["gradient_radial"] * du_radial * dv_radial
                    + coeff["gradient_tangential"] * du_tangent * dv_tangent
                ) / rho
                - (omega * omega / bulk) * coeff["mass"] * u * v
            )

        return form

    def _assemble_domain_form(
        self,
        frequency_Hz: float,
    ) -> tuple[csr_matrix, csr_matrix, dict[str, Any]]:
        frequency = float(frequency_Hz)
        omega = 2.0 * math.pi * frequency
        alpha = self._alpha_for_frequency(frequency)
        pml = self.mesh_data.pml_triangle_indices
        non_pml = self.mesh_data.non_pml_triangle_indices
        matrix = csr_matrix((self.basis.N, self.basis.N), dtype=np.complex128)
        pml_matrix = csr_matrix((self.basis.N, self.basis.N), dtype=np.complex128)
        if len(non_pml):
            matrix = matrix + asm(self._ordinary_form(omega), self.basis.with_elements(non_pml))
        if len(pml):
            pml_matrix = asm(
                self._pml_form(omega, alpha),
                self.basis.with_elements(pml),
            ).tocsr()
            matrix = matrix + pml_matrix
        points = self.mesh_data.points_rz
        hk_points: list[np.ndarray] = []
        for name in HK_BOUNDARIES:
            for edge in self.mesh_data.facet_edges.get(name, ()):
                hk_points.extend([points[edge[0]], points[edge[1]]])
        if hk_points:
            interface = np.asarray(hk_points, dtype=float)
            coeff = pml_coefficients(
                interface[:, 0],
                interface[:, 1],
                self.parameters.pml_inner_radius_m,
                self.parameters.pml_thickness_m,
                alpha=alpha,
                exponent=self.parameters.pml_exponent,
            )
            interface_error = max(
                float(np.max(np.abs(coeff[name] - 1.0)))
                for name in ("s_R", "s_t", "gradient_radial", "gradient_tangential", "mass")
            )
            operator_coeff = pml_operator_coefficients(
                interface[:, 0],
                interface[:, 1],
                self.parameters.pml_inner_radius_m,
                self.parameters.pml_thickness_m,
                self.parameters.rho0_kg_m3,
                self.parameters.bulk_modulus_Pa,
                alpha=alpha,
                exponent=self.parameters.pml_exponent,
            )
            interface_operator_error = max(
                float(
                    np.max(
                        np.abs(
                            operator_coeff[name]
                            - reference
                        )
                    )
                )
                for name, reference in (
                    (
                        "operator_gradient_radial",
                        1.0 / self.parameters.rho0_kg_m3,
                    ),
                    (
                        "operator_gradient_tangential",
                        1.0 / self.parameters.rho0_kg_m3,
                    ),
                    ("operator_mass", 1.0 / self.parameters.bulk_modulus_Pa),
                )
            )
        else:
            interface_error = math.inf
            interface_operator_error = math.inf
        pml_mid_radius = self.parameters.pml_inner_radius_m + 0.5 * self.parameters.pml_thickness_m
        mid_coeff = pml_coefficients(
            pml_mid_radius,
            0.0,
            self.parameters.pml_inner_radius_m,
            self.parameters.pml_thickness_m,
            alpha=alpha,
            exponent=self.parameters.pml_exponent,
        )
        target = self.parameters.pml_target_attenuation_nepers
        wavenumber = omega / self.parameters.c0_m_s
        if len(pml):
            pml_vertices = self.mesh_data.mesh.p[:, self.mesh_data.mesh.t[:, pml]].transpose(2, 1, 0)
            pml_edge_vectors = pml_vertices[:, [1, 2, 0], :] - pml_vertices[:, [0, 1, 2], :]
            pml_edge_lengths = np.linalg.norm(pml_edge_vectors, axis=2)
            pml_h_min = float(np.min(pml_edge_lengths))
            pml_h_max = float(np.max(pml_edge_lengths))
            pml_centroids = np.mean(pml_vertices, axis=1)
            centroid_coeff = pml_coefficients(
                pml_centroids[:, 0],
                pml_centroids[:, 1],
                self.parameters.pml_inner_radius_m,
                self.parameters.pml_thickness_m,
                alpha=alpha,
                exponent=self.parameters.pml_exponent,
            )
            max_stretched_phase_per_edge = float(
                np.max(
                    np.abs(
                        wavenumber
                        * centroid_coeff["s_R"]
                        * pml_h_max
                    )
                )
            )
        else:
            pml_h_min = None
            pml_h_max = None
            max_stretched_phase_per_edge = None
        pml_diagnostics = {
            "implementation": "spherical radial coordinate transform",
            "affected_domain_names": sorted(PML_DOMAINS),
            "pml_triangle_count": int(len(pml)),
            "non_pml_triangle_count": int(len(non_pml)),
            "inner_radius_m": float(self.parameters.pml_inner_radius_m),
            "thickness_m": float(self.parameters.pml_thickness_m),
            "mode": self.parameters.pml_mode,
            "alpha": float(alpha),
            "configured_alpha": (
                None
                if self.parameters.pml_alpha is None
                else float(self.parameters.pml_alpha)
            ),
            "actual_alpha": float(alpha),
            "exponent": int(self.parameters.pml_exponent),
            "target_attenuation_nepers": float(target),
            "wavenumber_rad_m": float(wavenumber),
            "mesh_resolution_indicator": {
                "pml_edge_length_min_m": pml_h_min,
                "pml_edge_length_max_m": pml_h_max,
                "elements_per_pml_thickness_using_hmax": (
                    None
                    if pml_h_max is None
                    else float(self.parameters.pml_thickness_m / pml_h_max)
                ),
                "max_abs_stretched_k_hmax": max_stretched_phase_per_edge,
            },
            "theoretical_outer_amplitude_factor": float(
                math.exp(-target)
                if self.parameters.pml_mode == TARGET_PML_MODE
                else math.exp(
                    -wavenumber
                    * alpha
                    * self.parameters.pml_thickness_m
                    / (self.parameters.pml_exponent + 1)
                )
            ),
            "interface_max_coefficient_error": float(interface_error),
            "interface_coefficients_equal_one": bool(interface_error <= PML_RADIUS_TOLERANCE_M),
            "interface_operator_max_error": float(interface_operator_error),
            "interface_operator_continuity": bool(
                interface_operator_error <= PML_RADIUS_TOLERANCE_M
            ),
            "mid_pml_s_R_imag": float(np.imag(mid_coeff["s_R"].reshape(-1)[0])),
            "mid_pml_mass_imag": float(np.imag(mid_coeff["mass"].reshape(-1)[0])),
            "exp_iwt_absorption_sign": bool(np.imag(mid_coeff["s_R"].reshape(-1)[0]) < 0.0),
            "outer_boundary_condition": "Dirichlet p=0 on outer_pml_boundary",
            "geometry_contract": self.mesh_data.pml_geometry_report,
        }
        return matrix.tocsr(), pml_matrix, pml_diagnostics

    def _assemble_bli_form(
        self,
        frequency_Hz: float,
    ) -> tuple[
        csr_matrix,
        csr_matrix,
        csr_matrix,
        csr_matrix,
        csr_matrix,
        dict[str, Any],
    ]:
        """Assemble C cavity-wall BLI matrices on selected FacetBasis facets."""

        shape = (self.basis.N, self.basis.N)
        zero = csr_matrix(shape, dtype=np.complex128)
        zero_real = csr_matrix(shape, dtype=float)
        if self.mesh_data.case_id != "C":
            return (
                zero,
                zero,
                zero,
                zero_real,
                zero_real,
                {
                    "status": "not_applicable",
                    "case": self.mesh_data.case_id,
                    "physical_wall_loss": False,
                    "loss_scale": 0.0,
                    "boundary_selection": self.mesh_data.cavity_bli_boundary_report,
                },
            )

        frequency = float(frequency_Hz)
        applicability = assess_bli_applicability(
            self.mesh_data.config,
            (frequency,),
        )
        scale = self.parameters.thermoviscous_loss_scale
        if scale > 0.0 and applicability["route"] != "BLI":
            raise ValueError(
                f"case C BLI is not applicable at {frequency:g} Hz: "
                f"route={applicability['route']}"
            )
        air = ThermoviscousAirProperties.from_config(self.mesh_data.config)
        coefficients = bli_bilinear_coefficients(
            air,
            frequency,
            loss_scale=scale,
        )
        facets = self.mesh_data.cavity_bli_facets
        if not len(facets):
            raise ValueError("case C has no selected cavity BLI facets")
        facet_basis = FacetBasis(
            self.mesh_data.mesh,
            self.element,
            facets=facets,
            intorder=6,
        )

        @BilinearForm
        def tangential_gradient(u, v, w):
            gu = grad(u)
            gv = grad(v)
            normal_gu = gu[0] * w.n[0] + gu[1] * w.n[1]
            normal_gv = gv[0] * w.n[0] + gv[1] * w.n[1]
            tangent_product = (
                gu[0] * gv[0] + gu[1] * gv[1] - normal_gu * normal_gv
            )
            return 2.0 * math.pi * w.x[0] * tangent_product

        @BilinearForm
        def boundary_mass(u, v, w):
            return 2.0 * math.pi * w.x[0] * u * v

        @LinearForm
        def boundary_area(v, w):
            return 2.0 * math.pi * w.x[0] * v

        gradient_geometry = asm(tangential_gradient, facet_basis).tocsr()
        mass_geometry = asm(boundary_mass, facet_basis).tocsr()
        viscous_geometry = (
            self.parameters.viscous_loss_scale * gradient_geometry
        ).tocsr()
        thermal_geometry = (
            self.parameters.thermal_loss_scale * mass_geometry
        ).tocsr()
        viscous_matrix = (
            coefficients.viscous_tangential_gradient_m4_kg * viscous_geometry
        ).astype(np.complex128).tocsr()
        thermal_matrix = (
            coefficients.thermal_pressure_m2_kg * thermal_geometry
        ).astype(np.complex128).tocsr()
        matrix = (viscous_matrix + thermal_matrix).tocsr()
        area_facetbasis = float(np.sum(asm(boundary_area, facet_basis)))
        area_reported = float(
            self.mesh_data.cavity_bli_boundary_report[
                "selected_axisymmetric_area_m2"
            ]
        )
        area_error = abs(area_facetbasis - area_reported)
        if area_error > 1.0e-11 * max(1.0, area_reported):
            raise ValueError(
                "C BLI FacetBasis area disagrees with edge-geometry selection report"
            )
        group_area_crosscheck: dict[str, dict[str, float]] = {}
        for name, group_facets in self.mesh_data.cavity_bli_group_facets.items():
            group_basis = FacetBasis(
                self.mesh_data.mesh,
                self.element,
                facets=group_facets,
                intorder=6,
            )
            integrated = float(np.sum(asm(boundary_area, group_basis)))
            geometric = float(
                self.mesh_data.cavity_bli_boundary_report["physical_groups"][name][
                    "axisymmetric_area_m2"
                ]
            )
            group_area_crosscheck[name] = {
                "edge_geometry_area_m2": geometric,
                "facetbasis_integrated_area_m2": integrated,
                "absolute_error_m2": abs(integrated - geometric),
            }
        diagnostics = {
            "status": "assembled",
            "case": "C",
            "model": "fixed_isothermal_no_slip_BLI",
            "harmonic_convention": "exp(+i*omega*t)",
            "physical_wall_loss": True,
            "loss_scale": float(scale),
            "viscous_loss_scale": float(self.parameters.viscous_loss_scale),
            "thermal_loss_scale": float(self.parameters.thermal_loss_scale),
            "coefficient_source": "thermoviscous_boundaries.bli_bilinear_coefficients",
            "coefficients": coefficients.to_dict(),
            "axisymmetric_weight": "2*pi*r in FacetBasis forms",
            "viscous_operator": "tangential pressure gradient only",
            "thermal_operator": "boundary pressure mass/compliance",
            "boundary_selection": self.mesh_data.cavity_bli_boundary_report,
            "facetbasis_area_m2": area_facetbasis,
            "edge_area_crosscheck_absolute_error_m2": area_error,
            "physical_group_area_crosscheck": group_area_crosscheck,
            "applicability": applicability,
            "final_production_interface_ready": False,
        }
        return (
            matrix,
            viscous_matrix,
            thermal_matrix,
            viscous_geometry,
            thermal_geometry,
            diagnostics,
        )

    def _assemble_trace_rhs(self, frequency_Hz: float, trace_name: str) -> np.ndarray:
        omega = 2.0 * math.pi * float(frequency_Hz)
        facets = self.mesh_data.line_facets[trace_name]
        facet_basis = FacetBasis(self.mesh_data.mesh, self.element, facets=facets, intorder=4)
        velocity = self.parameters.reference_velocity_m_s

        @LinearForm(dtype=np.complex128)
        def rhs(v, w):
            # (1/rho) dp/dn = -i*omega*(v dot n), with the FacetBasis normal
            # pointing from this pressure component into the rigid body.
            return -1j * omega * 2.0 * math.pi * w.x[0] * velocity * w.n[1] * v

        return np.asarray(asm(rhs, facet_basis), dtype=complex)

    def _drive_power_into_fluid(
        self,
        trace_name: str,
        pressure: np.ndarray,
    ) -> float:
        """Return driver power into fluid from the outward-from-fluid normal.

        The FE trace normal points from the pressure domain into the rigid
        displacement body.  Thus the outward acoustic flux is
        ``P_out = 0.5 Re integral(p * conj(v_n) dS)`` with
        ``v_n = v_z*n_z``; power delivered into the fluid is ``-P_out``.
        This sign is derived here independently of the Q bookkeeping.
        """

        samples = facet_samples_from_fe(
            self.mesh_data.mesh,
            self.element,
            self.mesh_data.line_facets[trace_name],
            pressure,
            intorder=6,
        )
        rs, _, _, nz, ds_w, p_boundary, _ = samples
        vn = self.parameters.reference_velocity_m_s * nz
        p_out = np.sum(
            0.5
            * p_boundary
            * np.conj(vn)
            * 2.0
            * math.pi
            * rs
            * ds_w
        )
        return float(-np.real(p_out))

    def _hk_diagnostics(
        self,
        frequency_Hz: float,
        pressure: np.ndarray,
    ) -> dict[str, Any]:
        """Evaluate the complete front+rear HK sphere and its flux power."""

        facets = np.unique(
            np.concatenate(
                [self.mesh_data.line_facets[name] for name in HK_BOUNDARIES]
            )
        )
        if not len(facets):
            raise ValueError("closed HK evaluation requires hk_front and hk_rear facets")
        samples = facet_samples_from_fe(
            self.mesh_data.mesh,
            self.element,
            facets,
            pressure,
            intorder=6,
            prefer_inside_radius=self.parameters.pml_inner_radius_m,
            force_radial_normals=True,
        )
        rs, zs, nr, nz, ds_w, p_boundary, dpdn_boundary = samples
        frequency = float(frequency_Hz)
        wavenumber = 2.0 * math.pi * frequency / self.parameters.c0_m_s
        hk_radius = float(self.parameters.pml_inner_radius_m)

        def axis_pressure_at(z_observation_m: float) -> complex:
            return complex(
                np.asarray(
                    hk_pressure_from_samples(
                        frequency,
                        self.parameters.c0_m_s,
                        rs,
                        zs,
                        nr,
                        nz,
                        ds_w,
                        p_boundary,
                        dpdn_boundary,
                        obs_r=0.0,
                        obs_z=float(z_observation_m),
                        nphi=64,
                        mirror=False,
                        sign=-1,
                    )
                ).reshape(-1)[0]
            )

        # This is the actual closed-HK reconstruction at the requested 1 m
        # point.  At the low frequencies used by the dipole check it is a
        # near-field point (kR << 20), so it is intentionally not used for a
        # far-field 20 dB/decade asymptotic assertion.
        actual_pressure_1m = axis_pressure_at(FAR_FIELD_REFERENCE_DISTANCE_M)
        r_eval = far_field_evaluation_radius(
            frequency,
            self.parameters.c0_m_s,
            hk_radius,
            minimum_kR=FAR_FIELD_MIN_KR,
        )
        pressure_at_r_eval = axis_pressure_at(r_eval)
        # With exp(+i omega t), an outgoing wave is exp(-i k R)/R.  Removing
        # propagation from R_eval to 1 m therefore multiplies by
        # exp(+i*k*(R_eval-1 m)); this is a far-field normalization, not an
        # assertion that the normalized pressure occurs physically at 1 m.
        far_field_phase_factor = np.exp(
            1j * wavenumber * (r_eval - FAR_FIELD_REFERENCE_DISTANCE_M)
        )
        far_field_pressure_1m = pressure_at_r_eval * r_eval * far_field_phase_factor
        axis_pressure = actual_pressure_1m
        hk_power = intensity_power_from_samples(
            frequency,
            self.parameters.rho0_kg_m3,
            rs,
            ds_w,
            p_boundary,
            dpdn_boundary,
        )
        p_ref = float(self.mesh_data.config.raw["air"]["p_ref_Pa"])
        peak = abs(axis_pressure)
        rms = peak / math.sqrt(2.0)
        far_peak = abs(far_field_pressure_1m)
        far_rms = far_peak / math.sqrt(2.0)
        actual_kR = wavenumber * FAR_FIELD_REFERENCE_DISTANCE_M
        actual_is_far_field = bool(actual_kR >= FAR_FIELD_MIN_KR)
        return {
            "mirror": False,
            "surface": "hk_front + hk_rear complete closed spherical surface",
            "normal": "forced spherical radial outward normal (r,z)/R",
            "pressure_side": "physical free-field side of HK, excluding PML side",
            "velocity_formula": "v_n = i/(omega*rho0) * dp/dn",
            "facet_count": int(len(facets)),
            "front_facet_count": int(len(self.mesh_data.line_facets["hk_front"])),
            "rear_facet_count": int(len(self.mesh_data.line_facets["hk_rear"])),
            "axis_observation_r_m": 0.0,
            "axis_observation_z_m": 1.0,
            "actual_pressure_1m_field_regime": (
                "far_field" if actual_is_far_field else "near_field"
            ),
            "actual_pressure_1m_asymptote_eligible": actual_is_far_field,
            "actual_pressure_1m_kR": float(actual_kR),
            "axis_pressure_1m_Pa": {
                "real": float(np.real(axis_pressure)),
                "imag": float(np.imag(axis_pressure)),
            },
            "axis_pressure_peak_Pa": float(peak),
            "axis_pressure_rms_Pa": float(rms),
            "axis_peak_spl_dB": float(20.0 * math.log10(max(peak / p_ref, 1.0e-300))),
            "axis_rms_spl_dB": float(20.0 * math.log10(max(rms / p_ref, 1.0e-300))),
            "axis_phase_deg": float(np.angle(axis_pressure, deg=True)),
            "actual_pressure_1m": {
                "pressure_Pa": {
                    "real": float(np.real(actual_pressure_1m)),
                    "imag": float(np.imag(actual_pressure_1m)),
                },
                "rms_spl_dB": float(
                    20.0 * math.log10(max(rms / p_ref, 1.0e-300))
                ),
                "phase_deg": float(np.angle(actual_pressure_1m, deg=True)),
                "kR": float(actual_kR),
                "field_regime": (
                    "far_field" if actual_is_far_field else "near_field"
                ),
                "asymptote_eligible": actual_is_far_field,
                "asymptote_note": (
                    "At 5-20 Hz this is a near-field 1 m pressure and must not "
                    "be used for a dipole asymptote fit."
                ),
            },
            "far_field_normalized_to_1m": {
                "kernel": "closed HK exact kernel",
                "mirror": False,
                "normalization": (
                    "p_ff_1m = p(R_eval)*R_eval*exp(+i*k*(R_eval-1 m)) "
                    "for exp(+i omega t) and outgoing exp(-i*k*R)/R"
                ),
                "reference_distance_m": FAR_FIELD_REFERENCE_DISTANCE_M,
                "R_eval_m": float(r_eval),
                "R_eval_over_hk_radius": float(r_eval / hk_radius),
                "kR_eval": float(wavenumber * r_eval),
                "minimum_kR": FAR_FIELD_MIN_KR,
                "outside_hk": bool(r_eval > hk_radius),
                "pressure_R_eval_Pa": {
                    "real": float(np.real(pressure_at_r_eval)),
                    "imag": float(np.imag(pressure_at_r_eval)),
                },
                "phase_factor": {
                    "real": float(np.real(far_field_phase_factor)),
                    "imag": float(np.imag(far_field_phase_factor)),
                },
                "pressure_ff_1m_Pa": {
                    "real": float(np.real(far_field_pressure_1m)),
                    "imag": float(np.imag(far_field_pressure_1m)),
                },
                "rms_spl_dB": float(
                    20.0 * math.log10(max(far_rms / p_ref, 1.0e-300))
                ),
                "phase_deg": float(np.angle(far_field_pressure_1m, deg=True)),
                "is_actual_1m_pressure": False,
            },
            "hk_flux_power_W": float(hk_power),
            "geometry_contract": self.mesh_data.hk_geometry_report,
        }

    def assemble(self, frequency_Hz: float) -> AcousticAssemblyResult:
        """Assemble one frequency without solving or calibrating a sign."""

        frequency = float(frequency_Hz)
        if frequency <= 0.0:
            raise ValueError("frequency must be positive")
        matrix, pml_matrix, pml_diagnostics = self._assemble_domain_form(frequency)
        (
            bli_matrix,
            bli_viscous_matrix,
            bli_thermal_matrix,
            bli_tangential_gradient_matrix,
            bli_boundary_mass_matrix,
            thermoviscous_diagnostics,
        ) = self._assemble_bli_form(frequency)
        matrix = (matrix + bli_matrix).tocsr()
        rhs_front = self._assemble_trace_rhs(frequency, REFERENCE_PLANAR_PISTON_FRONT)
        rhs_back = self._assemble_trace_rhs(frequency, REFERENCE_PLANAR_PISTON_BACK)
        rhs = rhs_front + rhs_back
        symmetry = matrix - matrix.T
        symmetry_error = float(np.max(np.abs(symmetry.data))) if symmetry.nnz else 0.0
        return AcousticAssemblyResult(
            frequency_Hz=frequency,
            omega_rad_s=2.0 * math.pi * frequency,
            matrix=matrix,
            pml_matrix=pml_matrix,
            bli_matrix=bli_matrix,
            bli_viscous_matrix=bli_viscous_matrix,
            bli_thermal_matrix=bli_thermal_matrix,
            bli_tangential_gradient_matrix=bli_tangential_gradient_matrix,
            bli_boundary_mass_matrix=bli_boundary_mass_matrix,
            rhs=rhs,
            rhs_front=rhs_front,
            rhs_back=rhs_back,
            dirichlet_dofs=self._outer_dirichlet_dofs.copy(),
            free_dofs=self._free_dofs.copy(),
            dof_component=self._dof_component.copy(),
            component_dofs={key: value.copy() for key, value in self._component_dofs.items()},
            pml_diagnostics=pml_diagnostics,
            thermoviscous_diagnostics=thermoviscous_diagnostics,
            matrix_symmetry_error=symmetry_error,
        )

    def solve(
        self,
        frequency_Hz: float,
        *,
        allow_unresolved_pml: bool = False,
    ) -> AcousticSolveResult:
        """Solve one prescribed-velocity point with p=0 outer PML.

        A non-positive discrete PML absorption or outward HK/input power is a
        hard physical failure.  ``allow_unresolved_pml=True`` is an explicit
        diagnostic escape hatch for recording an unresolved target-nepers
        result; it never changes a sign or calibrates a result.
        """

        assembly = self.assemble(frequency_Hz)
        pressure = np.zeros(self.basis.N, dtype=complex)
        if len(assembly.free_dofs):
            reduced = assembly.matrix[assembly.free_dofs][:, assembly.free_dofs]
            pressure[assembly.free_dofs] = spsolve(reduced, assembly.rhs[assembly.free_dofs])
        residual = assembly.matrix[assembly.free_dofs][:, assembly.free_dofs].dot(
            pressure[assembly.free_dofs]
        ) - assembly.rhs[assembly.free_dofs]
        residual_absolute = float(np.linalg.norm(residual))
        rhs_norm = float(np.linalg.norm(assembly.rhs[assembly.free_dofs]))
        residual_relative = residual_absolute / max(rhs_norm, 1.0e-30)
        omega = assembly.omega_rad_s
        rhs_input_power = float(
            np.imag(np.vdot(pressure, assembly.rhs)) / (2.0 * omega)
        )
        pml_quadratic_form = complex(
            np.vdot(pressure, assembly.pml_matrix.dot(pressure))
        )
        pml_discrete_absorption = float(
            np.imag(pml_quadratic_form) / (2.0 * omega)
        )
        bli_quadratic_form = complex(
            np.vdot(pressure, assembly.bli_matrix.dot(pressure))
        )
        bli_viscous_quadratic_form = complex(
            np.vdot(pressure, assembly.bli_viscous_matrix.dot(pressure))
        )
        bli_thermal_quadratic_form = complex(
            np.vdot(pressure, assembly.bli_thermal_matrix.dot(pressure))
        )
        bli_power_matrix = float(np.imag(bli_quadratic_form) / (2.0 * omega))
        bli_viscous_power_matrix = float(
            np.imag(bli_viscous_quadratic_form) / (2.0 * omega)
        )
        bli_thermal_power_matrix = float(
            np.imag(bli_thermal_quadratic_form) / (2.0 * omega)
        )
        if self.mesh_data.case_id == "C":
            low_level_coefficients = bli_bilinear_coefficients(
                ThermoviscousAirProperties.from_config(self.mesh_data.config),
                frequency_Hz,
                loss_scale=self.parameters.thermoviscous_loss_scale,
            )
            independent_bli_power = bli_dissipation(
                pressure,
                assembly.bli_tangential_gradient_matrix,
                assembly.bli_boundary_mass_matrix,
                low_level_coefficients,
            )
            bli_power_cross_error = float(
                bli_power_matrix - independent_bli_power.P_total_W
            )
        else:
            independent_bli_power = None
            bli_power_cross_error = 0.0

        means: dict[int, complex] = {}
        for component, triangle_indices in self.mesh_data.component_triangles.items():
            triangles = self.mesh_data.mesh.t[:, triangle_indices].T
            volume = axisymmetric_volume(self.mesh_data.points_rz, triangles)
            integral = self._pressure_volume_integral(pressure, triangle_indices)
            means[int(component)] = integral / volume
        cavity_triangles = self.mesh_data.mesh.t[:, self.mesh_data.cavity_triangle_indices].T
        cavity_integral = self._pressure_volume_integral(
            pressure,
            self.mesh_data.cavity_triangle_indices,
        )
        cavity_mean = cavity_integral / self.mesh_data.cavity_volume_m3
        back_trace = self.mesh_data.trace_metrics[REFERENCE_PLANAR_PISTON_BACK]
        q_into_cavity = complex(back_trace["q_into_m3_s"])
        z_box = cavity_mean / q_into_cavity
        analytic = (
            sealed_b_analytic_limit(
                frequency_Hz,
                self.mesh_data.cavity_volume_m3,
                self.parameters.rho0_kg_m3,
                self.parameters.c0_m_s,
                q_into_cavity,
            )
            if self.mesh_data.case_id == "B"
            else None
        )
        relative_error = (
            float(abs(z_box - analytic.impedance_Pa_s_m3) / abs(analytic.impedance_Pa_s_m3))
            if analytic is not None
            else None
        )
        real_ratio = (
            float(abs(np.real(z_box)) / max(abs(z_box), 1.0e-30))
            if analytic is not None or self.mesh_data.case_id == "C"
            else None
        )
        front_trace = self.mesh_data.trace_metrics[REFERENCE_PLANAR_PISTON_FRONT]
        q_out_total = float(
            front_trace["q_out_m3_s"] + back_trace["q_out_m3_s"]
        )
        q_into_total = float(-q_out_total)
        q_scale = max(
            abs(float(front_trace["q_out_m3_s"])),
            abs(float(back_trace["q_out_m3_s"])),
            1.0e-30,
        )
        q_balance_relative_error = float(abs(q_out_total) / q_scale)
        drive_power = {
            "front": self._drive_power_into_fluid(
                REFERENCE_PLANAR_PISTON_FRONT,
                pressure,
            ),
            "back": self._drive_power_into_fluid(
                REFERENCE_PLANAR_PISTON_BACK,
                pressure,
            ),
        }
        drive_power["total"] = float(drive_power["front"] + drive_power["back"])
        boundary_input_cross_error = float(drive_power["total"] - rhs_input_power)
        hk_diagnostics = self._hk_diagnostics(frequency_Hz, pressure)
        hk_diagnostics["drive_power_total_W"] = drive_power["total"]
        hk_diagnostics["power_balance_residual_W"] = float(
            hk_diagnostics["hk_flux_power_W"] - drive_power["total"]
        )
        hk_diagnostics["power_balance_relative_to_drive"] = float(
            abs(hk_diagnostics["power_balance_residual_W"])
            / max(abs(drive_power["total"]), 1.0e-30)
        )
        hk_diagnostics["power_balance_with_bli_residual_W"] = float(
            drive_power["total"]
            - hk_diagnostics["hk_flux_power_W"]
            - bli_power_matrix
        )
        hk_diagnostics["power_balance_with_bli_relative_to_input"] = float(
            abs(hk_diagnostics["power_balance_with_bli_residual_W"])
            / max(abs(drive_power["total"]), 1.0e-30)
        )
        pml_diagnostics = dict(assembly.pml_diagnostics)
        pml_diagnostics.update(
            {
                "discrete_absorption_definition": (
                    "Im(p^H A_pml p)/(2 omega); numerical PML absorption, "
                    "not physical material loss"
                ),
                "pml_quadratic_form": {
                    "real": float(np.real(pml_quadratic_form)),
                    "imag": float(np.imag(pml_quadratic_form)),
                },
                "discrete_absorption_power_W": pml_discrete_absorption,
                "input_power_boundary_W": float(drive_power["total"]),
                "input_power_rhs_W": rhs_input_power,
                "input_power_boundary_rhs_error_W": boundary_input_cross_error,
            }
        )
        input_power_closure_residual = float(
            rhs_input_power - pml_discrete_absorption - bli_power_matrix
        )
        input_power_closure_relative = float(
            abs(input_power_closure_residual)
            / max(abs(rhs_input_power), 1.0e-30)
        )
        thermoviscous_diagnostics = dict(assembly.thermoviscous_diagnostics)
        thermoviscous_diagnostics.update(
            {
                "dissipation_definition": (
                    "physical fixed-wall BLI loss; distinct from numerical PML absorption"
                ),
                "P_visc_W": bli_viscous_power_matrix,
                "P_thermal_W": bli_thermal_power_matrix,
                "P_total_W": bli_power_matrix,
                "matrix_quadratic_form": {
                    "real": float(np.real(bli_quadratic_form)),
                    "imag": float(np.imag(bli_quadratic_form)),
                },
                "independent_quadratic": (
                    independent_bli_power.to_dict()
                    if independent_bli_power is not None
                    else {
                        "P_visc_W": 0.0,
                        "P_thermal_W": 0.0,
                        "P_total_W": 0.0,
                        "passive": True,
                    }
                ),
                "matrix_independent_power_cross_error_W": bli_power_cross_error,
                "input_power_closure": {
                    "identity": "P_input = PML numerical absorption + physical BLI loss",
                    "input_power_W": rhs_input_power,
                    "pml_numerical_absorption_W": pml_discrete_absorption,
                    "bli_physical_loss_W": bli_power_matrix,
                    "residual_W": input_power_closure_residual,
                    "relative_to_input": input_power_closure_relative,
                },
                "hk_crosscheck": {
                    "identity": "P_input approximately HK outward flux + physical BLI loss",
                    "hk_outward_flux_W": float(hk_diagnostics["hk_flux_power_W"]),
                    "residual_W": float(
                        hk_diagnostics["power_balance_with_bli_residual_W"]
                    ),
                    "relative_to_input": float(
                        hk_diagnostics[
                            "power_balance_with_bli_relative_to_input"
                        ]
                    ),
                    "is_primary_closure": False,
                },
            }
        )
        passivity_failures = []
        if drive_power["total"] <= 0.0:
            passivity_failures.append("non_positive_input_power")
        if hk_diagnostics["hk_flux_power_W"] <= 0.0:
            passivity_failures.append("non_positive_hk_outward_power")
        if pml_discrete_absorption <= 0.0:
            passivity_failures.append("non_positive_discrete_pml_absorption")
        bli_tolerance = 1.0e-13 * max(abs(rhs_input_power), 1.0)
        if bli_viscous_power_matrix < -bli_tolerance:
            passivity_failures.append("negative_physical_bli_viscous_loss")
        if bli_thermal_power_matrix < -bli_tolerance:
            passivity_failures.append("negative_physical_bli_thermal_loss")
        if bli_power_matrix < -bli_tolerance:
            passivity_failures.append("negative_physical_bli_total_loss")
        # This cross-check compares two algebraically equivalent evaluations
        # of the same assembled BLI quadratic.  Large low-frequency pressures
        # make their floating-point cancellation error dimensional and
        # scale-dependent; this tolerance does not enter either physical
        # power value or the L1/L2 convergence gate.
        bli_cross_absolute_tolerance_W = 5.0e-9
        bli_cross_relative_tolerance = 1.0e-6
        bli_cross_scale_W = max(
            abs(bli_power_matrix),
            abs(rhs_input_power),
            abs(pml_discrete_absorption),
            (
                abs(independent_bli_power.P_total_W)
                if independent_bli_power is not None
                else 0.0
            ),
        )
        bli_cross_tolerance = (
            bli_cross_absolute_tolerance_W
            + bli_cross_relative_tolerance * bli_cross_scale_W
        )
        thermoviscous_diagnostics.update(
            {
                "matrix_independent_power_cross_absolute_tolerance_W": (
                    bli_cross_absolute_tolerance_W
                ),
                "matrix_independent_power_cross_relative_tolerance": (
                    bli_cross_relative_tolerance
                ),
                "matrix_independent_power_cross_scale_W": bli_cross_scale_W,
            }
        )
        thermoviscous_diagnostics["matrix_independent_power_cross_tolerance_W"] = (
            bli_cross_tolerance
        )
        if abs(bli_power_cross_error) > bli_cross_tolerance:
            passivity_failures.append("bli_matrix_independent_power_mismatch")
        if input_power_closure_relative > 2.0e-2 and abs(
            input_power_closure_residual
        ) > 1.0e-10:
            passivity_failures.append("input_pml_bli_power_closure")
        thermoviscous_diagnostics["passivity_status"] = (
            "pass" if not any("bli" in item for item in passivity_failures) else "fail"
        )
        pml_diagnostics["passivity_status"] = "pass" if not passivity_failures else "unresolved"
        pml_diagnostics["passivity_failures"] = passivity_failures
        if passivity_failures and not allow_unresolved_pml:
            raise RuntimeError(
                "reference PML/passivity unresolved: "
                + ", ".join(passivity_failures)
                + f"; input={drive_power['total']:.6e}, "
                + f"hk={hk_diagnostics['hk_flux_power_W']:.6e}, "
                + f"pml={pml_discrete_absorption:.6e}"
                + f", bli={bli_power_matrix:.6e}, "
                + f"closure={input_power_closure_residual:.6e}"
            )
        return AcousticSolveResult(
            frequency_Hz=float(frequency_Hz),
            pressure=pressure,
            assembly=assembly,
            mesh=self.mesh_data,
            parameters=self.parameters,
            component_mean_pressure_Pa=means,
            cavity_mean_pressure_Pa=cavity_mean,
            front_back_traces={
                name: dict(metrics)
                for name, metrics in self.mesh_data.trace_metrics.items()
            },
            q_into_cavity_m3_s=q_into_cavity,
            z_box_Pa_s_m3=z_box,
            analytic_limit=analytic,
            relative_impedance_error=relative_error,
            residual_absolute=residual_absolute,
            residual_relative=float(residual_relative),
            cavity_real_impedance_ratio=real_ratio,
            q_out_total_m3_s=q_out_total,
            q_into_total_m3_s=q_into_total,
            q_balance_relative_error=q_balance_relative_error,
            drive_power_into_fluid_W=drive_power,
            input_power_from_rhs_W=rhs_input_power,
            input_power_boundary_cross_error_W=boundary_input_cross_error,
            hk_diagnostics=hk_diagnostics,
            pml_diagnostics=pml_diagnostics,
            thermoviscous_diagnostics=thermoviscous_diagnostics,
        )


# Functional aliases keep the loading/assembly/solve steps discoverable to
# stage-3 callers without exposing the legacy production FEM solver.
load_pressure_mesh = load_reference_pressure_mesh


def assemble_reference_acoustics(
    mesh_path: str | Path,
    config_path: str | Path,
    frequency_Hz: float,
    **kwargs: Any,
) -> AcousticAssemblyResult:
    """Load reference A/B and assemble one reference acoustic frequency."""

    return ReferencePrescribedVelocityAcoustics.from_files(
        mesh_path,
        config_path,
        **kwargs,
    ).assemble(frequency_Hz)


def solve_reference_acoustics(
    mesh_path: str | Path,
    config_path: str | Path,
    frequency_Hz: float,
    **kwargs: Any,
) -> AcousticSolveResult:
    """Load reference A/B and solve one reference acoustic frequency."""

    return ReferencePrescribedVelocityAcoustics.from_files(
        mesh_path,
        config_path,
        **kwargs,
    ).solve(frequency_Hz)


# Short aliases for likely stage-3 notebook callers.
AxisymmetricAcousticModel = ReferencePrescribedVelocityAcoustics
AcousticModel = ReferencePrescribedVelocityAcoustics
analytic_sealed_b_impedance = sealed_b_analytic_limit


__all__ = [
    "AcousticAssemblyResult",
    "AcousticModel",
    "AcousticPhysicalParameters",
    "AcousticSolveResult",
    "AxisymmetricP1Operators",
    "AxisymmetricAcousticModel",
    "CAVITY_VOLUME_RELATIVE_TOLERANCE",
    "DEFAULT_PML_MODE",
    "DEFAULT_PML_TARGET_ATTENUATION_NEPERS",
    "EXPLICIT_PML_MODE",
    "FAR_FIELD_MIN_KR",
    "FAR_FIELD_REFERENCE_DISTANCE_M",
    "TARGET_PML_MODE",
    "ReferencePrescribedVelocityAcoustics",
    "ReferencePressureMesh",
    "SealedBAnalyticLimit",
    "PML_GEOMETRY_TOLERANCE_M",
    "analytic_sealed_b_impedance",
    "assemble_reference_acoustics",
    "assemble_axisymmetric_operators",
    "axisymmetric_triangle_volumes",
    "axisymmetric_volume",
    "far_field_evaluation_radius",
    "load_pressure_mesh",
    "load_reference_pressure_mesh",
    "pml_coefficients",
    "pml_alpha_for_frequency",
    "pml_operator_coefficients",
    "sealed_b_analytic_limit",
    "sha256_file",
    "solve_reference_acoustics",
    "validate_pml_geometry",
    "validate_closed_hk_geometry",
]
