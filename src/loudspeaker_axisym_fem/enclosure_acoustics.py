r"""Reference-only axisymmetric acoustics for the sealed enclosure demonstrator.

This module is the deliberately small ``Stage 3A-B sealed reference-only
core``; it is not a complete Stage 3A implementation.  It loads the
audited reference mesh, keeps only ``air_*`` triangles in the pressure space,
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
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import meshio
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from skfem import Basis, BilinearForm, ElementTriP1, FacetBasis, LinearForm, MeshTri, asm
from skfem.helpers import grad

from .enclosure_geometry import DOMAIN_PHYSICAL_TAGS, expected_domain_names
from .enclosure_schema import EnclosureConfig, load_enclosure_config
from .enclosure_topology import audit_mesh


REFERENCE_PLANAR_PISTON_IDENTITY = "reference planar piston"
REFERENCE_PLANAR_PISTON_FRONT = "reference_planar_piston_front"
REFERENCE_PLANAR_PISTON_BACK = "reference_planar_piston_back"
OUTER_PML_BOUNDARY = "outer_pml_boundary"
HK_BOUNDARIES = ("hk_front", "hk_rear")
PML_DOMAINS = frozenset(("air_pml_front", "air_pml_rear"))

DEFAULT_REFERENCE_VELOCITY_M_S = 1.0
DEFAULT_PML_ALPHA = 4.0
DEFAULT_PML_EXPONENT = 2
PML_RADIUS_TOLERANCE_M = 2.0e-10
PML_GEOMETRY_TOLERANCE_M = 2.0e-8
CAVITY_VOLUME_RELATIVE_TOLERANCE = 5.0e-3


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


@dataclass(frozen=True)
class AcousticPhysicalParameters:
    """Validated material and reference-drive parameters for phase 3A."""

    rho0_kg_m3: float
    c0_m_s: float
    bulk_modulus_Pa: float
    pml_inner_radius_m: float
    pml_thickness_m: float
    reference_velocity_m_s: float = DEFAULT_REFERENCE_VELOCITY_M_S
    pml_alpha: float = DEFAULT_PML_ALPHA
    pml_exponent: int = DEFAULT_PML_EXPONENT


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
    trace_metrics: dict[str, dict[str, Any]]

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
    rhs: np.ndarray
    rhs_front: np.ndarray
    rhs_back: np.ndarray
    dirichlet_dofs: np.ndarray
    free_dofs: np.ndarray
    dof_component: np.ndarray
    component_dofs: dict[int, np.ndarray]
    pml_diagnostics: dict[str, Any]
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
    analytic_limit: SealedBAnalyticLimit
    relative_impedance_error: float
    residual_absolute: float
    residual_relative: float
    cavity_real_impedance_ratio: float
    pml_diagnostics: dict[str, Any]

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
            "pressure_dof_count": int(len(self.pressure)),
            "pressure_triangle_count": int(self.mesh.pressure_triangle_count),
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
            },
            "relative_impedance_error": float(self.relative_impedance_error),
            "residual_absolute": float(self.residual_absolute),
            "residual_relative": float(self.residual_relative),
            "cavity_real_impedance_ratio": float(self.cavity_real_impedance_ratio),
            "front_back_traces": self.front_back_traces,
            "pml_diagnostics": self.pml_diagnostics,
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


def load_reference_pressure_mesh(
    mesh_path: str | Path,
    config_path: str | Path,
    *,
    case_id: str = "B",
    reference_velocity_m_s: float = DEFAULT_REFERENCE_VELOCITY_M_S,
) -> ReferencePressureMesh:
    """Load an audited sealed-B mesh and build a pressure-only P1 map.

    Phase 3A intentionally rejects all cases other than sealed lossless B.  In
    particular, this prevents an open A mesh from being mistaken for a tested
    radiation or production interface.
    """

    path = Path(mesh_path)
    config_file = Path(config_path)
    requested_case = str(case_id).upper()
    if requested_case != "B":
        raise ValueError("phase 3A reference acoustics supports sealed case B only")
    config = load_enclosure_config(config_file)
    if config.case != "sealed_lossless":
        raise ValueError("phase 3A analytic limit requires sealed_lossless.json")
    audit = audit_mesh(path, case_id="B", config_path=config_file)
    if audit["status"] != "pass":
        raise ValueError(
            "reference mesh failed exact enclosure audit: "
            + ", ".join(audit.get("failures", []))
        )
    meshio_mesh = meshio.read(path)
    field_data = _field_data_map(meshio_mesh)
    expected_names = set(expected_domain_names("B"))
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
        "B",
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
        line_facets[name] = facets
        facet_edges[name] = edges
        facet_normals[name] = normals

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
        raise ValueError("sealed B pressure mesh must have one air_cavity component")
    cavity_component = int(cavity_components[0])
    if component_domains[cavity_component] != ("air_cavity",):
        raise ValueError("sealed B air_cavity is not pressure-disconnected from exterior")
    exterior_components = tuple(
        sorted(component for component in component_triangles if component != cavity_component)
    )
    if len(exterior_components) != 1:
        raise ValueError("sealed B exterior pressure field must be one connected component")

    cavity_triangle_indices = component_triangles[cavity_component]
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
        raise ValueError("pressure-only cavity volume fails the sealed B volume contract")

    dof_component = np.full(pressure_mesh.p.shape[1], -1, dtype=np.int64)
    for component, dofs in component_dofs.items():
        dof_component[dofs] = int(component)
    if np.any(dof_component < 0):
        raise ValueError("pressure mesh contains an unassigned pressure DOF")

    return ReferencePressureMesh(
        path=path,
        source_sha256=sha256_file(path),
        case_id="B",
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
        trace_metrics=trace_metrics,
    )


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
    """P1 axisymmetric reference solver for sealed demonstrator B only."""

    def __init__(
        self,
        pressure_mesh: ReferencePressureMesh,
        *,
        reference_velocity_m_s: float = DEFAULT_REFERENCE_VELOCITY_M_S,
        pml_alpha: float = DEFAULT_PML_ALPHA,
        pml_exponent: int = DEFAULT_PML_EXPONENT,
    ) -> None:
        if pressure_mesh.case_id != "B":
            raise ValueError("phase 3A reference solver supports sealed case B only")
        if pml_alpha <= 0.0 or int(pml_exponent) < 1:
            raise ValueError("PML alpha must be positive and exponent >= 1")
        air = pressure_mesh.config.raw["air"]
        geometry = pressure_mesh.config.raw["geometry"]
        self.mesh_data = pressure_mesh
        self.element = ElementTriP1()
        self.basis = Basis(pressure_mesh.mesh, self.element, intorder=4)
        self.parameters = AcousticPhysicalParameters(
            rho0_kg_m3=float(air["rho0_kg_m3"]),
            c0_m_s=float(air["c0_m_s"]),
            bulk_modulus_Pa=float(air["rho0_kg_m3"]) * float(air["c0_m_s"]) ** 2,
            pml_inner_radius_m=float(geometry["pml_inner_radius_m"]),
            pml_thickness_m=float(geometry["pml_thickness_m"]),
            reference_velocity_m_s=float(reference_velocity_m_s),
            pml_alpha=float(pml_alpha),
            pml_exponent=int(pml_exponent),
        )
        self._dof_component = np.full(self.basis.N, -1, dtype=np.int64)
        for component, dofs in pressure_mesh.component_dofs.items():
            self._dof_component[dofs] = int(component)
        if np.any(self._dof_component < 0):
            raise ValueError("scikit-fem P1 pressure DOFs are not component-mapped")
        self._free_dofs = np.setdiff1d(
            np.arange(self.basis.N, dtype=np.int64), pressure_mesh.outer_dirichlet_dofs
        )

    @classmethod
    def from_files(
        cls,
        mesh_path: str | Path,
        config_path: str | Path,
        *,
        case_id: str = "B",
        reference_velocity_m_s: float = DEFAULT_REFERENCE_VELOCITY_M_S,
        pml_alpha: float = DEFAULT_PML_ALPHA,
        pml_exponent: int = DEFAULT_PML_EXPONENT,
    ) -> "ReferencePrescribedVelocityAcoustics":
        mesh_data = load_reference_pressure_mesh(
            mesh_path,
            config_path,
            case_id=case_id,
            reference_velocity_m_s=reference_velocity_m_s,
        )
        return cls(
            mesh_data,
            reference_velocity_m_s=reference_velocity_m_s,
            pml_alpha=pml_alpha,
            pml_exponent=pml_exponent,
        )

    @property
    def pressure_dof_count(self) -> int:
        return int(self.basis.N)

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

    def _pml_form(self, omega: float):
        rho = self.parameters.rho0_kg_m3
        bulk = self.parameters.bulk_modulus_Pa
        inner = self.parameters.pml_inner_radius_m
        thickness = self.parameters.pml_thickness_m
        alpha = self.parameters.pml_alpha
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
                coeff["gradient_radial"] * du_radial * dv_radial
                + coeff["gradient_tangential"] * du_tangent * dv_tangent
                - (omega * omega / bulk) * coeff["mass"] * u * v
            )

        return form

    def _assemble_domain_form(
        self,
        frequency_Hz: float,
    ) -> tuple[csr_matrix, dict[str, Any]]:
        omega = 2.0 * math.pi * float(frequency_Hz)
        pml = self.mesh_data.pml_triangle_indices
        non_pml = self.mesh_data.non_pml_triangle_indices
        matrix = csr_matrix((self.basis.N, self.basis.N), dtype=np.complex128)
        if len(non_pml):
            matrix = matrix + asm(self._ordinary_form(omega), self.basis.with_elements(non_pml))
        if len(pml):
            matrix = matrix + asm(self._pml_form(omega), self.basis.with_elements(pml))
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
                alpha=self.parameters.pml_alpha,
                exponent=self.parameters.pml_exponent,
            )
            interface_error = max(
                float(np.max(np.abs(coeff[name] - 1.0)))
                for name in ("s_R", "s_t", "gradient_radial", "gradient_tangential", "mass")
            )
        else:
            interface_error = math.inf
        pml_mid_radius = self.parameters.pml_inner_radius_m + 0.5 * self.parameters.pml_thickness_m
        mid_coeff = pml_coefficients(
            pml_mid_radius,
            0.0,
            self.parameters.pml_inner_radius_m,
            self.parameters.pml_thickness_m,
            alpha=self.parameters.pml_alpha,
            exponent=self.parameters.pml_exponent,
        )
        pml_diagnostics = {
            "implementation": "spherical radial coordinate transform",
            "affected_domain_names": sorted(PML_DOMAINS),
            "pml_triangle_count": int(len(pml)),
            "non_pml_triangle_count": int(len(non_pml)),
            "inner_radius_m": float(self.parameters.pml_inner_radius_m),
            "thickness_m": float(self.parameters.pml_thickness_m),
            "alpha": float(self.parameters.pml_alpha),
            "exponent": int(self.parameters.pml_exponent),
            "interface_max_coefficient_error": float(interface_error),
            "interface_coefficients_equal_one": bool(interface_error <= PML_RADIUS_TOLERANCE_M),
            "mid_pml_s_R_imag": float(np.imag(mid_coeff["s_R"].reshape(-1)[0])),
            "mid_pml_mass_imag": float(np.imag(mid_coeff["mass"].reshape(-1)[0])),
            "exp_iwt_absorption_sign": bool(np.imag(mid_coeff["s_R"].reshape(-1)[0]) < 0.0),
            "outer_boundary_condition": "Dirichlet p=0 on outer_pml_boundary",
            "geometry_contract": self.mesh_data.pml_geometry_report,
        }
        return matrix.tocsr(), pml_diagnostics

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

    def assemble(self, frequency_Hz: float) -> AcousticAssemblyResult:
        """Assemble one frequency without solving or calibrating a sign."""

        frequency = float(frequency_Hz)
        if frequency <= 0.0:
            raise ValueError("frequency must be positive")
        matrix, pml_diagnostics = self._assemble_domain_form(frequency)
        rhs_front = self._assemble_trace_rhs(frequency, REFERENCE_PLANAR_PISTON_FRONT)
        rhs_back = self._assemble_trace_rhs(frequency, REFERENCE_PLANAR_PISTON_BACK)
        rhs = rhs_front + rhs_back
        symmetry = matrix - matrix.T
        symmetry_error = float(np.max(np.abs(symmetry.data))) if symmetry.nnz else 0.0
        return AcousticAssemblyResult(
            frequency_Hz=frequency,
            omega_rad_s=2.0 * math.pi * frequency,
            matrix=matrix,
            rhs=rhs,
            rhs_front=rhs_front,
            rhs_back=rhs_back,
            dirichlet_dofs=self.mesh_data.outer_dirichlet_dofs.copy(),
            free_dofs=self._free_dofs.copy(),
            dof_component=self._dof_component.copy(),
            component_dofs={key: value.copy() for key, value in self.mesh_data.component_dofs.items()},
            pml_diagnostics=pml_diagnostics,
            matrix_symmetry_error=symmetry_error,
        )

    def solve(self, frequency_Hz: float) -> AcousticSolveResult:
        """Solve one prescribed-velocity reference point, with p=0 outer PML."""

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

        means: dict[int, complex] = {}
        for component, triangle_indices in self.mesh_data.component_triangles.items():
            triangles = self.mesh_data.mesh.t[:, triangle_indices].T
            volume = axisymmetric_volume(self.mesh_data.points_rz, triangles)
            integral = _integral_r_times_linear_field(
                self.mesh_data.points_rz,
                triangles,
                pressure,
            )
            means[int(component)] = integral / volume
        cavity_mean = means[self.mesh_data.cavity_component]
        back_trace = self.mesh_data.trace_metrics[REFERENCE_PLANAR_PISTON_BACK]
        q_into_cavity = complex(back_trace["q_into_m3_s"])
        z_box = cavity_mean / q_into_cavity
        analytic = sealed_b_analytic_limit(
            frequency_Hz,
            self.mesh_data.cavity_volume_m3,
            self.parameters.rho0_kg_m3,
            self.parameters.c0_m_s,
            q_into_cavity,
        )
        relative_error = float(abs(z_box - analytic.impedance_Pa_s_m3) / abs(analytic.impedance_Pa_s_m3))
        real_ratio = float(abs(np.real(z_box)) / max(abs(z_box), 1.0e-30))
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
            pml_diagnostics=dict(assembly.pml_diagnostics),
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
    """Load sealed B and assemble one reference acoustic frequency."""

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
    """Load sealed B and solve one reference acoustic frequency."""

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
    "ReferencePrescribedVelocityAcoustics",
    "ReferencePressureMesh",
    "SealedBAnalyticLimit",
    "PML_GEOMETRY_TOLERANCE_M",
    "analytic_sealed_b_impedance",
    "assemble_reference_acoustics",
    "assemble_axisymmetric_operators",
    "axisymmetric_triangle_volumes",
    "axisymmetric_volume",
    "load_pressure_mesh",
    "load_reference_pressure_mesh",
    "pml_coefficients",
    "sealed_b_analytic_limit",
    "sha256_file",
    "solve_reference_acoustics",
    "validate_pml_geometry",
]
