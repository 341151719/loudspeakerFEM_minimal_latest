from __future__ import annotations

from pathlib import Path

import meshio
import numpy as np
import pytest
from scipy.sparse.linalg import eigsh

from loudspeaker_axisym_fem.enclosure_acoustics import (
    AxisymmetricP1Operators,
    REFERENCE_PLANAR_PISTON_BACK,
    REFERENCE_PLANAR_PISTON_FRONT,
    ReferencePrescribedVelocityAcoustics,
    assemble_axisymmetric_operators,
    axisymmetric_triangle_volumes,
    load_reference_pressure_mesh,
    pml_coefficients,
    validate_pml_geometry,
)
from loudspeaker_axisym_fem.enclosure_geometry import generate_reference_mesh


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "enclosures"
B_CONFIG = CONFIG_DIR / "sealed_lossless.json"


@pytest.fixture(scope="module")
def b_meshes(tmp_path_factory):
    directory = tmp_path_factory.mktemp("enclosure_acoustics_b")
    paths = {}
    for level in ("L0", "L1", "L2"):
        path = directory / f"B_{level}.msh"
        generate_reference_mesh(B_CONFIG, level, path)
        paths[level] = path
    return paths


@pytest.fixture(scope="module")
def b_models(b_meshes):
    return {
        level: ReferencePrescribedVelocityAcoustics.from_files(path, B_CONFIG)
        for level, path in b_meshes.items()
    }


def _write_msh22_field_variant(source: Path, destination: Path, mutator, point_mutator=None) -> None:
    mesh = meshio.read(source)
    fields = {
        name: np.array(value, dtype=np.int64, copy=True)
        for name, value in mesh.field_data.items()
    }
    mutator(fields)
    points = np.asarray(mesh.points, dtype=float, copy=True)
    if point_mutator is not None:
        point_mutator(points, mesh)
    cells = [
        ("line", np.asarray(mesh.cells_dict["line"], dtype=np.int64)),
        ("triangle", np.asarray(mesh.cells_dict["triangle"], dtype=np.int64)),
    ]
    physical = mesh.cell_data_dict["gmsh:physical"]
    variant = meshio.Mesh(
        points=points,
        cells=cells,
        cell_data={
            "gmsh:physical": [
                np.asarray(physical["line"], dtype=np.int64),
                np.asarray(physical["triangle"], dtype=np.int64),
            ]
        },
        point_data={
            "gmsh:dim_tags": np.column_stack(
                [
                    np.zeros(len(mesh.points), dtype=np.int64),
                    np.ones(len(mesh.points), dtype=np.int64),
                ]
            )
        },
        field_data=fields,
    )
    meshio.write(destination, variant, file_format="gmsh22")


def _simple_cylinder_mesh(a: float, length: float, nr: int, nz: int):
    radii = np.linspace(0.0, a, nr + 1)
    heights = np.linspace(0.0, length, nz + 1)
    points = np.asarray(
        [(radius, height) for height in heights for radius in radii],
        dtype=float,
    )
    triangles = []
    for j in range(nz):
        for i in range(nr):
            lower = j * (nr + 1) + i
            triangles.extend(
                (
                    [lower, lower + 1, lower + nr + 2],
                    [lower, lower + nr + 2, lower + nr + 1],
                )
            )
    return points, np.asarray(triangles, dtype=np.int64)


def _first_nonzero_neumann_frequency(operators: AxisymmetricP1Operators, c0: float) -> float:
    eigenvalues, _ = eigsh(
        operators.stiffness,
        M=operators.compressibility_mass,
        k=5,
        which="SM",
        tol=1.0e-10,
        maxiter=10000,
    )
    eigenvalues = np.sort(np.asarray(eigenvalues, dtype=float))
    threshold = max(1.0e-12, 1.0e-11 * float(np.max(np.abs(eigenvalues))))
    positive = eigenvalues[eigenvalues > threshold]
    if not len(positive):
        raise AssertionError(f"no nonzero Neumann mode found: {eigenvalues}")
    return float(np.sqrt(positive[0]) / (2.0 * np.pi))


def test_axisymmetric_constant_integral_preserves_two_pi_r():
    points = np.asarray([[0.0, 0.0], [0.1, 0.0], [0.1, 0.2], [0.0, 0.2]])
    triangles = np.asarray([[0, 1, 2], [0, 2, 3]])
    volumes = axisymmetric_triangle_volumes(points, triangles)
    assert np.sum(volumes) == pytest.approx(np.pi * 0.1**2 * 0.2, rel=1.0e-13)


def test_pressure_only_mesh_has_separate_b_components_and_automatic_trace_normals(b_models):
    model = b_models["L1"]
    data = model.mesh_data
    assert all(name.startswith("air_") for name in data.triangle_domain_names)
    assert all("rigid_" not in domains for domains in data.component_domains.values())
    assert data.component_count == 2
    assert data.component_domains[data.cavity_component] == ("air_cavity",)
    assert len(data.exterior_components) == 1
    assert not (
        set(data.component_dofs[data.cavity_component])
        & set(data.component_dofs[data.exterior_components[0]])
    )
    front = data.trace_metrics[REFERENCE_PLANAR_PISTON_FRONT]
    back = data.trace_metrics[REFERENCE_PLANAR_PISTON_BACK]
    assert set(front["node_indices"]).isdisjoint(back["node_indices"])
    assert front["q_out_m3_s"] == pytest.approx(-back["q_out_m3_s"], rel=1.0e-13)
    assert front["q_into_m3_s"] == pytest.approx(-back["q_into_m3_s"], rel=1.0e-13)
    assert front["normal_z_range"] == [-1.0, -1.0]
    assert back["normal_z_range"] == [1.0, 1.0]
    assert len(data.outer_dirichlet_dofs) > 0


def test_pml_is_radial_tensor_only_and_matrix_is_complex_symmetric(b_models):
    model = b_models["L0"]
    interface = pml_coefficients(0.35, 0.0, 0.35, 0.1)
    for name in ("s_R", "s_t", "gradient_radial", "gradient_tangential", "mass"):
        assert interface[name] == pytest.approx(1.0 + 0.0j, abs=1.0e-14)
    middle = pml_coefficients(0.4, 0.0, 0.35, 0.1)
    assert np.imag(middle["s_R"]) < 0.0
    assert np.imag(middle["mass"]) < 0.0

    assembly = model.assemble(20.0)
    assert np.iscomplexobj(assembly.matrix.data)
    assert assembly.matrix_symmetry_error < 1.0e-10
    cavity_dofs = model.mesh_data.component_dofs[model.mesh_data.cavity_component]
    cavity_block = assembly.matrix[cavity_dofs][:, cavity_dofs]
    cavity_asymmetry = cavity_block - cavity_block.T
    assert not cavity_asymmetry.nnz or np.max(np.abs(cavity_asymmetry.data)) < 1.0e-10
    assert assembly.pml_diagnostics["affected_domain_names"] == [
        "air_pml_front",
        "air_pml_rear",
    ]
    assert assembly.pml_diagnostics["interface_coefficients_equal_one"] is True
    assert assembly.pml_diagnostics["exp_iwt_absorption_sign"] is True
    assert assembly.pml_diagnostics["outer_boundary_condition"] == "Dirichlet p=0 on outer_pml_boundary"


def test_independent_axisymmetric_gradient_eigenmode_converges():
    # The analytic value is deliberately written independently of the tested
    # acoustics code: for m=0 Neumann cylinder modes, f_z=c/(2L).
    a = 0.01
    length = 0.2
    c0 = 343.0
    rho0 = 1.2
    bulk = rho0 * c0**2
    measured = []
    for nr, nz in ((4, 8), (8, 16)):
        points, triangles = _simple_cylinder_mesh(a, length, nr, nz)
        operators = assemble_axisymmetric_operators(
            points,
            triangles,
            rho0,
            bulk,
        )
        assert operators.stiffness.shape == operators.compressibility_mass.shape
        assert operators.stiffness.shape[0] == len(points)
        stiffness_asymmetry = operators.stiffness - operators.stiffness.T
        mass_asymmetry = operators.compressibility_mass - operators.compressibility_mass.T
        assert not stiffness_asymmetry.nnz or np.max(np.abs(stiffness_asymmetry.data)) < 1.0e-12
        assert not mass_asymmetry.nnz or np.max(np.abs(mass_asymmetry.data)) < 1.0e-12
        measured.append(_first_nonzero_neumann_frequency(operators, c0))

    expected = c0 / (2.0 * length)
    coarse_error = abs(measured[0] - expected) / expected
    fine_error = abs(measured[1] - expected) / expected
    assert measured[0] < 1.1 * expected
    assert fine_error < coarse_error
    assert fine_error < 0.01


def test_pml_geometry_contract_and_mislabelled_triangle_rejection(b_models):
    data = b_models["L1"].mesh_data
    report = validate_pml_geometry(
        data.points_rz,
        data.mesh.t.T,
        data.triangle_domain_names,
        data.facet_edges,
        data.config.raw["geometry"]["pml_inner_radius_m"],
        data.config.raw["geometry"]["pml_thickness_m"],
    )
    assert report["passed"] is True
    assert report["pml_triangle_R_range_m"][0] >= report["inner_radius_m"] - report["tolerance_m"]
    assert report["pml_triangle_R_range_m"][1] <= report["outer_radius_m"] + report["tolerance_m"]
    for name in ("hk_front", "hk_rear"):
        low, high = report["hk_node_R_ranges_m"][name]
        assert low == pytest.approx(report["inner_radius_m"], abs=report["tolerance_m"])
        assert high == pytest.approx(report["inner_radius_m"], abs=report["tolerance_m"])
    low, high = report["outer_node_R_range_m"]
    assert low == pytest.approx(report["outer_radius_m"], abs=report["tolerance_m"])
    assert high == pytest.approx(report["outer_radius_m"], abs=report["tolerance_m"])

    mutated_names = list(data.triangle_domain_names)
    pml_index = next(index for index, name in enumerate(mutated_names) if name == "air_pml_front")
    mutated_names[pml_index] = "air_front_free"
    with pytest.raises(ValueError, match="non_pml_domain_crosses_inner"):
        validate_pml_geometry(
            data.points_rz,
            data.mesh.t.T,
            mutated_names,
            data.facet_edges,
            data.config.raw["geometry"]["pml_inner_radius_m"],
            data.config.raw["geometry"]["pml_thickness_m"],
        )


@pytest.mark.parametrize(
    ("level", "relative_limit"),
    [("L1", 0.02), ("L2", 0.01)],
)
def test_sealed_b_analytic_limit_at_two_low_frequencies(b_models, level, relative_limit):
    model = b_models[level]
    target_q = np.pi * 0.045**2
    max_kl = 0.0
    for frequency in (10.0, 40.0):
        result = model.solve(frequency)
        max_kl = max(
            max_kl,
            2.0 * np.pi * frequency * 0.203718327157626 / model.parameters.c0_m_s,
        )
        q_error = abs(abs(result.q_into_cavity_m3_s) - target_q) / target_q
        assert q_error < 0.005
        assert result.relative_impedance_error < relative_limit
        assert result.cavity_mean_pressure_Pa == pytest.approx(
            result.analytic_limit.mean_pressure_Pa,
            rel=relative_limit,
        )
        assert np.imag(result.z_box_Pa_s_m3) < 0.0
        assert result.cavity_real_impedance_ratio < 1.0e-8
        assert result.residual_relative < 1.0e-7
    assert max_kl <= 0.2


def test_wrong_case_and_wrong_physical_tag_are_rejected(b_meshes, tmp_path):
    with pytest.raises(ValueError, match="sealed case B only"):
        ReferencePrescribedVelocityAcoustics.from_files(
            b_meshes["L0"],
            B_CONFIG,
            case_id="A",
        )

    wrong_tag = tmp_path / "wrong_air_cavity_tag.msh"

    def mutate(fields):
        fields["air_cavity"][0] = 9999

    _write_msh22_field_variant(b_meshes["L0"], wrong_tag, mutate)
    with pytest.raises(ValueError, match="physical_group_contract_exact"):
        load_reference_pressure_mesh(wrong_tag, B_CONFIG)


def test_loader_enforces_half_percent_cavity_volume_contract(b_meshes, tmp_path):
    source = b_meshes["L0"]
    variant = tmp_path / "cavity_volume_outside_half_percent.msh"
    source_mesh = meshio.read(source)
    cavity_tag = int(np.asarray(source_mesh.field_data["air_cavity"]).reshape(-1)[0])
    triangle_tags = np.asarray(
        source_mesh.cell_data_dict["gmsh:physical"]["triangle"],
        dtype=np.int64,
    )
    cavity_nodes = np.unique(
        np.asarray(source_mesh.cells_dict["triangle"], dtype=np.int64)[triangle_tags == cavity_tag]
    )

    def mutate_points(points, _mesh):
        points[cavity_nodes, 1] *= 1.0075

    _write_msh22_field_variant(source, variant, lambda _fields: None, mutate_points)
    with pytest.raises(ValueError, match="cavity volume"):
        load_reference_pressure_mesh(variant, B_CONFIG)
