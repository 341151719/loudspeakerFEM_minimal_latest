from __future__ import annotations

from pathlib import Path
import math

import pytest

from loudspeaker_axisym_fem.enclosure_acoustics import (
    DEFAULT_PML_ALPHA,
    EXPLICIT_PML_MODE,
    ReferencePrescribedVelocityAcoustics,
    TARGET_PML_MODE,
    pml_alpha_for_frequency,
    pml_operator_coefficients,
)
from loudspeaker_axisym_fem.enclosure_geometry import generate_reference_mesh


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "enclosures"
A_CONFIG = CONFIG_DIR / "open_back.json"
B_CONFIG = CONFIG_DIR / "sealed_lossless.json"


@pytest.fixture(scope="module")
def ab_models(tmp_path_factory):
    directory = tmp_path_factory.mktemp("enclosure_open_back_l1")
    models = {}
    for case_id, config in (("A", A_CONFIG), ("B", B_CONFIG)):
        mesh_path = directory / f"{case_id}_L1.msh"
        generate_reference_mesh(config, "L1", mesh_path)
        models[case_id] = ReferencePrescribedVelocityAcoustics.from_files(
            mesh_path,
            config,
        )
    return models


def _pressure_magnitude(result) -> float:
    value = result.hk_diagnostics["axis_pressure_1m_Pa"]
    return math.hypot(value["real"], value["imag"])


def test_a_open_back_and_b_sealed_components_closed_hk(ab_models):
    a = ab_models["A"].mesh_data
    b = ab_models["B"].mesh_data
    assert a.case_id == "A"
    assert b.case_id == "B"
    assert a.component_count == 1
    assert b.component_count == 2
    assert a.component_domains[a.cavity_component] != ("air_cavity",)
    assert b.component_domains[b.cavity_component] == ("air_cavity",)
    assert a.exterior_components == (a.cavity_component,)
    assert len(b.exterior_components) == 1
    for data in (a, b):
        assert data.hk_geometry_report["passed"] is True
        assert data.hk_geometry_report["mirror"] is False
        assert data.hk_geometry_report["front_edge_count"] > 0
        assert data.hk_geometry_report["rear_edge_count"] > 0
        assert data.hk_geometry_report["angular_coverage_rad"] == pytest.approx(
            [0.0, math.pi]
        )
        assert data.pml_geometry_report["passed"] is True


def test_pml_complete_operator_is_continuous_and_default_is_safe(ab_models):
    rho = ab_models["A"].parameters.rho0_kg_m3
    bulk = ab_models["A"].parameters.bulk_modulus_Pa
    coeff = pml_operator_coefficients(
        0.35,
        0.0,
        0.35,
        0.1,
        rho,
        bulk,
        alpha=DEFAULT_PML_ALPHA,
    )
    assert coeff["operator_gradient_radial"] == pytest.approx(1.0 / rho)
    assert coeff["operator_gradient_tangential"] == pytest.approx(1.0 / rho)
    assert coeff["operator_mass"] == pytest.approx(1.0 / bulk)

    assembly = ab_models["A"].assemble(20.0)
    assert ab_models["A"].parameters.pml_mode == EXPLICIT_PML_MODE
    assert ab_models["A"].parameters.pml_alpha == pytest.approx(DEFAULT_PML_ALPHA)
    assert assembly.pml_diagnostics["interface_operator_continuity"] is True
    assert assembly.pml_diagnostics["interface_operator_max_error"] < 1.0e-12


def test_safe_reference_a_b_low_frequency_power_and_flux(ab_models):
    values = {}
    for frequency in (20.0, 50.0, 100.0):
        values[frequency] = {}
        for case_id, model in ab_models.items():
            result = model.solve(frequency)
            values[frequency][case_id] = result
            assert result.pml_diagnostics["passivity_status"] == "pass"
            assert result.drive_power_into_fluid_W["total"] > 0.0
            assert result.input_power_from_rhs_W > 0.0
            assert result.hk_diagnostics["hk_flux_power_W"] > 0.0
            assert result.pml_diagnostics["discrete_absorption_power_W"] > 0.0
            assert abs(result.input_power_boundary_cross_error_W) < 1.0e-12
            front = result.front_back_traces["reference_planar_piston_front"]["q_out_m3_s"]
            back = result.front_back_traces["reference_planar_piston_back"]["q_out_m3_s"]
            assert front == pytest.approx(-back, rel=1.0e-13)
            assert abs(result.q_out_total_m3_s) < 0.005 * abs(front)

    ratios = [_pressure_magnitude(values[f]["A"]) / _pressure_magnitude(values[f]["B"]) for f in (20.0, 50.0, 100.0)]
    assert all(ratio < 1.0 for ratio in ratios)
    assert ratios[0] < ratios[1] < ratios[2]
    slope_db_per_decade = 20.0 * math.log10(ratios[-1] / ratios[0]) / math.log10(100.0 / 20.0)
    print(f"A_vs_B_low_frequency_slope_db_per_decade={slope_db_per_decade:.6f}")
    assert slope_db_per_decade > 0.0


def test_target_nepers_formula_and_unresolved_l1_failure(ab_models):
    frequency = 20.0
    target = 8.0
    exponent = 2
    c0 = ab_models["A"].parameters.c0_m_s
    thickness = ab_models["A"].parameters.pml_thickness_m
    expected = target * (exponent + 1) / (
        (2.0 * math.pi * frequency / c0) * thickness
    )
    assert pml_alpha_for_frequency(
        frequency,
        c0,
        thickness,
        target,
        exponent=exponent,
    ) == pytest.approx(expected, rel=1.0e-14)

    target_model = ReferencePrescribedVelocityAcoustics.from_files(
        ab_models["A"].mesh_data.path,
        A_CONFIG,
        pml_mode=TARGET_PML_MODE,
        target_attenuation_nepers=target,
    )
    unresolved = target_model.solve(frequency, allow_unresolved_pml=True)
    assert unresolved.pml_diagnostics["mode"] == TARGET_PML_MODE
    assert unresolved.pml_diagnostics["target_attenuation_nepers"] == target
    assert unresolved.pml_diagnostics["passivity_status"] == "unresolved"
    assert unresolved.pml_diagnostics["discrete_absorption_power_W"] < 0.0
    assert unresolved.pml_diagnostics["theoretical_outer_amplitude_factor"] == pytest.approx(
        math.exp(-target), rel=1.0e-14
    )
    with pytest.raises(RuntimeError, match="PML/passivity unresolved"):
        target_model.solve(frequency)
