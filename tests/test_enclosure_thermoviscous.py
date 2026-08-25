from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.special import jve

from loudspeaker_axisym_fem.thermoviscous_boundaries import (
    HARMONIC_CONVENTION,
    ThermoviscousAirProperties,
    assess_bli_applicability,
    bli_bilinear_coefficients,
    bli_dissipation,
    boundary_layer_thicknesses,
    circular_duct_wide_asymptotic,
    circular_duct_zwikker_kosten,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "enclosures" / "sealed_thermoviscous.json"
FREQUENCIES_HZ = (10.0, 100.0, 1000.0)


@pytest.fixture(scope="module")
def raw_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def air() -> ThermoviscousAirProperties:
    return ThermoviscousAirProperties.from_config(CONFIG)


def _manual_layers(raw: dict, frequency_Hz: float) -> tuple[float, float]:
    props = raw["air"]
    omega = 2.0 * math.pi * frequency_Hz
    delta_v = math.sqrt(
        2.0 * props["dynamic_viscosity_Pa_s"]
        / (props["rho0_kg_m3"] * omega)
    )
    delta_t = math.sqrt(
        2.0 * props["thermal_conductivity_W_mK"]
        / (
            props["rho0_kg_m3"]
            * props["heat_capacity_cp_J_kgK"]
            * omega
        )
    )
    return delta_v, delta_t


@pytest.mark.parametrize("frequency_Hz", FREQUENCIES_HZ)
def test_boundary_layers_match_independent_formula_and_are_positive(
    air: ThermoviscousAirProperties,
    raw_config: dict,
    frequency_Hz: float,
) -> None:
    measured = boundary_layer_thicknesses(air, frequency_Hz)
    expected_v, expected_t = _manual_layers(raw_config, frequency_Hz)
    assert measured.delta_v_m == pytest.approx(expected_v, rel=2.0e-14)
    assert measured.delta_t_m == pytest.approx(expected_t, rel=2.0e-14)
    assert measured.delta_v_m > 0.0
    assert measured.delta_t_m > 0.0


def test_boundary_layers_follow_inverse_square_root_frequency(
    air: ThermoviscousAirProperties,
) -> None:
    rows = [boundary_layer_thicknesses(air, value) for value in FREQUENCIES_HZ]
    expected_ratio = math.sqrt(10.0)
    for first, second in zip(rows[:-1], rows[1:], strict=True):
        assert first.delta_v_m / second.delta_v_m == pytest.approx(
            expected_ratio, rel=2.0e-14
        )
        assert first.delta_t_m / second.delta_t_m == pytest.approx(
            expected_ratio, rel=2.0e-14
        )


def test_phase4a_numeric_checkpoint_matches_independent_reference_values(
    air: ThermoviscousAirProperties,
) -> None:
    expected_layers = (
        (10.0, 6.92419735707e-4, 8.23042498938e-4),
        (100.0, 2.18962346169e-4, 2.60268890776e-4),
        (1000.0, 6.92419735707e-5, 8.23042498938e-5),
    )
    expected_ratios = (
        (1.0102156606284267 - 0.010320616450896151j,
         1.0102163673528235 - 0.010216367352823535j),
        (1.003230676754063 - 0.003241075853873761j,
         1.0032306990247908 - 0.0032306990247907424j),
        (1.001021636031806 - 0.0010226728978560908j,
         1.0010216367352824 - 0.0010216367352823533j),
    )
    for (frequency, delta_v, delta_t), (exact_ratio, wide_ratio) in zip(
        expected_layers, expected_ratios, strict=True
    ):
        layers = boundary_layer_thicknesses(air, frequency)
        exact = circular_duct_zwikker_kosten(air, frequency, 0.05)
        wide = circular_duct_wide_asymptotic(air, frequency, 0.05)
        assert layers.delta_v_m == pytest.approx(delta_v, rel=6.0e-13)
        assert layers.delta_t_m == pytest.approx(delta_t, rel=6.0e-13)
        assert exact.wavenumber_ratio == pytest.approx(exact_ratio, rel=2.0e-13)
        assert wide.wavenumber_ratio == pytest.approx(wide_ratio, rel=2.0e-13)


def test_case_c_applicability_ratios_and_routes_are_independent(
    raw_config: dict,
) -> None:
    report = assess_bli_applicability(CONFIG, FREQUENCIES_HZ)
    assert report["case"] == "sealed_thermoviscous"
    assert report["demonstrator"] is True
    assert report["status"] == "pass"
    assert report["route"] == "BLI"
    assert report["conservative_curvature_radius_m"] == pytest.approx(0.045)

    minimum_gap = raw_config["thermoviscous"]["applicability"]["minimum_gap_m"]
    radius = min(
        raw_config["geometry"]["inner_radius_m"],
        raw_config["geometry"]["driver_radius_m"],
    )
    for row in report["frequencies"]:
        delta_v, delta_t = _manual_layers(raw_config, row["frequency_Hz"])
        delta = max(delta_v, delta_t)
        assert row["max_delta_over_gap"] == pytest.approx(delta / minimum_gap)
        assert row["two_max_delta_over_gap"] == pytest.approx(
            2.0 * delta / minimum_gap
        )
        assert row["max_delta_over_radius"] == pytest.approx(delta / radius)
        assert row["boundary_layers_overlap"] is False
        assert row["route"] == "BLI"

    outside = assess_bli_applicability(CONFIG, (5.0, 1001.0))
    assert outside["status"] == "fail"
    assert outside["route"] == "reject"
    assert {row["route"] for row in outside["frequencies"]} == {"reject"}


def test_bad_curvature_routes_to_full_tv_not_circular_or_slit_nra() -> None:
    report = assess_bli_applicability(
        CONFIG,
        (10.0,),
        local_curvature_radii_m={"deliberately_tight_curve": 1.0e-4},
        cross_section_kind="circular_duct",
    )
    assert report["route"] == "full-TV"
    assert report["status"] == "fail"
    assert report["frequencies"][0]["curvature_ratio_pass"] is False
    assert report["route"] != "NRA"


@pytest.mark.parametrize("frequency_Hz", FREQUENCIES_HZ)
def test_bli_coefficients_match_hand_weak_form_and_are_passive(
    air: ThermoviscousAirProperties,
    raw_config: dict,
    frequency_Hz: float,
) -> None:
    coeff = bli_bilinear_coefficients(air, frequency_Hz)
    delta_v, delta_t = _manual_layers(raw_config, frequency_Hz)
    omega = 2.0 * math.pi * frequency_Hz
    rho = raw_config["air"]["rho0_kg_m3"]
    bulk = rho * raw_config["air"]["c0_m_s"] ** 2
    gamma = raw_config["air"]["gamma"]
    expected_v = -delta_v / (rho * (1.0 + 1.0j))
    expected_t = -(omega**2 * (gamma - 1.0) * delta_t) / (
        bulk * (1.0 + 1.0j)
    )
    assert coeff.viscous_tangential_gradient_m4_kg == pytest.approx(expected_v)
    assert coeff.thermal_pressure_m2_kg == pytest.approx(expected_t)
    assert coeff.viscous_tangential_gradient_m4_kg.imag > 0.0
    assert coeff.thermal_pressure_m2_kg.imag > 0.0
    assert coeff.passive
    assert HARMONIC_CONVENTION == "exp(+i*omega*t)"


@pytest.mark.parametrize("frequency_Hz", FREQUENCIES_HZ)
def test_dissipation_quadratics_are_nonnegative_and_independently_computed(
    air: ThermoviscousAirProperties,
    frequency_Hz: float,
) -> None:
    pressure = np.array([1.0 + 2.0j, 2.0 - 1.0j])
    gradient = np.array([[2.0, -1.0], [-1.0, 2.0]])
    mass = np.array([[3.0, 0.25], [0.25, 1.5]])
    coeff = bli_bilinear_coefficients(air, frequency_Hz)
    result = bli_dissipation(pressure, gradient, mass, coeff)

    gradient_q = float(np.real(np.conjugate(pressure) @ gradient @ pressure))
    mass_q = float(np.real(np.conjugate(pressure) @ mass @ pressure))
    omega = 2.0 * math.pi * frequency_Hz
    expected_v = coeff.viscous_tangential_gradient_m4_kg.imag * gradient_q / (
        2.0 * omega
    )
    expected_t = coeff.thermal_pressure_m2_kg.imag * mass_q / (2.0 * omega)
    assert result.viscous_quadratic == pytest.approx(gradient_q)
    assert result.thermal_quadratic == pytest.approx(mass_q)
    assert result.P_visc_W == pytest.approx(expected_v)
    assert result.P_thermal_W == pytest.approx(expected_t)
    assert result.P_total_W == pytest.approx(expected_v + expected_t)
    assert result.P_visc_W > 0.0
    assert result.P_thermal_W > 0.0
    assert result.P_total_W > 0.0
    assert result.passive


def _manual_circular_ratio(raw: dict, frequency_Hz: float, radius_m: float) -> complex:
    delta_v, delta_t = _manual_layers(raw, frequency_Hz)
    gamma = raw["air"]["gamma"]
    z_v = (1.0 - 1.0j) * radius_m / delta_v
    z_t = (1.0 - 1.0j) * radius_m / delta_t
    # Write the COMSOL circular-field equation independently here.  Exponential
    # scaling cancels in each same-argument ratio and prevents overflow at
    # 1000 Hz; the production function is not called to form this expectation.
    upsilon_v = -jve(2, z_v) / jve(0, z_v)
    upsilon_t = -jve(2, z_t) / jve(0, z_t)
    ratio = complex(np.sqrt((gamma - (gamma - 1.0) * upsilon_t) / upsilon_v))
    return -ratio if ratio.real < 0.0 else ratio


@pytest.mark.parametrize("frequency_Hz", FREQUENCIES_HZ)
def test_circular_exact_and_wide_asymptotic_match_independent_equations(
    air: ThermoviscousAirProperties,
    raw_config: dict,
    frequency_Hz: float,
) -> None:
    radius = 0.05
    exact = circular_duct_zwikker_kosten(air, frequency_Hz, radius)
    wide = circular_duct_wide_asymptotic(air, frequency_Hz, radius)
    manual_exact = _manual_circular_ratio(raw_config, frequency_Hz, radius)
    delta_v, delta_t = _manual_layers(raw_config, frequency_Hz)
    manual_wide = 1.0 + (1.0 - 1.0j) * (
        delta_v + (raw_config["air"]["gamma"] - 1.0) * delta_t
    ) / (2.0 * radius)
    assert exact.wavenumber_ratio == pytest.approx(manual_exact, rel=2.0e-12)
    assert wide.wavenumber_ratio == pytest.approx(manual_wide, rel=2.0e-14)
    assert exact.wavenumber_1_m.imag <= 0.0
    assert exact.attenuation_Np_m == pytest.approx(-exact.wavenumber_1_m.imag)
    assert exact.attenuation_Np_m >= 0.0


def test_circular_wide_limit_error_converges() -> None:
    air = ThermoviscousAirProperties.from_config(CONFIG)
    errors = []
    for radius in (0.005, 0.02, 0.08):
        exact = circular_duct_zwikker_kosten(air, 100.0, radius)
        wide = circular_duct_wide_asymptotic(air, 100.0, radius)
        errors.append(
            abs(exact.wavenumber_ratio - wide.wavenumber_ratio)
            / abs(exact.wavenumber_ratio)
        )
    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < 1.0e-5


def test_viscosity_and_conductivity_increase_their_loss_contributions(
    air: ThermoviscousAirProperties,
) -> None:
    base = bli_bilinear_coefficients(air, 100.0)
    more_mu = bli_bilinear_coefficients(
        air.with_changes(dynamic_viscosity_Pa_s=4.0 * air.dynamic_viscosity_Pa_s),
        100.0,
    )
    more_kappa = bli_bilinear_coefficients(
        air.with_changes(
            thermal_conductivity_W_mK=4.0 * air.thermal_conductivity_W_mK
        ),
        100.0,
    )
    assert more_mu.viscous_tangential_gradient_m4_kg.imag == pytest.approx(
        2.0 * base.viscous_tangential_gradient_m4_kg.imag
    )
    assert more_mu.thermal_pressure_m2_kg.imag == pytest.approx(
        base.thermal_pressure_m2_kg.imag
    )
    assert more_kappa.thermal_pressure_m2_kg.imag == pytest.approx(
        2.0 * base.thermal_pressure_m2_kg.imag
    )
    assert more_kappa.viscous_tangential_gradient_m4_kg.imag == pytest.approx(
        base.viscous_tangential_gradient_m4_kg.imag
    )

    base_k = circular_duct_zwikker_kosten(air, 100.0, 0.05)
    mu_k = circular_duct_zwikker_kosten(
        air.with_changes(dynamic_viscosity_Pa_s=4.0 * air.dynamic_viscosity_Pa_s),
        100.0,
        0.05,
    )
    kappa_k = circular_duct_zwikker_kosten(
        air.with_changes(
            thermal_conductivity_W_mK=4.0 * air.thermal_conductivity_W_mK
        ),
        100.0,
        0.05,
    )
    assert mu_k.attenuation_Np_m > base_k.attenuation_Np_m
    assert kappa_k.attenuation_Np_m > base_k.attenuation_Np_m


def test_loss_scale_zero_is_exact_and_limit_is_continuous(
    air: ThermoviscousAirProperties,
) -> None:
    pressure = np.array([1.0 + 0.5j, -0.25j])
    identity = np.eye(2)
    zero_coeff = bli_bilinear_coefficients(air, 100.0, loss_scale=0.0)
    zero_power = bli_dissipation(pressure, identity, identity, zero_coeff)
    zero_duct = circular_duct_zwikker_kosten(air, 100.0, 0.05, loss_scale=0.0)
    assert zero_coeff.viscous_tangential_gradient_m4_kg == 0.0j
    assert zero_coeff.thermal_pressure_m2_kg == 0.0j
    assert zero_power.P_total_W == 0.0
    assert zero_duct.wavenumber_ratio == 1.0 + 0.0j
    assert zero_duct.attenuation_Np_m == 0.0

    errors = [
        abs(
            circular_duct_zwikker_kosten(
                air, 100.0, 0.05, loss_scale=scale
            ).wavenumber_ratio
            - 1.0
        )
        for scale in (1.0, 0.1, 0.01)
    ]
    assert errors[0] > errors[1] > errors[2] > 0.0


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("rho0_kg_m3", -1.0),
        ("c0_m_s", 0.0),
        ("dynamic_viscosity_Pa_s", -1.0e-5),
        ("thermal_conductivity_W_mK", 0.0),
        ("heat_capacity_cp_J_kgK", -1.0),
        ("gamma", 1.0),
    ),
)
def test_invalid_air_parameters_are_rejected(
    air: ThermoviscousAirProperties, field: str, value: float
) -> None:
    with pytest.raises(ValueError):
        replace(air, **{field: value})


def test_invalid_frequency_radius_scale_and_quadratic_are_rejected(
    air: ThermoviscousAirProperties,
) -> None:
    with pytest.raises(ValueError, match="frequency_Hz"):
        boundary_layer_thicknesses(air, 0.0)
    with pytest.raises(ValueError, match="loss_scale"):
        boundary_layer_thicknesses(air, 100.0, loss_scale=-1.0)
    with pytest.raises(ValueError, match="radius_m"):
        circular_duct_zwikker_kosten(air, 100.0, 0.0)
    coeff = bli_bilinear_coefficients(air, 100.0)
    with pytest.raises(ValueError, match="Hermitian"):
        bli_dissipation(
            np.array([1.0, 1.0j]),
            np.array([[1.0, 2.0], [0.0, 1.0]]),
            np.eye(2),
            coeff,
        )
