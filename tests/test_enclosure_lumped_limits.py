from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]

from loudspeaker_axisym_fem.enclosure_models import (  # noqa: E402
    AirProperties,
    ClosedBox,
    PassiveRadiator,
    PassiveRadiatorBox,
    Port,
    VentedBox,
    acoustic_impedance_from_mechanical,
    loudspeaker_side_mechanical_impedance_from_acoustic,
)


AIR = AirProperties(rho0=1.2043175745358388, c0=343.2035820928282)


def test_sealed_box_is_a_compliance_and_legacy_loss_name_is_explicitly_a_leak():
    volume = 0.0061
    frequency = 17.0
    omega = 2.0 * math.pi * frequency
    box = ClosedBox(volume_m3=volume)

    compliance_hand = volume / (AIR.rho0 * AIR.c0**2)
    admittance = box.input_admittance(frequency, AIR)
    impedance = box.input_impedance(frequency, AIR)

    assert math.isclose(box.compliance(AIR), compliance_hand, rel_tol=1e-13)
    assert math.isclose(admittance.real, 0.0, abs_tol=1e-15)
    assert math.isclose(admittance.imag, omega * compliance_hand, rel_tol=1e-13)
    assert math.isclose(impedance.real, 0.0, abs_tol=1e-12)
    assert math.isclose(impedance.imag, -1.0 / (omega * compliance_hand), rel_tol=1e-13)

    leak_resistance = 2400.0
    legacy = ClosedBox(volume, loss_resistance_Pa_s_m3=leak_resistance)
    assert legacy.leak_resistance_Pa_s_m3 == leak_resistance
    assert legacy.loss_resistance_Pa_s_m3 == leak_resistance
    assert math.isclose(legacy.input_admittance(frequency, AIR).real, 1.0 / leak_resistance)


def test_vented_helmholtz_reference_uses_parallel_box_and_port_network():
    volume = 0.0061
    radius = 0.018
    length = 0.098
    end_correction_factor = 1.46
    port = Port(
        radius_m=radius,
        length_m=length,
        end_correction_radii=end_correction_factor,
    )
    model = VentedBox(ClosedBox(volume), port)

    area_hand = math.pi * radius**2
    effective_length_hand = length + end_correction_factor * radius
    mass_hand = AIR.rho0 * effective_length_hand / area_hand
    compliance_hand = volume / (AIR.rho0 * AIR.c0**2)
    frequency_hand = 1.0 / (2.0 * math.pi * math.sqrt(mass_hand * compliance_hand))

    assert math.isclose(model.helmholtz_frequency_Hz(AIR), frequency_hand, rel_tol=1e-13)
    network_admittance = model.input_admittance(frequency_hand, AIR)
    assert abs(network_admittance) < 1e-11

    low_frequency = 0.001
    low_impedance = model.input_impedance(low_frequency, AIR)
    assert math.isclose(
        (low_impedance / (1j * 2.0 * math.pi * low_frequency)).real,
        mass_hand,
        rel_tol=1e-8,
    )
    assert abs(
        (low_impedance / (1j * 2.0 * math.pi * low_frequency)).imag
    ) < 1e-6 * mass_hand


def test_port_mode_defaults_to_open_open_and_quarter_wave_is_explicit():
    port = Port(radius_m=0.018, length_m=0.098)
    leff = 0.098 + 1.46 * 0.018
    half_wave = AIR.c0 / (2.0 * leff)
    quarter_wave = AIR.c0 / (4.0 * leff)

    assert math.isclose(port.first_pipe_resonance_Hz(AIR), half_wave, rel_tol=1e-13)
    assert math.isclose(
        port.first_pipe_resonance_Hz(AIR, terminal_condition="closed-open"),
        quarter_wave,
        rel_tol=1e-13,
    )
    # The old positional boolean remains a compatibility spelling, but is not the default.
    assert math.isclose(port.first_pipe_resonance_Hz(AIR, True), quarter_wave, rel_tol=1e-13)
    with pytest.raises(ValueError, match="terminal"):
        port.first_pipe_resonance_Hz(AIR, terminal_condition="unknown")


def test_passive_radiator_resonance_and_mechanical_acoustic_conversion_are_independent():
    area = 0.0030
    mass = 0.012
    compliance = 0.00035
    resistance = 0.45
    radiator = PassiveRadiator(area, mass, compliance, resistance)

    frequency_hand = 1.0 / (2.0 * math.pi * math.sqrt(mass * compliance))
    omega = 2.0 * math.pi * frequency_hand
    impedance_hand = resistance + 1j * (omega * mass - 1.0 / (omega * compliance))
    mechanical = radiator.mechanical_impedance(frequency_hand)

    assert math.isclose(radiator.resonance_Hz(), frequency_hand, rel_tol=1e-13)
    assert abs(mechanical - impedance_hand) < 1e-12
    assert math.isclose(mechanical.real, resistance, rel_tol=1e-13)
    assert abs(mechanical.imag) < 1e-12

    # F/v = (p Sd)/(U/Sd) gives Za = Zm / Sd^2, independently of the model method.
    z_mechanical = 2.3 - 1.7j
    z_acoustic_hand = z_mechanical / (area**2)
    assert radiator.acoustic_impedance(83.0) == pytest.approx(
        radiator.mechanical_impedance(83.0) / (area**2)
    )
    assert acoustic_impedance_from_mechanical(z_mechanical, area) == pytest.approx(z_acoustic_hand)
    assert loudspeaker_side_mechanical_impedance_from_acoustic(z_acoustic_hand, area) == pytest.approx(
        z_mechanical
    )


def test_zero_loss_is_reactive_and_positive_damping_is_passive():
    frequency = 63.0
    port = Port(radius_m=0.018, length_m=0.098)
    lossless_pr = PassiveRadiator(0.003, 0.012, 0.00035, 0.0)
    lossy_pr = PassiveRadiator(0.003, 0.012, 0.00035, 0.45)

    assert abs(port.impedance(frequency, AIR).real) < 1e-15
    assert abs(lossless_pr.mechanical_impedance(frequency).real) < 1e-15
    assert lossy_pr.mechanical_impedance(frequency).real > 0.0

    velocity = 0.02 + 0.01j
    average_loss_W = 0.5 * lossy_pr.Rms_N_s_m * abs(velocity) ** 2
    assert average_loss_W > 0.0

    # In the PR box, low frequency compliance is the parallel sum of box air and PR Cms.
    volume = 0.0061
    box = ClosedBox(volume)
    pr_box = PassiveRadiatorBox(box, lossless_pr)
    low_frequency = 0.01
    low_admittance = pr_box.input_admittance(low_frequency, AIR)
    compliance_hand = volume / (AIR.rho0 * AIR.c0**2) + lossless_pr.Cms_m_N * lossless_pr.Sd_m2**2
    ratio = low_admittance / (1j * 2.0 * math.pi * low_frequency)
    assert math.isclose(ratio.real, compliance_hand, rel_tol=1e-8)
    assert abs(ratio.imag) < 1e-6 * compliance_hand


def test_pr_velocity_uses_pressure_force_and_area_once():
    radiator = PassiveRadiator(0.003, 0.012, 0.00035, 0.45)
    pressure = 1.2 - 0.4j
    frequency = 80.0
    expected_velocity = pressure * radiator.Sd_m2 / radiator.mechanical_impedance(frequency)
    model = PassiveRadiatorBox(ClosedBox(0.0061), radiator)

    assert model.radiator_velocity(frequency, pressure) == pytest.approx(expected_velocity)
