from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
FR10 = ROOT / "fr10_full360_cyclic"
sys.path.insert(0, str(FR10))

import cyclic_full360_solver as cyclic  # noqa: E402
import rocking_frequency as rocking  # noqa: E402
import cli  # noqa: E402


@pytest.fixture(scope="module")
def sector_model():
    return cyclic.build_sector_model(cyclic.load_cfg())


def test_unit_rocking_load_has_unit_transverse_moment(sector_model):
    force, moment = rocking.unit_rocking_moment_force(sector_model)
    assert force.shape == (sector_model["Nd"],)
    assert math.isclose(np.linalg.norm(moment[:2]), 1.0, rel_tol=1e-12)
    assert abs(moment[2]) < 1e-12


def test_coil_tilt_fit_recovers_small_rotation(sector_model):
    theta_x = 2.5e-4 - 1.0e-4j
    # A single k=1 Bloch class is one circular member of the rocking doublet;
    # its two Cartesian tilt components therefore differ by 90 degrees.
    theta_y = 1j * theta_x
    displacement = np.zeros((len(sector_model["P"]), 3), complex)
    x = sector_model["P"][:, 0]
    y = sector_model["P"][:, 1]
    # Construct the sector values of the k=1 field. Its cyclic continuation is
    # exactly the same global linear tilt field in all four quadrants.
    displacement[:, 2] = theta_x * y - theta_y * x
    fitted = rocking.extract_coil_tilt(sector_model, displacement, kclass=1)
    np.testing.assert_allclose(fitted["tilt_x_rad"], theta_x, rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(fitted["tilt_y_rad"], theta_y, rtol=1e-10, atol=1e-13)
    assert fitted["fit_relative_residual"] < 1e-10


def test_root_cause_frequency_signatures():
    cfg = {
        "mass_unbalance_kg_m": 2e-6,
        "stiffness_coupling_N": 3.0,
        "bl_moment_Nm_per_A": 4e-3,
    }
    displacement = 2e-4 + 1e-4j
    current = 0.2 - 0.1j
    result = rocking.root_cause_moments(100.0, displacement, current, cfg)
    omega = 2 * math.pi * 100.0
    assert result["mass_Nm"] == pytest.approx(omega**2 * 2e-6 * displacement)
    assert result["stiffness_Nm"] == pytest.approx(-3.0 * displacement)
    assert result["bl_Nm"] == pytest.approx(4e-3 * current)
    assert result["total_Nm"] == pytest.approx(
        result["mass_Nm"] + result["stiffness_Nm"] + result["bl_Nm"]
    )


def test_rocking_config_declares_diagnostic_scope():
    cfg = rocking.load_rocking_config()
    assert cfg["analysis_kind"] == "frequency_domain_weak_asymmetry_rocking_modes"
    assert cfg["root_causes"]["mass_unbalance_kg_m"] > 0
    assert "not measured" in cfg["root_causes"]["parameter_note"]


def test_cli_exposes_rocking_command():
    args = cli.build_parser().parse_args(["fr10-rocking", "--freq", "300"])
    assert args.func is cli.cmd_fr10_rocking
    assert args.freq == [300.0]
