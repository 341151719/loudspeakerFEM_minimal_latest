from __future__ import annotations

import math
from pathlib import Path

import pytest

from loudspeaker_axisym_fem.enclosure_validation import (
    amplitude_difference_db,
    fit_db_per_decade,
    make_temporary_geometry_config,
    phase_difference_deg,
    scan_reference_cases,
    write_validation_outputs,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "enclosures"
A_CONFIG = CONFIG_DIR / "open_back.json"
B_CONFIG = CONFIG_DIR / "sealed_lossless.json"


@pytest.fixture(scope="module")
def ab_base_report(tmp_path_factory):
    root = tmp_path_factory.mktemp("stage3b2_ab_base")
    return scan_reference_cases(
        {"A": A_CONFIG, "B": B_CONFIG},
        ("L0", "L1", "L2"),
        (10.0, 20.0, 50.0, 100.0),
        mesh_dir=root / "meshes",
    )


@pytest.fixture(scope="module")
def ab_low_frequency_report(tmp_path_factory):
    root = tmp_path_factory.mktemp("stage3b2_ab_low")
    return scan_reference_cases(
        {"A": A_CONFIG, "B": B_CONFIG},
        ("L1",),
        (5.0, 10.0, 20.0),
        mesh_dir=root / "meshes",
    )


def test_far_field_dipole_is_separate_from_actual_one_m_near_field(ab_low_frequency_report):
    comparison = ab_low_frequency_report["case_comparison"]["L1"]
    slope = comparison["far_field_slope_dB_per_decade"]
    actual_slope = comparison["actual_1m_slope_dB_per_decade"]
    assert 18.0 <= slope <= 22.0
    assert actual_slope < 18.0
    assert ab_low_frequency_report["acceptance"]["far_field_dipole_gate"] is True

    for row in ab_low_frequency_report["rows"]:
        actual = row["actual_pressure_1m"]
        far = row["far_field_normalized_to_1m"]
        assert actual["field_regime"] == "near_field"
        assert actual["asymptote_eligible"] is False
        assert actual["asymptote_note"]
        assert far["kernel"] == "closed HK exact kernel"
        assert far["mirror"] is False
        assert far["outside_hk"] is True
        assert far["kR_eval"] >= 20.0
        assert far["R_eval_m"] > row["cavity_volume_m3"] ** (1.0 / 3.0)
        assert far["is_actual_1m_pressure"] is False


def test_base_hierarchy_and_physical_acceptance(ab_base_report):
    assert len(ab_base_report["rows"]) == 24
    assert all(row["status"] == "pass" for row in ab_base_report["rows"])
    assert all(row["audit_status"] == "pass" for row in ab_base_report["rows"])
    assert len(ab_base_report["grid_convergence"]) == 8
    assert all(
        item["passes_far_field_grid_gate"]
        for item in ab_base_report["grid_convergence"]
    )
    assert all(
        item["l2_power_balance_relative"] < 0.02
        and item["l2_volume_velocity_error_relative"] < 0.005
        for item in ab_base_report["grid_convergence"]
    )
    assert ab_base_report["acceptance"]["l0_is_final_conclusion"] is False


def test_alpha_variants_are_passive_and_stable(tmp_path_factory):
    root = tmp_path_factory.mktemp("stage3b2_alpha")
    report = scan_reference_cases(
        {"A": A_CONFIG},
        ("L1",),
        (20.0, 50.0, 100.0),
        mesh_dir=root / "meshes",
        pml_alphas=(2.0, 4.0, 8.0, 12.0),
    )
    assert len(report["rows"]) == 12
    assert all(row["pml_passivity_status"] == "pass" for row in report["rows"])
    assert all(row["pin_W"] > 0.0 and row["phk_W"] > 0.0 for row in report["rows"])
    assert len(report["alpha_comparison"]) == 9
    assert all(
        item["passive"]
        and abs(item["relative_to_alpha4_dB"]) < 0.2
        and abs(item["relative_to_alpha4_phase_deg"]) < 2.0
        for item in report["alpha_comparison"]
    )


def test_temporary_geometry_variants_are_audited_and_stable(tmp_path_factory):
    root = tmp_path_factory.mktemp("stage3b2_geometry")
    base = scan_reference_cases(
        {"A": A_CONFIG},
        ("L1",),
        (20.0, 50.0, 100.0),
        mesh_dir=root / "base_meshes",
    )
    base_by_frequency = {row["frequency_Hz"]: row for row in base["rows"]}
    variants = {
        "R0_0.35": {"pml_inner_radius_m": 0.35},
        "R0_0.50": {"pml_inner_radius_m": 0.50},
        "R0_0.70": {"pml_inner_radius_m": 0.70},
        "t_0.1": {"pml_thickness_m": 0.10},
        "t_0.2": {"pml_thickness_m": 0.20},
        "t_0.3": {"pml_thickness_m": 0.30},
    }
    for name, values in variants.items():
        config = make_temporary_geometry_config(
            A_CONFIG,
            root / name / "configs",
            variant=name,
            **values,
        )
        report = scan_reference_cases(
            {"A": config},
            ("L1",),
            (20.0, 50.0, 100.0),
            mesh_dir=root / name / "meshes",
        )
        assert all(row["audit_status"] == "pass" for row in report["rows"])
        for row in report["rows"]:
            frequency = row["frequency_Hz"]
            current = row["far_field_normalized_to_1m"]["pressure_ff_1m_Pa"]
            reference = base_by_frequency[frequency]["far_field_normalized_to_1m"][
                "pressure_ff_1m_Pa"
            ]
            assert abs(amplitude_difference_db(
                complex(current["real"], current["imag"]),
                complex(reference["real"], reference["imag"]),
            )) < 0.2
            assert abs(phase_difference_deg(
                complex(current["real"], current["imag"]),
                complex(reference["real"], reference["imag"]),
            )) < 2.0


def test_validation_helpers_and_stable_outputs(tmp_path):
    assert amplitude_difference_db(2.0 + 0.0j, 1.0 + 0.0j) == pytest.approx(
        20.0 * math.log10(2.0)
    )
    assert phase_difference_deg(1.0j, 1.0) == pytest.approx(90.0)
    assert fit_db_per_decade((5.0, 10.0, 20.0), (0.25, 0.5, 1.0)) == pytest.approx(
        20.0
    )
    report = {
        "rows": [],
        "acceptance": {},
    }
    write_validation_outputs(report, tmp_path / "report.json", tmp_path / "report.csv")
    assert (tmp_path / "report.json").read_text(encoding="utf-8").endswith("\n")
    assert (tmp_path / "report.csv").read_text(encoding="utf-8").splitlines() == [
        "case,level,frequency_Hz,pml_alpha,status,pressure_dof_count,pressure_triangle_count,mesh_sha256,actual_1m_rms_spl_dB,actual_1m_phase_deg,actual_1m_field_regime,far_field_R_eval_m,far_field_kR_eval,far_field_rms_spl_dB,far_field_phase_deg,far_field_pressure_real_Pa,far_field_pressure_imag_Pa,pin_W,phk_W,pml_discrete_absorption_W,power_balance_relative,volume_velocity_error_relative,solve_wall_s"
    ]
