from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pytest

from loudspeaker_axisym_fem.enclosure_acoustics import (
    ReferencePrescribedVelocityAcoustics,
    axisymmetric_volume,
)
from loudspeaker_axisym_fem.enclosure_geometry import (
    BOUNDARY_PHYSICAL_TAGS,
    generate_reference_mesh,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "enclosures"
B_CONFIG = CONFIG_DIR / "sealed_lossless.json"
C_CONFIG = CONFIG_DIR / "sealed_thermoviscous.json"
REPRESENTATIVE_FREQUENCIES_HZ = (10.0, 100.0, 500.0, 1000.0)
BOUNDARY_LOCAL_REFINEMENT_LEVELS = 3


@pytest.fixture(scope="module")
def sealed_meshes(tmp_path_factory):
    directory = tmp_path_factory.mktemp("enclosure_thermoviscous_fem")
    paths = {}
    for level in ("L1", "L2"):
        path = directory / f"BC_{level}.msh"
        generate_reference_mesh(C_CONFIG, level, path)
        paths[level] = path
    return paths


@pytest.fixture(scope="module")
def models(sealed_meshes):
    result = {}
    for level, path in sealed_meshes.items():
        result[("B", level)] = ReferencePrescribedVelocityAcoustics.from_files(
            path,
            B_CONFIG,
            pressure_boundary_local_refinements=BOUNDARY_LOCAL_REFINEMENT_LEVELS,
        )
        result[("C0", level)] = ReferencePrescribedVelocityAcoustics.from_files(
            path,
            C_CONFIG,
            loss_scale=0.0,
            pressure_boundary_local_refinements=BOUNDARY_LOCAL_REFINEMENT_LEVELS,
        )
        result[("C", level)] = ReferencePrescribedVelocityAcoustics.from_files(
            path,
            C_CONFIG,
            pressure_boundary_local_refinements=BOUNDARY_LOCAL_REFINEMENT_LEVELS,
        )
    return result


@pytest.fixture(scope="module")
def representative_results(models):
    results = {}
    for level in ("L1", "L2"):
        for frequency in REPRESENTATIVE_FREQUENCIES_HZ:
            for case in ("B", "C0", "C"):
                results[(case, level, frequency)] = models[(case, level)].solve(
                    frequency
                )
    return results


def _complex_nrmse(reference: np.ndarray, candidate: np.ndarray) -> float:
    return float(
        np.linalg.norm(candidate - reference)
        / max(np.linalg.norm(reference), 1.0e-30)
    )


def _cavity_peak(model, result) -> float:
    dofs = model.mesh_data.component_dofs[model.mesh_data.cavity_component]
    return float(np.max(np.abs(result.pressure[dofs])))


def test_c_loader_selects_only_unique_cavity_rigid_wall_facets(models) -> None:
    data = models[("C", "L1")].mesh_data
    report = data.cavity_bli_boundary_report
    expected_groups = {
        "cabinet_front_wall",
        "cabinet_side_wall",
        "comparison_equalizer_face",
        "driver_side_wall",
        "reference_planar_piston_back",
    }
    assert data.case_id == "C"
    assert report["status"] == "pass"
    assert set(report["selected_group_names"]) == expected_groups
    assert report["duplicate_selected_facet_count"] == 0
    assert report["reference_planar_piston_back_included"] is True
    assert report["axis_excluded"] is True
    assert report["pressure_pressure_excluded"] is True
    assert report["pml_hk_exterior_excluded"] is True
    assert report["excluded"]["axis"]["facet_count"] > 0
    assert report["excluded"]["axis"]["axisymmetric_area_m2"] == 0.0

    group_facets = data.cavity_bli_group_facets
    all_group_facets = np.concatenate([group_facets[name] for name in sorted(group_facets)])
    assert len(all_group_facets) == len(np.unique(all_group_facets))
    assert np.array_equal(np.sort(all_group_facets), data.cavity_bli_facets)
    assert len(all_group_facets) == report["selected_facet_count"]
    total_area = 0.0
    for name, row in report["physical_groups"].items():
        assert row["physical_tag"] == BOUNDARY_PHYSICAL_TAGS[name]
        assert row["physical_dimension"] == 1
        assert row["facet_count"] == len(group_facets[name])
        assert row["axisymmetric_area_m2"] > 0.0
        assert row["wall_condition"] == "isothermal_no_slip_BLI"
        assert all("air_cavity" in domains for domains in row["adjacent_domain_sets"])
        assert all(
            domain == "air_cavity" or domain.startswith("rigid_")
            for domains in row["adjacent_domain_sets"]
            for domain in domains
        )
        total_area += row["axisymmetric_area_m2"]
    assert total_area == pytest.approx(report["selected_axisymmetric_area_m2"])
    piston_area = report["physical_groups"]["reference_planar_piston_back"][
        "axisymmetric_area_m2"
    ]
    assert piston_area == pytest.approx(math.pi * 0.045**2, rel=1.0e-13)


def _axisymmetric_facet_area(model, facets: np.ndarray) -> float:
    edges = model.mesh_data.mesh.facets[:, np.asarray(facets, dtype=np.int64)]
    first = model.mesh_data.mesh.p[:, edges[0]]
    second = model.mesh_data.mesh.p[:, edges[1]]
    lengths = np.linalg.norm(second - first, axis=0)
    radii = 0.5 * (first[0] + second[0])
    return float(np.sum(2.0 * math.pi * radii * lengths))


def test_local_refinement_preserves_tags_volume_areas_traces_and_q(
    sealed_meshes, models
) -> None:
    for level in ("L1", "L2"):
        source = ReferencePrescribedVelocityAcoustics.from_files(
            sealed_meshes[level], C_CONFIG
        )
        refined = models[("C", level)]
        source_data = source.mesh_data
        data = refined.mesh_data
        assert data.pressure_boundary_local_refinement_levels == 3
        assert data.pressure_triangle_count > source_data.pressure_triangle_count
        assert data.cavity_volume_m3 == pytest.approx(
            source_data.cavity_volume_m3, rel=2.0e-13, abs=1.0e-14
        )
        direct_volume = axisymmetric_volume(
            data.mesh.p.T,
            data.mesh.t[:, data.cavity_triangle_indices].T,
        )
        assert direct_volume == pytest.approx(source_data.cavity_volume_m3, rel=2.0e-13)
        assert data.pml_geometry_report["passed"] is True
        assert data.hk_geometry_report["passed"] is True
        assert set(data.triangle_domain_names) == set(source_data.triangle_domain_names)
        source_tag_by_name = {
            name: int(tag)
            for name, tag in zip(
                source_data.triangle_domain_names,
                source_data.triangle_domain_tags,
                strict=True,
            )
        }
        for name, tag in zip(
            data.triangle_domain_names, data.triangle_domain_tags, strict=True
        ):
            assert int(tag) > 0
            assert int(tag) == source_tag_by_name[name]
        for name in data.line_facets:
            assert _axisymmetric_facet_area(
                refined, data.line_facets[name]
            ) == pytest.approx(
                _axisymmetric_facet_area(source, source_data.line_facets[name]),
                rel=2.0e-13,
                abs=1.0e-14,
            )
        assert _axisymmetric_facet_area(
            refined, data.cavity_bli_facets
        ) == pytest.approx(
            _axisymmetric_facet_area(source, source_data.cavity_bli_facets),
            rel=2.0e-13,
            abs=1.0e-14,
        )
        for trace_name in (
            "reference_planar_piston_front",
            "reference_planar_piston_back",
        ):
            source_trace = source_data.trace_metrics[trace_name]
            refined_trace = data.trace_metrics[trace_name]
            assert refined_trace["q_out_m3_s"] == pytest.approx(
                source_trace["q_out_m3_s"], rel=2.0e-13, abs=1.0e-14
            )
            assert refined_trace["q_into_m3_s"] == pytest.approx(
                source_trace["q_into_m3_s"], rel=2.0e-13, abs=1.0e-14
            )
        assert data.trace_metrics["reference_planar_piston_front"][
            "q_out_m3_s"
        ] == pytest.approx(-math.pi * 0.045**2, rel=2.0e-13)
        assert data.trace_metrics["reference_planar_piston_back"][
            "q_out_m3_s"
        ] == pytest.approx(math.pi * 0.045**2, rel=2.0e-13)


def test_bli_matrix_is_axisymmetric_symmetric_and_cavity_local(models) -> None:
    model = models[("C", "L1")]
    assembly = model.assemble(100.0)
    diagnostics = assembly.thermoviscous_diagnostics
    assert assembly.bli_matrix.nnz > 0
    assert diagnostics["axisymmetric_weight"] == "2*pi*r in FacetBasis forms"
    assert diagnostics["applicability"]["route"] == "BLI"
    assert diagnostics["edge_area_crosscheck_absolute_error_m2"] < 1.0e-12
    assert all(
        row["absolute_error_m2"] < 1.0e-12
        for row in diagnostics["physical_group_area_crosscheck"].values()
    )
    asymmetry = assembly.bli_matrix - assembly.bli_matrix.T
    assert not asymmetry.nnz or np.max(np.abs(asymmetry.data)) < 1.0e-11
    cavity_dofs = set(
        model.mesh_data.component_dofs[model.mesh_data.cavity_component].tolist()
    )
    rows, columns = assembly.bli_matrix.nonzero()
    assert set(rows).issubset(cavity_dofs)
    assert set(columns).issubset(cavity_dofs)
    pml_dofs = set(
        model.mesh_data.mesh.t[:, model.mesh_data.pml_triangle_indices]
        .reshape(-1)
        .tolist()
    )
    assert not (set(rows) & pml_dofs)


def test_b_and_c_zero_loss_same_mesh_are_complex_identical(
    models, representative_results
) -> None:
    for level in ("L1", "L2"):
        assert (
            models[("B", level)].mesh_data.source_sha256
            == models[("C0", level)].mesh_data.source_sha256
        )
        for frequency in REPRESENTATIVE_FREQUENCIES_HZ:
            b = representative_results[("B", level, frequency)]
            c0 = representative_results[("C0", level, frequency)]
            nrmse = _complex_nrmse(b.pressure, c0.pressure)
            print(f"BC_zero_loss_nrmse level={level} f={frequency:g}: {nrmse:.3e}")
            assert nrmse < 2.0e-3
            assert nrmse < 1.0e-12
            assert c0.thermoviscous_diagnostics["P_total_W"] == 0.0
            assert c0.z_box_Pa_s_m3 == pytest.approx(b.z_box_Pa_s_m3, rel=1.0e-12)


def test_c_representative_losses_and_power_closure_are_passive(
    representative_results,
) -> None:
    for level in ("L1", "L2"):
        for frequency in REPRESENTATIVE_FREQUENCIES_HZ:
            result = representative_results[("C", level, frequency)]
            diagnostics = result.thermoviscous_diagnostics
            closure = diagnostics["input_power_closure"]
            assert diagnostics["passivity_status"] == "pass"
            assert diagnostics["applicability"]["route"] == "BLI"
            assert diagnostics["P_visc_W"] >= 0.0
            assert diagnostics["P_thermal_W"] >= 0.0
            assert diagnostics["P_total_W"] > 0.0
            assert abs(
                diagnostics["P_total_W"]
                - diagnostics["P_visc_W"]
                - diagnostics["P_thermal_W"]
            ) <= diagnostics["matrix_independent_power_cross_tolerance_W"]
            assert abs(diagnostics["matrix_independent_power_cross_error_W"]) <= (
                diagnostics["matrix_independent_power_cross_tolerance_W"]
            )
            assert (
                diagnostics[
                    "matrix_independent_power_cross_absolute_tolerance_W"
                ]
                == 5.0e-9
            )
            assert (
                diagnostics["matrix_independent_power_cross_relative_tolerance"]
                <= 1.0e-6
            )
            expected_cross_tolerance = 5.0e-9 + 1.0e-6 * diagnostics[
                "matrix_independent_power_cross_scale_W"
            ]
            assert diagnostics[
                "matrix_independent_power_cross_tolerance_W"
            ] == pytest.approx(expected_cross_tolerance)
            assert closure["relative_to_input"] < 0.02 or abs(closure["residual_W"]) < 1.0e-10
            assert closure["pml_numerical_absorption_W"] > 0.0
            assert closure["bli_physical_loss_W"] > 0.0
            assert result.pml_diagnostics["discrete_absorption_power_W"] == pytest.approx(
                closure["pml_numerical_absorption_W"]
            )
            assert result.residual_relative < 1.0e-7
            if level == "L2":
                assert diagnostics["hk_crosscheck"]["relative_to_input"] < 0.02


def test_l1_l2_thermoviscous_convergence_meets_stage4b_gates(
    representative_results,
) -> None:
    maximum_loss_change = 0.0
    maximum_impedance_change_dB = 0.0
    maximum_spl_change_dB = 0.0
    for frequency in REPRESENTATIVE_FREQUENCIES_HZ:
        coarse = representative_results[("C", "L1", frequency)]
        fine = representative_results[("C", "L2", frequency)]
        p_coarse = coarse.thermoviscous_diagnostics["P_total_W"]
        p_fine = fine.thermoviscous_diagnostics["P_total_W"]
        loss_change = abs(p_fine - p_coarse) / p_fine
        impedance_change = abs(
            20.0
            * math.log10(
                abs(fine.z_box_Pa_s_m3) / abs(coarse.z_box_Pa_s_m3)
            )
        )
        spl_change = abs(
            fine.hk_diagnostics["axis_rms_spl_dB"]
            - coarse.hk_diagnostics["axis_rms_spl_dB"]
        )
        maximum_loss_change = max(maximum_loss_change, loss_change)
        maximum_impedance_change_dB = max(
            maximum_impedance_change_dB, impedance_change
        )
        maximum_spl_change_dB = max(maximum_spl_change_dB, spl_change)
        assert loss_change < 0.05
        assert impedance_change < 0.3
        assert spl_change < 0.3
    print(
        "stage4b_convergence "
        f"P={maximum_loss_change:.6%} "
        f"Z={maximum_impedance_change_dB:.6g}dB "
        f"SPL={maximum_spl_change_dB:.6g}dB"
    )


def test_viscous_and_thermal_component_switches_zero_only_their_terms(
    sealed_meshes,
) -> None:
    viscous_off = ReferencePrescribedVelocityAcoustics.from_files(
        sealed_meshes["L1"],
        C_CONFIG,
        viscous_loss_scale=0.0,
    ).solve(100.0)
    thermal_off = ReferencePrescribedVelocityAcoustics.from_files(
        sealed_meshes["L1"],
        C_CONFIG,
        thermal_loss_scale=0.0,
    ).solve(100.0)
    first = viscous_off.thermoviscous_diagnostics
    second = thermal_off.thermoviscous_diagnostics
    assert first["P_visc_W"] == 0.0
    assert first["P_thermal_W"] > 0.0
    assert first["P_total_W"] == pytest.approx(first["P_thermal_W"])
    assert second["P_thermal_W"] == 0.0
    assert second["P_visc_W"] > 0.0
    assert second["P_total_W"] == pytest.approx(second["P_visc_W"])


def test_loss_scale_increase_does_not_amplify_cavity_response(sealed_meshes) -> None:
    responses = []
    for scale in (0.0, 0.25, 0.5, 0.75, 1.0):
        model = ReferencePrescribedVelocityAcoustics.from_files(
            sealed_meshes["L1"], C_CONFIG, loss_scale=scale
        )
        result = model.solve(876.75)
        responses.append(_cavity_peak(model, result))
    assert all(
        later < earlier
        for earlier, later in zip(responses[:-1], responses[1:], strict=True)
    )


def test_local_first_cavity_mode_has_lower_finite_c_peak_and_q(models) -> None:
    frequencies = np.arange(873.0, 879.0001, 0.25)
    amplitudes = {"B": [], "C": []}
    for case in ("B", "C"):
        model = models[(case, "L1")]
        for frequency in frequencies:
            result = model.solve(float(frequency))
            amplitudes[case].append(_cavity_peak(model, result))
        amplitudes[case] = np.asarray(amplitudes[case])

    peaks = {}
    for case in ("B", "C"):
        values = amplitudes[case]
        peak_index = int(np.argmax(values))
        above_half_power = np.flatnonzero(values >= values[peak_index] / math.sqrt(2.0))
        bandwidth = (
            float(frequencies[above_half_power[-1]] - frequencies[above_half_power[0]])
            if len(above_half_power) > 1
            else 0.0
        )
        peaks[case] = {
            "frequency_Hz": float(frequencies[peak_index]),
            "amplitude_Pa": float(values[peak_index]),
            "bandwidth_Hz": bandwidth,
            "Q": math.inf if bandwidth == 0.0 else float(frequencies[peak_index] / bandwidth),
        }
    print(f"stage4b_first_cavity_mode={peaks}")
    assert peaks["C"]["amplitude_Pa"] < peaks["B"]["amplitude_Pa"]
    assert peaks["C"]["amplitude_Pa"] < 0.2 * peaks["B"]["amplitude_Pa"]
    assert peaks["C"]["frequency_Hz"] < peaks["B"]["frequency_Hz"]
    assert math.isinf(peaks["B"]["Q"])
    assert math.isfinite(peaks["C"]["Q"])
    assert peaks["C"]["Q"] < peaks["B"]["Q"]


def test_c_rejects_invalid_scales_and_out_of_applicability_band(
    sealed_meshes,
) -> None:
    with pytest.raises(ValueError, match=r"loss_scale.*\[0, 1\]"):
        ReferencePrescribedVelocityAcoustics.from_files(
            sealed_meshes["L1"], C_CONFIG, loss_scale=1.01
        )
    model = ReferencePrescribedVelocityAcoustics.from_files(
        sealed_meshes["L1"], C_CONFIG
    )
    with pytest.raises(ValueError, match="route=reject"):
        model.assemble(1001.0)
