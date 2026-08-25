from pathlib import Path

import pytest

from loudspeaker_axisym_fem.production_wet_trace import (
    DEFAULT_MPHTXT,
    DEFAULT_PRODUCTION_MESH,
    extract_production_wet_traces,
)


ROOT = Path(__file__).resolve().parents[1]


def test_production_wet_trace_is_derived_from_common_edges_and_matches_contract_numbers():
    report = extract_production_wet_traces(
        ROOT / DEFAULT_PRODUCTION_MESH,
        mphtxt_path=ROOT / DEFAULT_MPHTXT,
    )

    assert report["status"] == "pass"
    assert report["labels"]["reference_name"] == "reference planar piston"
    assert report["labels"]["final_production_interface_ready"] is False
    assert report["comparison"]["mismatch"] is True
    assert report["comparison"]["reference_planar_piston"]["radius_m"] == pytest.approx(0.045)
    assert report["source"]["mesh_sha256"]
    assert report["source"]["mphtxt_sha256"]

    assert report["front"]["boundary_ids"] == [47, 92, 99, 102]
    assert report["rear"]["boundary_ids"] == [46, 91, 100, 101]
    assert report["front"]["axisymmetric_projected_area_2pi_r_abs_dr_m2"] == pytest.approx(
        0.0211011, rel=2.0e-5
    )
    assert report["rear"]["axisymmetric_projected_area_2pi_r_abs_dr_m2"] == pytest.approx(
        0.0203353, rel=2.0e-5
    )
    assert report["front"]["rotational_surface_area_2pi_r_ds_m2"] > report["front"]["axisymmetric_projected_area_2pi_r_abs_dr_m2"]
    assert report["rear"]["rotational_surface_area_2pi_r_ds_m2"] > report["rear"]["axisymmetric_projected_area_2pi_r_abs_dr_m2"]

    expected_pairs = {
        47: (21, 4),
        92: (3, 4),
        99: (25, 4),
        102: (25, 4),
        46: (21, 2),
        91: (3, 2),
        100: (25, 2),
        101: (25, 2),
    }
    rows = report["front"]["entities"] + report["rear"]["entities"]
    assert {row["boundary_id"] for row in rows} == set(expected_pairs)
    for row in rows:
        pair = row["domain_pairs"]
        assert len(pair) == 1
        assert (
            pair[0]["structural_domain"],
            pair[0]["acoustic_domain"],
        ) == expected_pairs[row["boundary_id"]]
        assert row["edge_count"] > 0
        assert row["length_2d_m"] > 0.0
        assert row["rotational_surface_area_2pi_r_ds_m2"] > 0.0
        assert row["polyline_rz"]
        assert len(row["stable_hash_sha256"]) == 64
        assert row["r_range_m"][0] >= -1.0e-12


def test_production_wet_trace_hash_is_repeatable():
    first = extract_production_wet_traces(ROOT / DEFAULT_PRODUCTION_MESH)
    second = extract_production_wet_traces(ROOT / DEFAULT_PRODUCTION_MESH)
    assert first["stable_trace_hash_sha256"] == second["stable_trace_hash_sha256"]
    assert [row["stable_hash_sha256"] for row in first["front"]["entities"]] == [
        row["stable_hash_sha256"] for row in second["front"]["entities"]
    ]
    assert [row["stable_hash_sha256"] for row in first["rear"]["entities"]] == [
        row["stable_hash_sha256"] for row in second["rear"]["entities"]
    ]

