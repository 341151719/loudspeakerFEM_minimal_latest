from __future__ import annotations

import math
from pathlib import Path
import sys

import meshio
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FR10 = ROOT / "fr10_full360_cyclic"
sys.path.insert(0, str(FR10))

import animate_full360_results as animate  # noqa: E402
import frequency_response_1m as response  # noqa: E402


def test_animation_phase_grid_and_plane_deduplication():
    phases = animate._phase_angles(24)
    assert len(phases) == 24
    assert phases[0] == 0.0
    assert phases[12] == math.pi
    field = np.array([1 + 2j, 3 + 4j, -1j])
    np.testing.assert_allclose(
        np.real(field * np.exp(1j * phases[12])), -np.real(field), atol=1e-14
    )
    points, values = animate._deduplicate_plane(
        np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]]),
        np.array([1 + 1j, 3 + 3j, 5 + 0j]),
    )
    assert len(points) == 2
    np.testing.assert_allclose(values, [2 + 2j, 5 + 0j])


def test_membrane_boundary_triangles_remove_shared_tetra_face():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    mesh = meshio.Mesh(
        points,
        [("tetra", np.array([[0, 1, 2, 3], [0, 2, 1, 4]]))],
        cell_data={"part_id": [np.array([0, 0], dtype=np.int32)]},
    )
    faces = animate._membrane_boundary_triangles(mesh)
    assert faces.shape == (6, 3)
    assert not any(np.array_equal(np.sort(face), [0, 1, 2]) for face in faces)

    quadratic = meshio.Mesh(
        np.vstack((points[:4], np.zeros((6, 3)))),
        [("tetra10", np.arange(10).reshape(1, 10))],
        cell_data={"part_id": [np.array([1], dtype=np.int32)]},
    )
    quadratic_faces = animate._membrane_boundary_triangles(quadratic)
    assert quadratic_faces.shape == (16, 3)
    assert set(range(10)) == set(quadratic_faces.ravel())

    mixed = meshio.Mesh(
        quadratic.points,
        [
            ("tetra", np.array([[0, 1, 2, 3]])),
            ("tetra10", np.arange(10).reshape(1, 10)),
        ],
        cell_data={
            "part_id": [np.array([0], dtype=np.int32), np.array([1], dtype=np.int32)]
        },
    )
    assert animate._membrane_boundary_triangles(mixed).shape == (20, 3)


def test_complete_assembly_loads_supplied_cad_stl_parts(tmp_path):
    components = tmp_path / "components"
    components.mkdir()
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    for _, filename, _ in animate._FIXED_CAD_PARTS:
        meshio.write_points_cells(
            components / filename, points, [("triangle", np.array([[0, 1, 2]]))]
        )
    rows = animate._load_complete_static_cad(tmp_path)
    assert [row[0] for row in rows] == [row[0] for row in animate._FIXED_CAD_PARTS]
    for _, triangles, colour, source in rows:
        assert triangles.ndim == 3 and triangles.shape[1:] == (3, 3)
        assert len(triangles) > 0
        assert np.isfinite(triangles).all()
        assert colour.startswith("#")
        assert source.endswith(".stl")
        assert np.max(triangles) == 1e-3


def test_frequency_response_writes_1m_csv_json_and_png(tmp_path):
    rows = []
    for frequency, front, rear in ((100.0, 70.0, 69.0), (1000.0, 80.0, 77.0)):
        rows.append(
            {
                "frequency_Hz": frequency,
                "front_SPL_1m_dB_at_1V_peak": front,
                "rear_SPL_1m_dB_at_1V_peak": rear,
                "Z_total_ohm": [8.0, 1.0],
                "Z_motional_ohm": [0.5, 0.2],
                "current_A_peak": [0.1, 0.0],
                "coil_displacement_m_peak": [1e-5, 0.0],
                "source": "test",
            }
        )
    paths = response._write_outputs(tmp_path, rows, 1.0)
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)
    expected_offset = 20.0 * math.log10(2.83 * math.sqrt(2.0))
    assert math.isclose(
        rows[0]["front_SPL_1m_dB_at_2p83Vrms"], 70.0 + expected_offset
    )
