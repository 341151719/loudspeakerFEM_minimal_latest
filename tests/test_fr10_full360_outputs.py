from __future__ import annotations

import math
from pathlib import Path
import sys

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
