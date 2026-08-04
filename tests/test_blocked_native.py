import json
from pathlib import Path

import numpy as np

from loudspeaker_axisym_fem.axisym_magnetics import load_tagged_meshio
from native_blocked_coil import NativeBlockedCoil


ROOT = Path(__file__).resolve().parents[1]


def _solver():
    cfg = json.loads((ROOT / "configs/best_model.json").read_text())
    b = cfg["blocked_coil"]
    mesh = load_tagged_meshio(ROOT / b["field_mesh"])
    return NativeBlockedCoil.from_vtu(mesh, ROOT / b["magnetostatic_vtu"], b)


def test_native_blocked_key_points_and_reciprocity():
    solver = _solver()
    z900 = solver.impedance(900.0)
    assert abs(z900 - complex(7.044841517699552, 6.874040959512264)) < 1e-9
    assert np.allclose(solver.b, solver.c, rtol=1e-13, atol=1e-15)

    z8000 = solver.impedance(8000.0)
    assert abs(z8000 - complex(20.402215437613613, 40.60824405672885)) < 1e-9


def test_native_blocked_is_passive_over_representative_band():
    solver = _solver()
    for f in (1.0, 50.0, 900.0, 2000.0, 5000.0, 8000.0):
        z = solver.impedance(f)
        assert z.real >= solver.Rdc
        assert z.imag > 0.0
