from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu


ROOT = Path(__file__).resolve().parents[1]
FR10 = ROOT / "fr10_full360_cyclic"
sys.path.insert(0, str(FR10))

import base_p2_local_solver as base  # noqa: E402
import cyclic_full360_solver as cyclic  # noqa: E402


def test_fr10_handoff_is_self_contained_and_exact_solver_selected():
    cfg = json.loads((FR10 / "configs/fr10_full360_cyclic.json").read_text())
    assert cfg["geometry"]["full_360_solution_space"] is True
    assert cfg["geometry"]["quarter_symmetry"] is False
    assert cfg["numerics"]["trace_solver"] == "exact_condensation"
    assert "gmres" not in cyclic.solve_phase.__code__.co_names
    expected = {
        "acoustic_base_quarter.msh",
        "surround.msh",
        "cone.msh",
        "dustcap.msh",
        "spider.msh",
        "former.msh",
        "coil.msh",
        "neck_glue.msh",
    }
    assert expected == {path.name for path in (FR10 / "meshes").glob("*.msh")}


def test_exact_condensation_helpers_match_dense_inverse():
    A = csc_matrix(np.array([[4 + 1j, 1], [1, 3 + 0.5j]], complex))
    H = csc_matrix(np.array([[5 + 0.2j, -1], [-1, 4 + 0.1j]], complex))
    indices = np.array([0, 1])
    green = cyclic._interface_green(splu(A), 2, indices, 1)
    np.testing.assert_allclose(green, np.linalg.inv(A.toarray()), rtol=1e-13, atol=1e-13)
    B = csc_matrix(np.array([[1, 2], [0.5, -1]], complex))
    C = B.conj().T.tocsc()
    compliance = cyclic._structural_trace_compliance(splu(H), C, B, 1)
    expected = C.toarray() @ np.linalg.solve(H.toarray(), B.toarray())
    np.testing.assert_allclose(compliance, expected, rtol=1e-13, atol=1e-13)


def test_reconstruction_obeys_vector_bloch_rotation():
    points = np.array([[1.0, 0.0, 0.0], [1.1, 0.0, 0.0], [1.0, 0.1, 0.0], [1.0, 0.0, 0.1]])
    tets = np.array([[0, 1, 2, 3]])
    field = np.tile(np.array([[1 + 2j, 3 - 1j, 0.5j]]), (4, 1))
    P, T, F = cyclic.reconstruct_full_points_cells(points, tets, field, 1, True)
    assert P.shape == (16, 3)
    assert T.shape == (4, 4)
    phase = np.exp(1j * math.pi / 2)
    np.testing.assert_allclose(F[4:8], phase * (field @ cyclic.R90.T))


@pytest.fixture(scope="module")
def startup_model():
    cfg = cyclic.load_cfg()
    model = cyclic.build_sector_model(cfg)
    front, rear = base.build_acoustic_domains(cfg)
    G, report = base.build_local_G(model, front, cfg)
    return cfg, model, front, rear, G, report


def test_periodic_model_build_and_local_asb_contract(startup_model):
    cfg, model, front, rear, G, report = startup_model
    assert model["Nd"] == 39699
    assert len(front["p"]) == len(rear["p"]) == 8072
    assert G.shape == (39699, 8072)
    assert G.nnz == 31743
    assert report["mapping_fallback_quadrature_points"] == 17
    assert report["max_interface_z_mismatch_mm"] < 0.45
    for kclass, expected_nodes in ((0, 7311), (1, 7273), (2, 7273), (3, 7273)):
        phase = np.exp(1j * kclass * math.pi / 2)
        _, _, _ = cyclic.structural_transform(model, phase)
        Tp, acoustic = cyclic.acoustic_transform(front, phase)
        assert Tp.shape[1] == expected_nodes
        assert acoustic["seam_max_mismatch_m"] < 5e-8


def test_front_rear_periodic_trace_operators_are_identical(startup_model):
    cfg, model, front, rear, G, _ = startup_model
    reduced = cyclic.reduced_system(cfg, model, front, rear, G, 500.0, 14.553, 0)
    If = cyclic._active_trace_indices(reduced["Bf"], reduced["Cf"])
    Ir = cyclic._active_trace_indices(reduced["Br"], reduced["Cr"])
    np.testing.assert_array_equal(If, Ir)
    assert cyclic._relative_sparse_difference(reduced["Bf"][:, If], reduced["Br"][:, Ir]) == 0.0
    assert cyclic._relative_sparse_difference(reduced["Cf"][If, :], reduced["Cr"][Ir, :]) == 0.0

