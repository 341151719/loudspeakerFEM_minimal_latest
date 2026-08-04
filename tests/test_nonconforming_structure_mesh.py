from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'best_model')]

from coupled_solver import build_best_model

def test_nonconforming_structure_mesh_contract():
    model=build_best_model(
        ROOT,
        magnetostatic_vtu=ROOT/'inputs/comsol_reference/magnetostatic_converged_55iter.vtu',
        build_blocked_coil=False,
    )
    assert model.mesh.n_nodes==11299
    assert len(model.solid.vertex_global_ids)==550
    assert model.G_info['mesh_coupling']=='nonconforming_closest_same_boundary'
    assert model.G_info['max_projection_distance_m'] < 1.5e-4
    assert model.G.shape[0]==model.solid.ndof
    assert model.G.shape[1]==model.acoustic_operator.n2
