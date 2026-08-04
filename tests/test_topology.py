from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'best_model')]
from loudspeaker_axisym_fem.axisym_magnetics import load_tagged_meshio
from loudspeaker_axisym_fem.stage4C_acoustic_structure import build_stage4C_acoustic_structure_model
from p2_axisym_solid import build_p2_solid,assemble_p2_G
from p2_pml_operator import LocalP2PMLOperator
def test_mixed_p2_topology_contract():
    mesh=load_tagged_meshio(ROOT/'inputs/meshes/comsol_geometry_polyline_coarse_2p5mm.msh')
    ac=build_stage4C_acoustic_structure_model(mesh,ROOT/'inputs/comsol_reference/Untitled.mphtxt',solid_uniform_refine=0,c0=343.203523929095)
    solid=build_p2_solid(mesh)
    op=LocalP2PMLOperator(ac,None,quadrature_order=2,physical_p2_domains=[4])
    G,_=assemble_p2_G(ac,solid,pressure_operator=op)
    A,_=op.matrix(50,2)
    assert solid.ndof==1128
    assert G.shape==(1128,29630)
    assert op.n2==29630
    assert A.shape==(29630,29630)
    assert op.topology.interface_constrained_edges==208
    points, tri3, tri6=op.mixed_points_and_cells()
    assert len(tri3)==8977
    assert len(tri6)==12363
    assert len(op.mixed_triangle6_domains())==len(tri6)
    assert len(points)>=op.n2
