from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
from vented1.elements import Material,cross,constitutive
from vented3d.native_surface_fields import read_matrix
from vented3d.native_solids import SolidShape,SolidElement,SolidTopology,load_solids,solid_matrices,surface_volume_coordinates
from vented3d.native_solid_connections import connection_constraints,fiber_segments
from vented3d.shell_topology import build_topology

SOURCE=Path(__file__).resolve().parents[1]/'inputs/vented_enclosure'

def make_element(kind,curved=False):
    geom=SOURCE/'native';names=np.array((geom/f'{kind}_dof_names.txt').read_text().splitlines())
    local=read_matrix(geom/f'{kind}_local_dof_coords.f64.gz','>f8')[names=='comp1.u'];shape=SolidShape(kind,local)
    points=local.copy()
    if curved:points[:,2]+=.07*local[:,0]*local[:,1]
    return SolidElement(kind,0,1,shape,points,np.arange(len(local)),np.arange(len(local)))


class NativeSolidTests(unittest.TestCase):
    def test_thickness_interval_on_a_native_edge_is_not_snapped_to_search_grid(self):
        e=make_element('hex')
        segments=fiber_segments(e,np.array([0.,0,.2]),np.array([0.,0,1]),1.)
        self.assertEqual(len(segments),1)
        np.testing.assert_allclose(segments[0],[-.2,.5],atol=1e-7)

    def test_native_shapes_gradients_partition_and_nodal_values(self):
        for kind in ['tet','pyr','hex']:
            e=make_element(kind);s=e.shape
            np.testing.assert_allclose(s.evaluate(s.local,False),np.eye(len(s.local)),atol=2e-14)
            q=np.array([[.1,.12,.2],[.2,.15,.3]]);N,D=s.evaluate(q)
            np.testing.assert_allclose(N.sum(axis=1),1,atol=1e-14);np.testing.assert_allclose(D.sum(axis=1),0,atol=1e-13)
            for a in range(3):
                delta=np.zeros(3);delta[a]=1e-6
                np.testing.assert_allclose(D[:,:,a],(s.evaluate(q+delta,False)-s.evaluate(q-delta,False))/2e-6,rtol=2e-7,atol=2e-9)

    def test_curved_native_solids_pass_rigid_and_constant_strain_patches(self):
        material=Material(2e9,.3,1200);strain=np.array([.002,-.001,.0005,.0002,.0003,-.0001])
        tensor=np.array([[strain[0],strain[3]/2,strain[5]/2],[strain[3]/2,strain[1],strain[4]/2],[strain[5]/2,strain[4]/2,strain[2]]])
        for kind in ['tet','pyr','hex']:
            e=make_element(kind,True);K,M,load,V,_=solid_matrices(e,material)
            R=np.zeros((3*len(e.nodes),6))
            for i,x in enumerate(e.points):R[3*i:3*i+3,:3]=np.eye(3);R[3*i:3*i+3,3:]=-cross(x)
            self.assertLess(np.linalg.norm(K@R)/(np.linalg.norm(K)*np.linalg.norm(R)),1e-13)
            u=(e.points@tensor.T).ravel();expected=V*(strain@constitutive(material.E,material.nu)@strain)
            self.assertAlmostEqual(u@K@u/expected,1.,places=10)
            self.assertAlmostEqual(R[:,0]@M@R[:,0]/(material.rho*V),1.,places=12)

    def test_actual_shared_shell_solid_geometry_is_conforming(self):
        solids=load_solids(SOURCE);shell=build_topology(SOURCE);worst=0.;tags=set()
        for e in shell.elements:
            for index in solids.surface_owners.get((e.kind,e.parent),[]):
                s=solids.elements[index];q=surface_volume_coordinates(e.kind,e.shape.local,solids.surface_coordinates[e.kind,e.parent,index])
                worst=max(worst,np.linalg.norm(s.shape.evaluate(q,False)@s.points-e.points,axis=1).max());tags.add(e.tag)
        self.assertEqual(solids.report['native_solid_nodes'],29635);self.assertEqual(len(tags),12)
        self.assertLess(worst,2e-12)

    def test_corner_plane_section_limit_contains_shear_not_half_curl(self):
        e=make_element('hex');solids=SolidTopology([e],e.points,{('hex',0):0},{('quad',0):[0]}, {},{})
        n=np.array([1.,-1,0])/np.sqrt(2);t=np.array([1.,1,0])/np.sqrt(2);x=np.array([0.,0,.5])
        shell=SimpleNamespace(elements=[SimpleNamespace(kind='quad',parent=0,tag=269,nodes=[0])],xyz=x[None],normals=n[None])
        C,report=connection_constraints(solids,shell,{'temperature_K':293.15,'connection':{}},3*len(e.nodes))
        self.assertEqual(report['edge_corner_limit_count'],1)
        u=np.r_[(.03*(e.points@n)[:,None]*t).ravel(),.03*(x@n)*t,.03*t]
        np.testing.assert_allclose(C@u,0,atol=1e-12)
        u[-3:]*=.5;self.assertGreater(abs(C@u).max(),.005)
