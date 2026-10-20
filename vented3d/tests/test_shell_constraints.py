import unittest
from pathlib import Path
import numpy as np
from vented1.elements import cross
from vented3d.shell_constraints import director_constraints,fold_constraints,point_connection,thickness_mortar
from vented3d.shell_topology import build_topology


class ShellConstraintTests(unittest.TestCase):
    def test_director_symmetry_preserves_plane_rotations_for_oblique_normals(self):
        from types import SimpleNamespace
        from vented3d.native_shell_model import DirectorSpace,rigid_modes,symmetry
        xyz=np.array([[0.,0,0],[1,0,0],[0,0,1]])
        normals=np.array([[.3,.4,np.sqrt(.75)],[0,1.,0],[0,0,1.]])
        topology=SimpleNamespace(xyz=xyz,normals=normals)
        mesh=SimpleNamespace(x=np.vstack([xyz,xyz]))
        space=DirectorSpace(SimpleNamespace(),np.arange(3,6),np.arange(3),normals)
        C=symmetry(mesh,space,topology);R=rigid_modes(mesh,space,topology)
        np.testing.assert_allclose(C@R[:,[0,2,4]],0,atol=1e-14)

    def test_director_reaction_is_converted_to_physical_moment(self):
        from types import SimpleNamespace
        from vented1.engine import _resultant
        space=SimpleNamespace(offset=3,snode=np.array([0]),hnode=np.array([1]),director_normals=np.array([[0.,0,1]]))
        model=SimpleNamespace(space=space,mesh=SimpleNamespace(x=np.array([[0.,0,0],[1,0,0]])))
        force=np.array([0.,0,0,0,1,0,2,3,0])
        result=_resultant(model,force)
        np.testing.assert_allclose(result['force_N'],[0,1,0])
        np.testing.assert_allclose(result['moment_Nm'],[-3,2,1])

    def test_shared_boundary_selection_does_not_add_edge_or_point_ties(self):
        from types import SimpleNamespace
        from vented3d.native_solid_connections import connection_constraints
        solids=SimpleNamespace(elements=[],surface_owners={})
        topology=SimpleNamespace(elements=[SimpleNamespace(kind='quad',parent=0,tag=271,nodes=np.array([0]))],xyz=np.zeros((1,3)),normals=np.array([[0.,0,1]]))
        C,report=connection_constraints(solids,topology,{'temperature_K':293.15,'connection':{}},12)
        self.assertEqual(C.shape,(0,18));self.assertEqual(report['connected_nodes'],0)
        self.assertEqual(report['connection_scope'],'shared_boundaries_only')

    def test_actual_hidden_settings_are_read_and_incompatible_angle_is_rejected(self):
        from vented3d.shell_settings import load_contract
        source=Path(__file__).resolve().parents[1]/'inputs/vented_enclosure'
        cfg={'shell':{'fold_angle_rad':.001}}
        contract=load_contract(source,cfg)
        self.assertTrue(contract['MITC_enabled']);self.assertEqual(contract['fold_angle_rad'],.001)
        with self.assertRaisesRegex(ValueError,'fold angle'):
            load_contract(source,{'shell':{'fold_angle_rad':.1}})

    def test_fold_and_opposite_normals_preserve_rigid_rotation(self):
        normals=np.array([[0.,0,1],[1,0,0],[0,0,-1.]])
        C,report=fold_constraints([[0,1,2]],normals,18)
        R=np.zeros((18,3))
        for i,n in enumerate(normals):R[6*i+3:6*i+6]=-cross(n)
        np.testing.assert_allclose(C@R,0,atol=1e-15)
        self.assertEqual(report['opposite_normal_pairs'],1)
        wrong=R[:,0].copy();wrong[6+3:6+6]+=[0,0,.5]
        self.assertGreater(abs(C@wrong).max(),.1)

    def test_offset_hard_connection_transfers_force_and_moment(self):
        z=.037;origin=np.array([.2,.1,-.3]);n=np.array([0.,0,1])
        # One solid trace sample and one shell interpolation node.
        C=point_connection(np.array([[0,1,2]]),[1.],np.array([[3,4,5,6,7,8]]),[1.],z,9)
        R=np.zeros((9,6));R[:3,:3]=R[3:6,:3]=np.eye(3)
        R[:3,3:]=-cross(origin+z*n);R[3:6,3:]=-cross(origin);R[6:9,3:]=-cross(n)
        np.testing.assert_allclose(C@R,0,atol=1e-15)
        force=C.T@np.array([2.,-3.,4.])
        np.testing.assert_allclose(R.T@force,0,atol=1e-15)
        self.assertGreater(np.linalg.norm(force[6:]),0)

    def test_thickness_mortar_retains_rotational_moment(self):
        h=.02;points=[]
        # Two solid nodes at the ends of a thickness segment.
        for z,w in zip(np.polynomial.legendre.leggauss(3)[0]*h/2,np.polynomial.legendre.leggauss(3)[1]*h/2):
            points.append({'Nsolid':[(h/2-z)/h,(z+h/2)/h],'solid_dofs':np.arange(6).reshape(2,3),
                'Nshell':[1.],'z':z,'weight':w})
        C,report=thickness_mortar([{'shell_dofs':np.arange(6,12).reshape(1,6),'samples':points}],12)
        R=np.zeros((12,6));R[:3,:3]=R[3:6,:3]=R[6:9,:3]=np.eye(3)
        R[:3,3:]=-cross([0,0,-h/2]);R[3:6,3:]=-cross([0,0,h/2]);R[9:12,3:]=-cross([0,0,1])
        np.testing.assert_allclose(C@R,0,atol=1e-12)
        self.assertEqual(report['quadrature_samples'],3)
        self.assertGreater(np.linalg.norm(C[:,9:12].toarray()),0)

    def test_actual_native_topology_and_constraints_preserve_rigid_motions(self):
        source=Path(__file__).resolve().parents[1]/'inputs/vented_enclosure'
        s=build_topology(source);size=6*len(s.xyz);R=np.zeros((size,6))
        for i,(x,n) in enumerate(zip(s.xyz,s.normals)):
            R[6*i:6*i+3,:3]=np.eye(3);R[6*i:6*i+3,3:]=-cross(x);R[6*i+3:6*i+6,3:]=-cross(n)
        C=director_constraints(np.arange(len(s.xyz)),s.normals,size)
        F,_=fold_constraints(s.groups,s.normals,size)
        self.assertGreater(s.report['native_quad9'],0);self.assertGreater(s.report['fold_node_groups'],0)
        self.assertEqual(s.report['director_nodes'],6995);self.assertEqual(s.report['fold_node_groups'],809)
        self.assertLess(abs(C@R).max(),1e-12);self.assertLess(abs(F@R).max(),1e-12)
