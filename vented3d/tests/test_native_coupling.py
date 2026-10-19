from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from vented1.elements import tri_rule,tri_shape
from vented1.model import Accumulator
from vented3d.model import geometry_patch


class NativeCouplingTest(unittest.TestCase):
    def test_native_quad_pullback_preserves_area_and_virtual_work(self):
        local=np.array([[x,y] for y in [0.,.5,1] for x in [0.,.5,1]])
        points=np.column_stack([local,np.zeros(9)])
        corners=points[[0,2,6,8]];triangle=np.array([corners[0],corners[1],points[4]])
        L,w=tri_rule(5);Nm=np.array([tri_shape(l) for l in L])
        Np,vec=geometry_patch(points,L@triangle,corners,triangle,'quad',local)
        np.testing.assert_allclose(w@vec,[0,0,.25],atol=1e-14)
        B=np.einsum('qi,qj,qk,q->ikj',Nm,Np,vec,w).reshape(18,9)
        rng=np.random.default_rng(764);u=rng.normal(size=(6,3));p=rng.normal(size=9)
        direct=np.sum(w*(Np@p)*np.einsum('qa,qa->q',Nm@u,vec))
        self.assertAlmostEqual(u.ravel()@(B@p),direct,places=13)
        normal=np.tile([0.,0,1],6)
        self.assertAlmostEqual(normal@B@np.ones(9),.25,places=13)

    def test_solid_and_shell_contributions_are_added(self):
        B=np.arange(18*6,dtype=float).reshape(18,6)/200
        accumulator=Accumulator((36,6))
        accumulator.add(np.arange(18),np.arange(6),B)
        accumulator.add(np.arange(18,36),np.arange(6),B)
        coupling=accumulator.finish()
        u=np.tile(np.linspace(-1,1,18),2)
        np.testing.assert_allclose(coupling.T@u,2*B.T@u[:18],atol=1e-14)


if __name__=='__main__':unittest.main()
