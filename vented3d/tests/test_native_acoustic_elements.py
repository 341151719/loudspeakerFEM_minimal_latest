import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from vented1.mesh import EDGES
from vented1.elements import scalar_element
from vented3d.native_acoustic_elements import NativeShape,basis_with_gradient


class NativeAcousticElementsTest(unittest.TestCase):
    def local(self,kind):
        if kind=='hex':return np.array([[i,j,k] for k in [0.,.5,1] for j in [0.,.5,1] for i in [0.,.5,1]])
        if kind=='pyr':return np.array([[i,j,0.] for j in [0.,.5,1] for i in [0.,.5,1]]+
                                      [[i,j,.5] for j in [0.,.5] for i in [0.,.5]]+[[0.,0,1]])
        p=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
        return np.vstack([p,[(p[i]+p[j])/2 for i,j in EDGES]])

    def test_native_volume_constant_and_coordinate_patch(self):
        for kind,volume in [('tet',1/6),('hex',1.),('pyr',1/3)]:
            local=self.local(kind);shape=NativeShape(kind,local)
            K,M,V,det=shape.matrices(local[None],5)
            self.assertAlmostEqual(V[0],volume,places=12)
            self.assertAlmostEqual(M.sum(),volume,places=12)
            np.testing.assert_allclose(K[0]@np.ones(len(local)),0,atol=3e-13)
            self.assertAlmostEqual(local[:,0]@K[0]@local[:,0],volume,places=12)
            np.testing.assert_allclose(shape.evaluate(local,False),np.eye(len(local)),atol=1e-13)
            if kind=='tet':
                k,m,_=scalar_element(local[:4]);np.testing.assert_allclose(K[0],k,atol=2e-14)
                np.testing.assert_allclose(M[0],m,atol=2e-14)

    def test_native_gradients_match_finite_difference(self):
        q=np.array([[.2,.15,.3]])
        for kind in ['tet','pyr','hex']:
            B,g=basis_with_gradient(q,kind)
            for a in range(3):
                step=np.eye(3)[a]*1e-6
                fd=(basis_with_gradient(q+step,kind)[0]-basis_with_gradient(q-step,kind)[0])/2e-6
                np.testing.assert_allclose(g[:,:,a],fd,atol=3e-10,rtol=1e-8)


if __name__=='__main__':unittest.main()
