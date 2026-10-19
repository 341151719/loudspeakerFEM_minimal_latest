import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from vented1.mesh import EDGES
from vented1.elements import scalar_element
from vented3d.curved_elements import scalar_batch, inverse_shape
from vented1.elements import tet_shape


class CurvedElementsTest(unittest.TestCase):
    def test_affine_tetrahedra_match_exact_matrices(self):
        for p in [np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]]),
                  np.array([[.2,.3,-.1],[2.,.4,.1],[.4,1.1,.2],[.1,.1,1.5]])]:
            x = np.vstack([p,[(p[i]+p[j])/2 for i,j in EDGES]])
            K,M,V,_ = scalar_batch(x[None])
            k,m,v = scalar_element(p)
            np.testing.assert_allclose(K[0],k,rtol=1e-12,atol=2e-14)
            np.testing.assert_allclose(M[0],m,rtol=1e-12,atol=2e-14)
            self.assertAlmostEqual(V[0],v,places=13)

    def test_curved_patch_preserves_constants_and_quadrature_converges(self):
        p = np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
        x = np.vstack([p,[(p[i]+p[j])/2 for i,j in EDGES]])
        x[4] += [0,.03,.02]
        k,m,v,_ = scalar_batch(x[None],4)
        k5,m5,v5,_ = scalar_batch(x[None],5)
        np.testing.assert_allclose(k[0]@np.ones(10),0,atol=1e-14)
        self.assertGreater(np.linalg.eigvalsh(m[0]).min(),0)
        self.assertAlmostEqual(m[0].sum(),v[0],places=13)
        self.assertLess(np.linalg.norm(k-k5)/np.linalg.norm(k5),1e-5)
        np.testing.assert_allclose(m,m5,rtol=1e-11,atol=1e-14)

    def test_inverted_cell_is_rejected(self):
        p = np.array([[0.,0,0],[0,1,0],[1,0,0],[0,0,1]])
        x = np.vstack([p,[(p[i]+p[j])/2 for i,j in EDGES]])
        with self.assertRaisesRegex(ValueError,'Nonpositive'):
            scalar_batch(x[None])

    def test_inverse_mapping_recovers_known_curved_coordinates(self):
        p=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
        x=np.vstack([p,[(p[i]+p[j])/2 for i,j in EDGES]])
        x[4]+=[0,.03,.02]
        bary=np.array([.4,.2,.25,.15]);shape=tet_shape(bary)
        N,L=inverse_shape(x[None],(shape@x)[None])
        np.testing.assert_allclose(L[0],bary,atol=1e-12)
        np.testing.assert_allclose(N[0],shape,atol=1e-12)


if __name__ == '__main__':
    unittest.main()
