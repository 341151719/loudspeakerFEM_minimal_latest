import unittest
import numpy as np
from vented3d.native_surface_fields import surface_basis,surface_inverse


class NativeSurfaceFieldsTest(unittest.TestCase):
    def test_native_surface_spaces_reproduce_polynomials(self):
        tri=np.array([[0.,0],[.5,0],[1,0],[0,.5],[.5,.5],[0,1]])
        quad=np.array([[x,y] for y in [0.,.5,1] for x in [0.,.5,1]])
        for kind,local in [('tri',tri),('quad',quad),('quad',quad[np.any(quad!=.5,axis=1)])]:
            query=np.array([[.2,.3],[.1,.4]])
            N=surface_basis(query,kind,len(local))@np.linalg.inv(surface_basis(local,kind,len(local)))
            np.testing.assert_allclose(N.sum(axis=1),1,atol=2e-14)
            values=1+local[:,0]**2+2*local[:,0]*local[:,1]-local[:,1]
            expected=1+query[:,0]**2+2*query[:,0]*query[:,1]-query[:,1]
            np.testing.assert_allclose(N@values,expected,atol=2e-14)

    def test_quad_inverse_recovers_nonrectangular_reference_coordinates(self):
        corners=np.array([[0.,0,0],[1,.1,0],[.1,1,0],[1.3,1.1,0]])
        q=np.array([[.2,.3],[.7,.8],[0.,1]])
        N=np.column_stack([(1-q[:,0])*(1-q[:,1]),q[:,0]*(1-q[:,1]),(1-q[:,0])*q[:,1],q[:,0]*q[:,1]])
        actual=surface_inverse(corners,N@corners,'quad')
        np.testing.assert_allclose(actual,q,atol=2e-11)


if __name__=='__main__':unittest.main()
