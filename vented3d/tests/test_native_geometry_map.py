import unittest
import numpy as np
from vented3d.native_geometry_map import map_points,linear_shape,basis


class NativeGeometryMapTest(unittest.TestCase):
    def test_linear_shape_gradients_and_partition(self):
        for kind,q in [('hex',np.array([[.2,.4,.6]])),('pyr',np.array([[.2,.3,.25]]))]:
            N,g=linear_shape(q,kind)
            np.testing.assert_allclose(N.sum(axis=1),1,atol=1e-14)
            np.testing.assert_allclose(g.sum(axis=1),0,atol=1e-14)
            for a in range(3):
                step=np.eye(3)[a]*1e-6
                fd=(linear_shape(q+step,kind)[0]-linear_shape(q-step,kind)[0])/2e-6
                np.testing.assert_allclose(fd,g[:,:,a],atol=2e-10)

    def test_native_quadratic_maps_reproduce_known_curved_geometry(self):
        for kind in ['hex','pyr']:
            if kind=='hex':
                local=np.array([[i,j,k] for k in [0,.5,1] for j in [0,.5,1] for i in [0,.5,1]])
                q=np.array([[.2,.3,.4],[.7,.6,.8]])
                corners=np.array([[i,j,k] for k in [0,1] for j in [0,1] for i in [0,1]])
            else:
                local=np.array([[i,j,0.] for j in [0,.5,1] for i in [0,.5,1]]+
                               [[i,j,.5] for j in [0,.5] for i in [0,.5]]+[[0,0,1]])
                q=np.array([[.2,.3,.25],[.1,.15,.6]])
                corners=np.array([[0.,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,1]])
            def warp(p):
                result=np.array(p,dtype=float);result[:,0]+=.05*p[:,0]*p[:,1];return result
            physical_query=linear_shape(q,kind)[0]@warp(corners)
            actual=map_points(np.repeat(warp(local)[None],len(q),axis=0),local,physical_query,kind)
            np.testing.assert_allclose(actual,warp(q),rtol=1e-11,atol=1e-12)
            np.testing.assert_allclose(basis(local,kind)@np.linalg.inv(basis(local,kind)),np.eye(len(local)),atol=1e-13)


if __name__=='__main__':unittest.main()
