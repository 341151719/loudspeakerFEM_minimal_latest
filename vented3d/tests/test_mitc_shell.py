import unittest
import numpy as np
from vented1.elements import Material,cross
from vented3d.mitc_shell import SurfaceShape,shell_matrices


def shape(kind):
    local=np.array([[0.,0],[1,0],[0,1],[.5,0],[.5,.5],[0,.5]]) if kind=='tri' else np.array([[r,s] for s in [0.,.5,1] for r in [0.,.5,1]])
    return SurfaceShape(kind,local)


class MITCShellTests(unittest.TestCase):
    def test_curved_shells_preserve_six_rigid_motions(self):
        for kind in ['tri','quad']:
            s=shape(kind);xy=s.local;x=np.column_stack([xy,.15*xy[:,0]*xy[:,1]+.07*xy[:,0]**2])
            normal=s.nodal_directors(x);K,M,_=shell_matrices(s,x,Material(2e9,.3,1200,thickness=.003))
            R=np.zeros((len(x)*6,6))
            for i in range(len(x)):
                R[6*i:6*i+3,:3]=np.eye(3);R[6*i:6*i+3,3:]=-cross(x[i]);R[6*i+3:6*i+6,3:]=-cross(normal[i])
            self.assertLess(np.linalg.norm(K@R)/(np.linalg.norm(K)*np.linalg.norm(R)),2e-13)
            np.testing.assert_allclose(K,K.T,rtol=1e-12,atol=1e-8)
            self.assertGreater(np.min(np.linalg.eigvalsh(M)),0)

    def test_constant_membrane_patch_and_consistent_mass(self):
        E,nu,rho,t=2e9,.3,1200,.003
        strain=np.array([.002,-.001,.0004]);D=E/(1-nu**2)*np.array([[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]])
        for kind in ['tri','quad']:
            s=shape(kind);x=np.column_stack([s.local,np.zeros(len(s.local))])
            K,M,a=shell_matrices(s,x,Material(E,nu,rho,thickness=t));u=np.zeros((len(x),6))
            u[:,0]=strain[0]*x[:,0]+.5*strain[2]*x[:,1]
            u[:,1]=strain[1]*x[:,1]+.5*strain[2]*x[:,0]
            expected=a['area']*t*(strain@D@strain)
            self.assertAlmostEqual(u.ravel()@K@u.ravel()/expected,1.,places=10)
            u[:]=0;u[:,0]=1
            self.assertAlmostEqual(u.ravel()@M@u.ravel()/(rho*t*a['area']),1.,places=12)

    def test_kirchhoff_bending_patch_is_free_of_shear_locking(self):
        for kind in ['tri','quad']:
            s=shape(kind);x=np.column_stack([s.local,np.zeros(len(s.local))]);u=np.zeros((len(x),6))
            u[:,2]=.5*x[:,0]**2;u[:,3]=-x[:,0]
            for t in [.1,.001,.00001]:
                material=Material(2e9,.3,1200,thickness=t)
                K,_,a=shell_matrices(s,x,material)
                expected=a['area']*material.E/(1-material.nu**2)*t**3/12
                self.assertLess(abs((u.ravel()@K@u.ravel()-expected)/expected),2e-5)
                self.assertLess(abs(u.ravel()@a['shear']@u.ravel())/expected,2e-5)

    def test_tangent_shell_space_has_only_six_zero_energy_modes(self):
        for kind in ['tri','quad']:
            s=shape(kind);x=np.column_stack([s.local,np.zeros(len(s.local))])
            K,_,_=shell_matrices(s,x,Material(2e9,.3,1200,thickness=.08))
            free=np.flatnonzero(np.arange(len(K))%6!=5)
            values=np.linalg.eigvalsh(K[np.ix_(free,free)])
            self.assertEqual(np.count_nonzero(abs(values)<values.max()*1e-10),6)
            self.assertGreater(values.min(),-values.max()*1e-12)

    def test_degenerate_geometry_is_rejected(self):
        s=shape('tri')
        with self.assertRaises((ValueError,np.linalg.LinAlgError)):
            shell_matrices(s,np.zeros((6,3)),Material(2e9,.3,1200,thickness=.01))
