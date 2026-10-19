"""Production condensed operator and native PARDISO regression checks."""
import unittest
import numpy as np
from scipy import sparse,linalg
from scipy.sparse.linalg import lsmr
from vented1.engine import PardisoFactor,boundary_operator


class BEMChainTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(12)
        self.n,self.m,self.r=17,8,11
        self.A=self.random((self.n,self.n))+12*np.eye(self.n)
        self.factor=PardisoFactor(sparse.csr_matrix(self.A),np.geomspace(.02,30,self.n))
        self.addCleanup(self.factor.solver.release)

    def random(self,shape):
        return self.rng.normal(size=shape)+1j*self.rng.normal(size=shape)

    def test_native_forward_adjoint_and_mode_restore(self):
        for shape in ((self.n,),(self.n,3)):
            b=self.random(shape)
            for adjoint in (False,True,False):
                A=self.A.conj().T if adjoint else self.A
                x=self.factor.solve_adj(b) if adjoint else self.factor.solve(b)
                self.assertLess(np.linalg.norm(A@x-b)/np.linalg.norm(b),1e-11)
                self.assertEqual(self.factor.solver.get_iparm_value(11),0)

    def test_mode_restore_on_solve_exception(self):
        original=self.factor.solver.solve
        def fail(rhs):raise RuntimeError('injected failure')
        self.factor.solver.solve=fail
        with self.assertRaisesRegex(RuntimeError,'injected'):
            self.factor.solve_adj(np.ones(self.n))
        self.assertEqual(self.factor.solver.get_iparm_value(11),0)
        self.factor.solver.solve=original
        b=self.random((self.n,))
        self.assertLess(np.linalg.norm(self.A@self.factor.solve(b)-b)/np.linalg.norm(b),1e-11)

    def test_all_chief_rows_adjoint_and_lsmr_dense_oracle(self):
        n,m,r=self.n,self.m,self.r
        B=sparse.csr_matrix(self.random((n,m)))
        Pe=sparse.csr_matrix(self.random((m,n)));Qe=sparse.csr_matrix(self.random((m,n)))
        Ew=sparse.csr_matrix(self.random((m,m)));Ea=sparse.csr_matrix(self.random((m,m)))
        Lp=self.random((r,m));Lq=self.random((r,m))
        w=np.geomspace(.1,3,r);xs=np.geomspace(.2,4,m)
        op,calls=boundary_operator(self.factor,B,Pe,Qe,Ew,Ea,Lp,Lq,w,xs)
        self.assertEqual(op.shape,(r,m))
        response=np.linalg.solve(self.A,B.toarray())
        M=w[:,None]*(Lp@(Pe@response+Ew.toarray())+Lq@(Qe@response+Ea.toarray()))*xs[None]
        z=self.random((m,));v=self.random((r,))
        self.assertLess(np.linalg.norm(op@z-M@z)/np.linalg.norm(M@z),1e-11)
        self.assertLess(np.linalg.norm(op.rmatvec(v)-M.conj().T@v)/np.linalg.norm(M.conj().T@v),1e-11)
        self.assertLess(abs(np.vdot(v,op@z)-np.vdot(op.rmatvec(v),z))/(np.linalg.norm(v)*np.linalg.norm(M@z)),1e-11)
        rhs=self.random((r,))
        result=lsmr(op,rhs,atol=1e-13,btol=1e-13,maxiter=100)
        expected=linalg.lstsq(M,rhs)[0]
        self.assertLess(np.linalg.norm(result[0]-expected)/np.linalg.norm(expected),1e-9)
        self.assertIn(result[1],(1,2,4,5))
        self.assertGreater(calls['adjoint'],0)


class ScalarCoreTests(unittest.TestCase):
    def test_scalar_forward_adjoint_multiple_rhs_and_release(self):
        from vented1.pardiso_native import PardisoSolver
        value=2.+3.j
        solver=PardisoSolver(sparse.csr_matrix([[value]]))
        for rhs in (np.array([1.-4.j]),np.array([[1.-4.j,2.+1.j]])):
            for mode in (0,1,2):
                solver.set_iparm(11,mode)
                expected=value.conjugate() if mode==1 else value
                np.testing.assert_allclose(solver.solve(rhs),rhs/expected,rtol=1e-15)
        solver.release();solver.release()
        with self.assertRaisesRegex(RuntimeError,'released'):solver.solve(np.ones(1))
        with self.assertRaisesRegex(ValueError,'nonsingular'):PardisoSolver(sparse.csr_matrix((1,1),dtype=complex))


class ProductionSphereTests(unittest.TestCase):
    def test_lsmr_spheres_mirrored_and_irregular_frequency(self):
        import json
        from vented1.bem import sphere_mesh,operators
        from pathlib import Path
        options=json.loads((Path(__file__).parents[1]/'configs/native_mixed.json').read_text())['bem']
        for half in (False,True):
            for k in (1.,np.pi):
                tri,normals=sphere_mesh(2,half)
                x=tri.mean(axis=1); radius=np.linalg.norm(x,axis=1)
                exact=np.exp(-1j*k*(radius-1))/radius
                q=(-1/radius-1j*k)*exact*np.einsum('ij,ij->i',normals,x/radius[:,None])
                D,S=operators(x,tri,normals,k,options,half)
                chiefs=np.array([[.13,-.17,.11],[-.21,-.12,.07],[.08,-.24,-.19],[.03,-.04,.02]])
                Dc,Sc=operators(chiefs,tri,normals,k,options,half)
                Lp=np.vstack((.5*np.eye(len(x))-D,Dc))
                Lq=np.vstack((S,-Sc))
                factor=PardisoFactor(sparse.eye(1,format='csr',dtype=complex),np.ones(1))
                try:
                    op,_=boundary_operator(factor,sparse.csr_matrix((1,len(x))),
                        sparse.csr_matrix((len(x),1)),sparse.csr_matrix((len(x),1)),
                        sparse.eye(len(x),format='csr'),sparse.csr_matrix((len(x),len(x))),
                        Lp,Lq,np.ones(Lp.shape[0]),np.ones(len(x)))
                    rhs=-Lq@q
                    result=lsmr(op,rhs,atol=1e-12,btol=1e-12,maxiter=500)
                    p=result[0];oracle=linalg.lstsq(Lp,rhs)[0]
                    error=np.linalg.norm(p-oracle)/np.linalg.norm(oracle)
                    self.assertLess(error,1e-8)
                    obs=np.array([[2.,0,0],[0,0,2.]])
                    Do,So=operators(obs,tri,normals,k,options,half)
                    evaluated=Do@p-So@q
                    expected=np.full(2,.5*np.exp(-1j*k))
                    acoustic_error=np.linalg.norm(evaluated-expected)/np.linalg.norm(expected)
                    self.assertLess(acoustic_error,.4 if k>2 else .15)
                    self.assertIn(result[1],(1,2,4,5))
                    print('PRODUCTION_LSMR_SPHERE',{'half':half,'ka':k,'oracle_relative_error':error,'exterior_relative_error':acoustic_error,'stop':result[1],'iterations':result[2]})
                finally:
                    factor.solver.release()


class TwoMediaSolveTests(unittest.TestCase):
    def test_shared_pressure_and_density_weighted_flux_in_assembled_tets(self):
        from vented1.elements import scalar_element,tet_geometry,tet_shape,tri_rule
        from vented1.mesh import EDGES
        from scipy.sparse.linalg import spsolve
        vertices=np.array([[0.,0,0],[0,1,0],[0,0,1],[-1,.2,.2],[1,.2,.2]])
        cells=np.array([[0,2,1,3],[0,1,2,4]])
        nodes=list(vertices);edges={};t10=[]
        for cell in cells:
            local=list(cell)
            for a,b in EDGES:
                edge=tuple(sorted((int(cell[a]),int(cell[b]))))
                if edge not in edges:
                    edges[edge]=len(nodes);nodes.append((vertices[edge[0]]+vertices[edge[1]])/2)
                local.append(edges[edge])
            t10.append(local)
        nodes=np.array(nodes);t10=np.array(t10)
        K=np.zeros((len(nodes),len(nodes)));rhs=np.zeros(len(nodes))
        rho=[1.2,3.7];flux=.42
        Ls,ws=tri_rule(4)
        for ci,cell in enumerate(cells):
            x=vertices[cell];k,_,_=scalar_element(x);ids=t10[ci]
            K[np.ix_(ids,ids)]+=k/rho[ci]
            for omitted in range(4):
                face=[j for j in range(4) if j!=omitted];tri=x[face]
                if np.max(abs(tri[:,0]))<1e-14:continue
                normal=np.cross(tri[1]-tri[0],tri[2]-tri[0]);area2=np.linalg.norm(normal);normal/=area2
                if normal@(tri.mean(axis=0)-x.mean(axis=0))<0:normal=-normal
                for L,w in zip(Ls,ws):
                    bary=np.zeros(4);bary[face]=L
                    rhs[ids]+=area2*w*tet_shape(bary)*flux*normal[0]
        free=np.arange(1,len(nodes));p=np.zeros(len(nodes));p[free]=spsolve(sparse.csr_matrix(K[np.ix_(free,free)]),rhs[free])
        exact=np.where(nodes[:,0]<0,rho[0],rho[1])*flux*nodes[:,0]
        np.testing.assert_allclose(p,exact,rtol=2e-12,atol=2e-12)
        gradients=[]
        for ci,cell in enumerate(cells):
            _,grad=tet_geometry(vertices[cell]);_,dN=tet_shape(np.array([1/3,1/3,1/3,0]),grad)
            gradients.append(p[t10[ci]]@dN/rho[ci])
        np.testing.assert_allclose(gradients[0],gradients[1],atol=2e-12)
        np.testing.assert_allclose(gradients[0],[flux,0,0],atol=2e-12)
        np.testing.assert_allclose(K@p,rhs,atol=2e-12)
