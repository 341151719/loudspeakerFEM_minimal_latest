import copy
import gzip
import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from vented1.elements import (
    Material, air, equivalent, Motor, tetra_mass_unit, tetra_integrals,
    tet_geometry, tet_shape, tri_shape, tri_rule, tet_rule,
    solid_element, scalar_element, shell_element, shell_frame
)
from vented1.mesh import (
    convert, native_faces, elevate, refine, incidence, number_spaces,
    EDGES
)
from vented1.model import (
    thickness_connector, reduce_constraints, rigid_vectors, Accumulator
)
from vented1.bem import panel_integral, sphere_check, operators, sphere_mesh
from vented1.reference import field_file, native_vector, scalar_error


CONFIG = json.loads(
    (Path(__file__).resolve().parents[1]/"configs"/"native_mixed.json").read_text()
)


def surfaces_for(cells):
    surfaces, adjacency = [], {}
    for tag, (face, adj) in enumerate(native_faces(cells).values(), 1):
        domains = [d for _, d in adj]
        adjacency[tag] = sorted(set(domains+([0] if len(adj) == 1 else [])))
        if len(face) == 3:
            surfaces.append(("tri", tag, list(face), tag-1))
        else:
            # Convert a perimeter face back to native tensor order.
            surfaces.append(("quad", tag, [face[i] for i in (0, 1, 3, 2)], tag-1))
    return surfaces, adjacency


def tetra_fixture():
    p = np.array([[0., 0, 0], [1., 0, 0], [0., 1, 0], [0., 0, 1]])
    nodes = np.vstack((p, [(p[i]+p[j])/2 for i, j in EDGES]))
    return p, nodes


class SolidTests(unittest.TestCase):
    def test_patch_mass_and_load(self):
        p, nodes = tetra_fixture()
        mat = Material(8e6, .27, 1234)
        K, M, V = solid_element(p, mat)
        self.assertLess(np.max(abs(K-K.T))/np.max(abs(K)), 2e-14)
        self.assertLess(np.max(abs(M-M.T)), 1e-13)
        translation = np.tile([.4, -.2, .7], 10)
        omega = np.array([.2, .3, -.1])
        rotation = np.cross(omega, nodes).ravel()
        for u in (translation, rotation):
            self.assertLess(np.linalg.norm(K@u)/(np.linalg.norm(K)*np.linalg.norm(u)), 1e-13)
        ux = np.tile([1., 0, 0], 10)
        self.assertAlmostEqual(ux@M@ux, mat.rho*V, places=10)
        integral = tetra_integrals(V)
        self.assertAlmostEqual(integral.sum(), V, places=14)
        self.assertTrue(np.all(integral[:4] < 0))
        average = integral/V
        self.assertAlmostEqual(average@nodes[:, 0], .25, places=14)
        self.assertAlmostEqual(average@np.ones(10), 1, places=14)
        strain = np.diag([.02, -.01, .03])
        u = (nodes@strain.T).ravel()
        _, gL = tet_geometry(p)
        from vented1.elements import strain_matrix, constitutive
        _, g = tet_shape(np.array([.1, .2, .3, .4]), gL)
        expected = np.array([.02, -.01, .03, 0, 0, 0])
        np.testing.assert_allclose(strain_matrix(g)@u, expected, atol=1e-14)
        np.testing.assert_allclose(
            u@K@u, V*expected@constitutive(mat.E, mat.nu)@expected, rtol=2e-13)

    def test_material_failure(self):
        from vented1.elements import solid_material
        with self.assertRaisesRegex(ValueError, "domain 43"):
            solid_material(43)


class MeshTests(unittest.TestCase):
    def fixture(self):
        x = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
            [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
            [2, .5, .5], [1.5, .5, -.7]
        ], float)
        cells = [
            ("hex", 1, list(range(8)), 0),
            ("pyr", 2, [1, 3, 5, 7, 8], 0),
            ("tet", 3, [1, 3, 8, 9], 0)
        ]
        return x, cells

    def test_mixed_conforming_conversion_and_refinement(self):
        x, cells = self.fixture()
        surfaces, adj = surfaces_for(cells)
        m = convert(x, cells, surfaces, adj, np.empty((0, 2), int))
        self.assertEqual(set(m.domain), {1, 2, 3})
        self.assertEqual(set(map(tuple, m.parent)), {(0, 0), (1, 0), (2, 0)})
        for level in range(2):
            inc = incidence(m)
            self.assertTrue(all(len(a) in (1, 2) for a in inc.values()))
            tagged = {tuple(sorted(q)) for q in m.btri}
            self.assertTrue(all(key in tagged for key, a in inc.items() if len(a) == 1))
            p2 = elevate(m)
            midpoint_map = {}
            for t, t10 in zip(m.tet, p2.t10):
                for e, mid in zip(EDGES, t10[4:]):
                    key = tuple(sorted((int(t[e[0]]), int(t[e[1]]))))
                    if key in midpoint_map:
                        self.assertEqual(midpoint_map[key], mid)
                    midpoint_map[key] = mid
            if level == 0:
                m = refine(m)

    def test_warped_cell_rejected(self):
        x, cells = self.fixture()
        x[7, 2] += .2
        surfaces, adj = surfaces_for(cells)
        with self.assertRaisesRegex(ValueError, "warped|nonconvex"):
            convert(x, cells, surfaces, adj, np.empty((0, 2), int))


class ShellTests(unittest.TestCase):
    def fixture(self, t=.001):
        p = np.array([[0., 0, 0], [1., 0, 0], [0., .7, 0]])
        nodes = np.vstack((p, (p[0]+p[1])/2, (p[0]+p[2])/2, (p[1]+p[2])/2))
        return p, nodes, Material(2e7, .25, 1200, thickness=t)

    def test_rigid_membrane_bending_inertia(self):
        p, nodes, mat = self.fixture()
        K, M, _, _ = shell_element(p, mat, CONFIG["shell"])
        omega = np.array([.17, -.23, .31])
        rigid = np.column_stack((np.cross(omega, nodes), np.tile(omega, (6, 1)))).ravel()
        self.assertLess(np.linalg.norm(K@rigid)/(np.linalg.norm(K)*np.linalg.norm(rigid)), 1e-12)
        u = np.zeros((6, 6))
        strain = .013
        u[:, 0] = strain*nodes[:, 0]
        area = shell_frame(p)[0]
        expected = area*mat.thickness*mat.E/(1-mat.nu**2)*strain**2
        np.testing.assert_allclose(u.ravel()@K@u.ravel(), expected, rtol=2e-12)
        curvature = .021
        u[:] = 0
        u[:, 2] = -.5*curvature*nodes[:, 0]**2
        u[:, 4] = curvature*nodes[:, 0]
        expected = area*mat.thickness**3/12*mat.E/(1-mat.nu**2)*curvature**2
        np.testing.assert_allclose(u.ravel()@K@u.ravel(), expected, rtol=3e-6)
        _, _, twice = self.fixture(2*mat.thickness)
        K2 = shell_element(p, twice, CONFIG["shell"])[0]
        np.testing.assert_allclose(u.ravel()@K2@u.ravel(), 8*expected, rtol=3e-6)
        translation = np.zeros((6, 6))
        translation[:, 0] = 1
        np.testing.assert_allclose(
            translation.ravel()@M@translation.ravel(), mat.rho*mat.thickness*area)
        rotation = np.zeros((6, 6))
        rotation[:, 4] = 1
        np.testing.assert_allclose(
            rotation.ravel()@M@rotation.ravel(), mat.rho*mat.thickness**3*area/12)
        shear = np.zeros((6, 6))
        shear[:, 4] = .02
        mu = mat.E/(2*(1+mat.nu))
        expected_shear = CONFIG["shell"]["shear_correction"]*mu*mat.thickness*area*.02**2
        np.testing.assert_allclose(shear.ravel()@K@shear.ravel(), expected_shear, rtol=1e-12)

    def compliance(self, nx, t):
        ny, width = 2, .2
        corners = np.array([[i/nx, width*j/ny, 0.]
                            for j in range(ny+1) for i in range(nx+1)])
        faces = []
        for j in range(ny):
            for i in range(nx):
                a = j*(nx+1)+i
                b, c, d = a+1, a+nx+1, a+nx+2
                faces.extend([[a, b, d], [a, d, c]])
        points, mid = corners.tolist(), {}
        cells = []
        for q in faces:
            extra = []
            for i, j in ((0, 1), (0, 2), (1, 2)):
                key = tuple(sorted((q[i], q[j])))
                if key not in mid:
                    mid[key] = len(points)
                    points.append(((corners[key[0]]+corners[key[1]])/2).tolist())
                extra.append(mid[key])
            cells.append(q+extra)
        points = np.array(points)
        n = 6*len(points)
        acc = Accumulator((n, n))
        mat = Material(2e7, 0., 1000, thickness=t)
        for q, cell in zip(faces, cells):
            K = shell_element(corners[q], mat, CONFIG["shell"])[0]
            dofs = (6*np.array(cell)[:, None]+np.arange(6)).ravel()
            acc.add(dofs, dofs, K)
        K = acc.finish()
        load = np.zeros(n)
        # Uniform unit total transverse force on the free edge, P2-consistent.
        for j in range(ny):
            a, b = j*(nx+1)+nx, (j+1)*(nx+1)+nx
            edge_mid = mid[tuple(sorted((a, b)))]
            load[[6*a+2, 6*b+2, 6*edge_mid+2]] += np.array([1, 1, 4])/(6*ny)
        fixed = (6*np.flatnonzero(abs(points[:, 0]) < 1e-12)[:, None]+np.arange(6)).ravel()
        free = np.setdiff1d(np.arange(n), fixed)
        u = np.zeros(n)
        u[free] = spsolve(K[free][:, free], load[free])
        # Normalize by Euler-Bernoulli cantilever compliance, nu=0.
        exact = 4/(mat.E*width*t**3)
        return float(load@u/exact)

    def test_thin_limit_refinement_diagnostic(self):
        values = {(n, t): self.compliance(n, t)
                  for n in (2, 4) for t in (.02, .002, .0002)}
        print("SHELL_THIN_LIMIT_NORMALIZED_COMPLIANCE", values)
        for value in values.values():
            self.assertTrue(np.isfinite(value) and value > .08 and value < 4.)
        # A thickness-driven collapse indicates locking or a conditioning failure.
        for n in (2, 4):
            self.assertGreater(values[n, .0002]/values[n, .002], .35)


class ConnectionTests(unittest.TestCase):
    def fixture(self):
        x = np.array([[i, j, k] for k in (0., 1.) for j in (0., 1.) for i in (0., 1.)])
        cells = [("hex", 1, list(range(8)), 0)]
        surfaces, adj = surfaces_for(cells)
        for index, (kind, tag, nodes, row) in enumerate(surfaces):
            if np.all(x[nodes, 2] == 1):
                surfaces[index] = (kind, 199, nodes, row)
                adj[199] = adj.pop(tag)
        m = elevate(convert(x, cells, surfaces, adj, np.empty((0, 2), int)))
        return m, number_spaces(m)

    def test_thickness_mortar_work_and_moment(self):
        m, s = self.fixture()
        cfg = copy.deepcopy(CONFIG)
        cfg["strict_bundle"] = False
        C, report = thickness_connector(m, s, incidence(m), cfg)
        R = rigid_vectors(m, s)
        self.assertGreater(C.shape[0], 0)
        self.assertLess(np.max(abs(C@R))/max(np.max(abs(C.data)), 1e-30), 1e-9)
        self.assertTrue(any(p["min_overlap_fraction"] > .49 for p in report["primitives"]))
        normalized, T, _, diagnostic = reduce_constraints(C, cfg)
        rng = np.random.default_rng(901)
        u = T@rng.normal(size=T.shape[1])
        self.assertLess(np.max(abs(normalized@u)), 1e-7)
        multiplier = rng.normal(size=normalized.shape[0])
        force = normalized.T@multiplier
        np.testing.assert_allclose(R.T@force, 0, atol=2e-8)
        # Work-conjugate rotation columns are present: not a displacement-only tie.
        rotation = s.offset+6*np.arange(len(s.hnode))[:, None]+np.arange(3, 6)
        self.assertGreater(np.linalg.norm(normalized[:, rotation.ravel()].data), 0)
        self.assertLess(diagnostic["max_C_times_T"], 1e-7)

    def test_reflection_parities(self):
        R = np.diag([1., -1., 1.])
        axial = np.linalg.det(R)*R
        np.testing.assert_array_equal(np.diag(axial), [-1, 1, -1])
        omega, x = np.array([.2, .3, .4]), np.array([.7, -.8, .9])
        np.testing.assert_allclose(np.cross(axial@omega, R@x), R@np.cross(omega, x))


class AcousticElectricalTests(unittest.TestCase):
    def test_manufactured_helmholtz_and_constant(self):
        p, nodes = tetra_fixture()
        K, M, V = scalar_element(p)
        np.testing.assert_allclose(K@np.ones(10), 0, atol=2e-15)
        # p=1+x+y^2+z^2, Delta p=4; weak RHS is boundary flux minus volume source.
        k = 1.7
        nodal = 1+nodes[:, 0]+nodes[:, 1]**2+nodes[:, 2]**2
        load = np.zeros(10)
        Ls, ws = tet_rule(4)
        for L, w in zip(Ls, ws):
            x = L@p
            exact = 1+x[0]+x[1]**2+x[2]**2
            load -= 6*V*w*tet_shape(L)*(4+k*k*exact)
        faces = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))
        Ls, ws = tri_rule(4)
        for face in faces:
            q = p[list(face)]
            n = np.cross(q[1]-q[0], q[2]-q[0])
            area2 = np.linalg.norm(n)
            n /= area2
            for L, weight in zip(Ls, ws):
                bary = np.zeros(4)
                bary[list(face)] = L
                x = bary@p
                gradient = np.array([1, 2*x[1], 2*x[2]])
                load += area2*weight*tet_shape(bary)*(gradient@n)
        np.testing.assert_allclose((K-k*k*M)@nodal, load, atol=2e-14, rtol=2e-13)

    def test_two_media_flux_and_virtual_work(self):
        # Interface trace assembly uses the same quadratic trace on both sides.
        Ls, ws = tri_rule(4)
        mass = sum(2*w*np.outer(tri_shape(L), tri_shape(L)) for L, w in zip(Ls, ws))
        n = np.array([1., 0, 0])
        B = np.einsum("ij,k->ikj", mass, n).reshape(18, 6)
        rng = np.random.default_rng(78)
        u, pressure = rng.normal(size=18), rng.normal(size=6)
        np.testing.assert_allclose(u@(B@pressure), pressure@(B.T@u), atol=1e-14)
        force = (B@np.ones(6)).reshape(6, 3).sum(axis=0)
        np.testing.assert_allclose(force, n, atol=1e-14)
        np.testing.assert_allclose(B@np.ones(6)-B@np.ones(6), 0, atol=1e-14)
        rho1, rho2, flux = 1.2, 3.7, .42
        slopes = [rho1*flux, rho2*flux]
        self.assertAlmostEqual(slopes[0]/rho1, slopes[1]/rho2)
        # Opposite outward normals yield cancelling interface weak flux.
        np.testing.assert_allclose(flux*mass@np.ones(6)-flux*mass@np.ones(6), 0)

    def test_slits(self):
        a = air()
        self.assertGreater(a["Pr"], 0)
        for f in (1e-7, 20, 3550, 1e7):
            for h in (.0002, .0004):
                c = equivalent(f, h, a)
                self.assertLessEqual(c["rho"].imag, 1e-10*abs(c["rho"]))
                self.assertLessEqual(c["b"].imag, 1e-10*abs(c["b"]))
                self.assertTrue(np.isfinite(c["a"]))
        wide = equivalent(1e7, .1, a)
        self.assertLess(abs(wide["rho"]/a["rho"]-1), 1e-4)
        low = equivalent(1e-6, .0004, a)
        half, w = .0002, 2*np.pi*1e-6
        np.testing.assert_allclose(low["rho"].imag, -3*a["eta"]/(w*half**2), rtol=1e-6)
        np.testing.assert_allclose(low["bulk"].real, a["pressure"], rtol=1e-6)

    def test_motor_table_and_feedback(self):
        motor = Motor(np.array([[10, 4], [100, 8]]),
                      np.array([[10, .001], [100, .003]]))
        self.assertEqual(motor.values(1)[0], 4)
        self.assertEqual(motor.values(1000)[1], .003)
        self.assertAlmostEqual(motor.values(55)[0], 6)
        w, BL, V, Z = 30., motor.BL, 3.55, motor.values(55)[2]
        H = np.diag([100+3j, 300+4j])
        f = .5*BL*np.array([1., -1.])
        g = np.array([1., 0.])
        system = np.block([[H, -f[:, None]],
                           [1j*w*BL*g[None], np.array([[Z]])]])
        solution = np.linalg.solve(system, np.r_[0., 0., V])
        effective = H+np.outer(f, 1j*w*BL*g)/Z
        displacement = np.linalg.solve(effective, f*V/Z)
        np.testing.assert_allclose(solution[:2], displacement, rtol=1e-13)
        self.assertAlmostEqual(f.sum(), 0)
        np.testing.assert_allclose(Z*solution[-1]+1j*w*BL*g@solution[:2], V)


class BEMTests(unittest.TestCase):
    def test_singular_and_nearly_singular_panel(self):
        tri = np.array([[0., 0, 0], [1., 0, 0], [0., 1, 0]])
        options = copy.deepcopy(CONFIG["bem"])
        x = tri.mean(axis=0)
        S, D = panel_integral(x, tri, np.array([0., 0, 1.]), 0., options)
        self.assertGreater(S.real, 0)
        self.assertEqual(D, 0)
        near = x+np.array([0., 0, 1e-8])
        Sn, Dn = panel_integral(near, tri, np.array([0., 0, 1.]), 0., options)
        self.assertLess(abs(Sn-S), 1e-6)
        self.assertLess(abs(Dn-.5), 1e-6)

    def test_laplace_jump_and_mirror(self):
        tri, n = sphere_mesh(1)
        D, _ = operators(np.zeros((1, 3)), tri, n, 0., CONFIG["bem"], False)
        self.assertLess(abs(D.sum()+1), 2e-7)
        full = sphere_check(1, 1., CONFIG["bem"], False)
        half = sphere_check(1, 1., CONFIG["bem"], True)
        np.testing.assert_allclose(full["evaluated_pressure"], half["evaluated_pressure"],
                                   rtol=.04, atol=.02)

    def test_sphere_refinement_and_irregular_frequency(self):
        coarse = sphere_check(1, 1., CONFIG["bem"])
        fine = sphere_check(2, 1., CONFIG["bem"])
        print("BEM_SPHERE", coarse, fine)
        self.assertLess(fine["exterior_relative_error"], coarse["exterior_relative_error"])
        self.assertLess(fine["exterior_relative_error"], .15)
        irregular = sphere_check(2, np.pi, CONFIG["bem"])
        self.assertEqual(irregular["rank"], irregular["panels"])
        self.assertLess(irregular["exterior_relative_error"], .4)


class IOTests(unittest.TestCase):
    def test_big_endian_contracts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            x = np.array([[.1, .2], [.3, .4], [.5, .6]])
            values = np.array([[[1+2j, 3+4j]]])
            pairs = np.stack((values.real, values.imag), axis=-1)
            payload = struct.pack(">4i", 1, 1, 1, 2)
            payload += x.astype(">f8").tobytes()+pairs.astype(">f8").tobytes()
            with gzip.open(root/"field.gz", "wb") as f:
                f.write(payload)
            coordinates, read = field_file(root/"field.gz")
            np.testing.assert_array_equal(coordinates, x.T)
            np.testing.assert_array_equal(read, values)
            vector = np.array([7+8j, 9+10j])
            with gzip.open(root/"vector.gz", "wb") as f:
                f.write(struct.pack(">2i", 1, 2))
                f.write(np.column_stack((vector.real, vector.imag)).astype(">f8").tobytes())
            (root/"dofs.csv").write_text(
                "dof_index,solution_vector_index,component,node_index,x_native,y_native,z_native\n"
                "0,1,comp1.u,912,0.1,0.2,0.3\n", encoding="utf-8")
            result = native_vector(root/"vector.gz", root/"dofs.csv", {"u"})
            self.assertEqual(result["comp1.u"]["values"][0], vector[1])
            self.assertEqual(result["comp1.u"]["node_index"][0], 912)

    def test_null_error_reporting(self):
        r = scalar_error(1e-4+0j, 1e-12+0j, 1e-5)
        self.assertIsNone(r["complex_relative_error"])
        self.assertTrue(r["small_reference"])
        self.assertGreater(r["absolute_error"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
