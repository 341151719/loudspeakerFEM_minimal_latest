from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict

import numpy as np
from scipy import sparse

from .core import (
    AIR_DOMAINS, SOLID_DOMAINS, SHELL_TAGS, SYMMETRY, guard, normmax
)
from .mesh import (
    Mesh, Space, incidence, number_spaces, trace_positions, Locator
)
from .elements import (
    solid_material, shell_material, solid_element, scalar_element,
    shell_element, shell_frame, tri_rule, tri_shape, tet_shape,
    tetra_integrals, unit_gauss, cross, Motor, tet_geometry
)


class Accumulator:
    """Bounded triplet buffer; duplicates are intentionally summed."""
    def __init__(self, shape, limit=180000):
        self.shape, self.limit = shape, limit
        self.r, self.c, self.d = [], [], []
        self.count = 0
        self.matrix = sparse.csr_matrix(shape, dtype=float)

    def add(self, rr, cc, values):
        rr, cc = np.asarray(rr, int), np.asarray(cc, int)
        a = np.asarray(values, float)
        if a.shape != (len(rr), len(cc)):
            raise ValueError(f"Triplet shape mismatch {a.shape}, {len(rr)}, {len(cc)}")
        if (len(rr) and (rr.min() < 0 or rr.max() >= self.shape[0])) or (
                len(cc) and (cc.min() < 0 or cc.max() >= self.shape[1])):
            raise IndexError("Assembly index outside field space")
        nzr, nzc = np.nonzero(a)
        if not len(nzr):
            return
        self.r.append(rr[nzr])
        self.c.append(cc[nzc])
        self.d.append(a[nzr, nzc])
        self.count += len(nzr)
        if self.count >= self.limit:
            self.flush()

    def flush(self):
        if self.count:
            a = sparse.coo_matrix(
                (np.concatenate(self.d), (np.concatenate(self.r), np.concatenate(self.c))),
                shape=self.shape
            ).tocsr()
            a.eliminate_zeros()
            self.matrix = self.matrix + a
            self.r, self.c, self.d, self.count = [], [], [], 0

    def finish(self):
        self.flush()
        self.matrix.sum_duplicates()
        self.matrix.eliminate_zeros()
        return self.matrix


def reduce_constraints(C: sparse.csr_matrix, cfg: dict):
    """Homogeneous sparse row elimination, with explicit fill and residual guards."""
    tol = cfg["constraint_tol"]
    C = C.tocsr().copy()
    maxima = np.asarray(abs(C).max(axis=1).toarray()).ravel()
    keep = maxima > 0
    C = sparse.diags(1 / maxima[keep]) @ C[keep]
    C = C.tocsr()
    basis, order = {}, []
    dependent = 0
    for row in range(C.shape[0]):
        start, end = C.indptr[row:row+2]
        d = {int(j): float(v) for j, v in zip(C.indices[start:end], C.data[start:end])
             if abs(v) > tol}
        while True:
            found = [p for p in d if p in basis]
            if not found:
                break
            p = min(found, key=lambda j: basis[j][0])
            factor = d.pop(p)
            for j, v in basis[p][1].items():
                d[j] = d.get(j, 0.) - factor * v
                if abs(d[j]) < tol:
                    d.pop(j, None)
        if not d:
            dependent += 1
            continue
        p = max(d, key=lambda j: (abs(d[j]), -j))
        factor = d.pop(p)
        d = {j: v / factor for j, v in d.items() if abs(v / factor) > tol}
        basis[p] = (len(order), d)
        order.append(p)
    n = C.shape[1]
    free = np.array([j for j in range(n) if j not in basis], int)
    free_map = {j: k for k, j in enumerate(free)}
    expressions = {}
    entries = len(free)
    for p in reversed(order):
        result = {}
        for j, value in basis[p][1].items():
            terms = expressions[j] if j in expressions else {free_map[j]: 1.}
            for k, v in terms.items():
                result[k] = result.get(k, 0.) - value * v
        result = {k: v for k, v in result.items() if abs(v) > tol}
        expressions[p] = result
        entries += len(result)
        if entries > cfg["max_transform_entries"]:
            raise MemoryError("Constraint transformation fill limit exceeded")
    rr, cc, dd = list(free), list(range(len(free))), [1.] * len(free)
    for p, ex in expressions.items():
        rr.extend([p] * len(ex))
        cc.extend(ex)
        dd.extend(ex.values())
    T = sparse.coo_matrix((dd, (rr, cc)), shape=(n, len(free))).tocsr()
    residual = normmax(C @ T)
    if residual > 200 * tol * max(1., normmax(T)):
        raise ValueError(f"Constraint reduction residual {residual:g}")
    return C, T, free, {
        "rows": C.shape[0], "rank": len(order), "dependent_rows": dependent,
        "free_dofs": len(free), "transform_nnz": T.nnz,
        "max_C_times_T": residual
    }


def rigid_vectors(m: Mesh, s: Space) -> np.ndarray:
    R = np.zeros((s.nm, 6))
    for i, g in enumerate(s.snode):
        R[3*i:3*i+3, :3] = np.eye(3)
        R[3*i:3*i+3, 3:] = -cross(m.x[g])
    for i, g in enumerate(s.hnode):
        base = s.offset + 6*i
        R[base:base+3, :3] = np.eye(3)
        R[base:base+3, 3:] = -cross(m.x[g])
        R[base+3:base+6, 3:] = np.eye(3)
    return R


def symmetry_constraints(m: Mesh, s: Space, inc: dict):
    fixed = set()
    for bi, tag in enumerate(m.tag):
        if tag not in SYMMETRY:
            continue
        if np.max(np.abs(m.x[m.b6[bi], 1])) > 1e-8:
            raise ValueError(f"Symmetry boundary {tag} is not on Y=0")
        for ci, _ in inc[tuple(sorted(m.btri[bi]))]:
            if m.domain[ci] in SOLID_DOMAINS:
                pos = trace_positions(m, ci, m.b6[bi])
                fixed.update(3 * s.scell[ci, pos] + 1)
    lookup = {int(g): j for j, g in enumerate(s.hnode)}
    selected = set(m.sym3.ravel())
    for g in selected.intersection(lookup):
        if abs(m.x[g, 1]) > 1e-8:
            raise ValueError(f"box2 shell node {g} is not on Y=0")
        b = s.offset + 6 * lookup[g]
        fixed.update((b + 1, b + 3, b + 5))
    on_plane = {int(g) for g in s.hnode if abs(m.x[g, 1]) < 1e-11}
    if not on_plane.issubset(selected):
        raise ValueError("Shell nodes on Y=0 are missing from box2 edge traces")
    fixed = np.array(sorted(fixed), int)
    return sparse.coo_matrix((np.ones(len(fixed)), (np.arange(len(fixed)), fixed)),
                             shape=(len(fixed), s.nm)).tocsr()


def thickness_connector(m: Mesh, s: Space, inc: dict, cfg: dict):
    """Hard mortar: integral S_shell^T (u_solid - S_shell U_shell) = 0."""
    acc = Accumulator((s.nm, s.nm), cfg["triplet_entries"])
    solid_cells = np.flatnonzero(np.isin(m.domain, list(SOLID_DOMAINS)))
    locator = Locator(m, solid_cells)
    local_of_cell = {int(ci): j for j, ci in enumerate(locator.cells)}
    shared, covered_edges = [], set()
    shell_edges = defaultdict(list)
    solid_edges = defaultdict(set)
    for key, adjacent in inc.items():
        solid_adj = [ci for ci, _ in adjacent if m.domain[ci] in SOLID_DOMAINS]
        for a, b in ((key[0], key[1]), (key[0], key[2]), (key[1], key[2])):
            for ci in solid_adj:
                solid_edges[tuple(sorted((a, b)))].add(int(m.domain[ci]))
    for bi, tag in enumerate(m.tag):
        if tag not in SHELL_TAGS:
            continue
        q = m.btri[bi]
        for a, b in ((q[0], q[1]), (q[0], q[2]), (q[1], q[2])):
            shell_edges[tuple(sorted((int(a), int(b))))].append(bi)
        adj = inc[tuple(sorted(q))]
        domains = {int(m.domain[ci]) for ci, _ in adj if m.domain[ci] in SOLID_DOMAINS}
        if domains:
            shared.append((bi, domains))
            covered_edges.update(tuple(sorted((int(q[i]), int(q[j]))))
                                 for i, j in ((0, 1), (0, 2), (1, 2)))
    primitives = []
    Lq, Wq = tri_rule(cfg["connection"]["quadrature"])
    for bi, domains in shared:
        A = shell_frame(m.x[m.btri[bi]])[0]
        primitives.append(("face", bi, domains, Lq, 2*A*Wq))
    gauss, weights = unit_gauss(cfg["connection"]["quadrature"])
    for edge, faces in shell_edges.items():
        if edge in covered_edges or edge not in solid_edges:
            continue
        for bi in faces:
            q = list(m.btri[bi])
            a, b = q.index(edge[0]), q.index(edge[1])
            L = np.zeros((len(gauss), 3))
            L[:, a], L[:, b] = 1-gauss, gauss
            length = np.linalg.norm(m.x[edge[1]] - m.x[edge[0]])
            primitives.append(("edge", bi, solid_edges[edge], L, weights*length))
    report = []
    zg, zw = unit_gauss(3)
    for kind, bi, domains, Ls, surface_weights in primitives:
        tag = int(m.tag[bi])
        mat = shell_material(tag, cfg["temperature_K"])
        h = mat.thickness
        points = m.x[m.btri[bi]]
        n = shell_frame(points)[3]
        hdofs = s.hdofs(bi)
        fractions, offsets = [], []
        for L, sw in zip(Ls, surface_weights):
            x = L @ points
            N = tri_shape(L)
            segments = locator.thickness_segments(x, n, h, domains)
            length = sum(b-a for a, b, _ in segments)
            fraction = length / h
            if fraction < cfg["connection"]["min_fraction"]:
                raise ValueError(
                    f"Solid-shell {kind} boundary {tag}, triangle {bi}: "
                    f"thickness overlap {fraction:g}, point {x.tolist()}, "
                    f"solid domains {sorted(domains)}"
                )
            fractions.append(fraction)
            offsets.extend([segments[0][0], segments[-1][1]])
            for a, b, ci in segments:
                for u, w in zip(zg, zw):
                    z = a + (b-a)*u
                    Lsolid = locator.bary(local_of_cell[ci], x + z*n)
                    Ns = tet_shape(Lsolid)
                    Ss = np.zeros((3, 36))
                    Sv = np.zeros((3, 30))
                    for j, Nj in enumerate(N):
                        Ss[:, 6*j:6*j+3] = Nj*np.eye(3)
                        Ss[:, 6*j+3:6*j+6] = -z*Nj*cross(n)
                    for j, Nj in enumerate(Ns):
                        Sv[:, 3*j:3*j+3] = Nj*np.eye(3)
                    # Face-overlay constraints use a thickness average; edge
                    # attachments use the actual shell-thickness connected area.
                    weight = sw * (b-a) * w / (h if kind == "face" else 1.)
                    acc.add(hdofs, s.sdofs(ci), weight * Ss.T @ Sv)
                    acc.add(hdofs, hdofs, -weight * Ss.T @ Ss)
        report.append({
            "kind": kind, "boundary": tag, "triangle": bi,
            "solid_domains": sorted(domains), "thickness_m": h,
            "min_overlap_fraction": min(fractions),
            "offset_range_m": [min(offsets), max(offsets)],
            "additional_offset_m": 0., "geometric_projection_distance_m": 0.
        })
    C = acc.finish()
    keep = np.diff(C.indptr) > 0
    C = C[keep]
    R = rigid_vectors(m, s)
    scale = max(normmax(C), 1e-300)
    error = normmax(C @ R) / scale
    if error > 2e-8:
        raise ValueError(f"Connection violates rigid motion: relative residual {error:g}")
    return C, {
        "primitives": report,
        "shared_shell_boundaries": sorted({int(m.tag[bi]) for bi, _ in shared}),
        "rows_before_dependency_removal": C.shape[0],
        "rigid_motion_error": error
    }


def nodal_spin_connector(m: Mesh, s: Space, inc: dict):
    """Local hard tie for a diagnostic run; solid spin supplies shell rotation.

    This is a declared approximation to the finite-thickness mortar. Each tied
    shell node uses the displacement and curl of an incident solid P2 element.
    """
    solid_by_node = {}
    for ci in np.flatnonzero(np.isin(m.domain, list(SOLID_DOMAINS))):
        for g in m.t10[ci]:
            solid_by_node.setdefault(int(g), int(ci))
    tied = {int(g) for bi, tag in enumerate(m.tag) if tag in SHELL_TAGS
            for g in m.b6[bi] if int(g) in solid_by_node}
    if not tied:
        raise ValueError("No topologically shared solid-shell nodes")
    shell_lookup = {int(g): i for i, g in enumerate(s.hnode)}
    rows, cols, vals = [], [], []
    eliminated = set()
    for g in sorted(tied):
        ci = solid_by_node[g]
        local = int(np.flatnonzero(m.t10[ci] == g)[0])
        L = np.zeros(4)
        if local < 4:
            L[local] = 1.
        else:
            from .mesh import EDGES
            a, b = EDGES[local-4]
            L[a] = L[b] = .5
        _, gradients = tet_geometry(m.x[m.tet[ci]])
        N, dN = tet_shape(L, gradients)
        base = s.offset + 6*shell_lookup[g]
        sdofs = s.sdofs(ci).reshape(10, 3)
        for axis in range(3):
            target = base + axis
            eliminated.add(target)
            rows.append(target); cols.append(int(sdofs[local, axis])); vals.append(1.)
        curl = ((2, 1, 1, 2), (0, 2, 2, 0), (1, 0, 0, 1))
        for axis, (v1, x1, v2, x2) in enumerate(curl):
            target = base + 3 + axis
            eliminated.add(target)
            for j in range(10):
                rows.extend((target, target))
                cols.extend((int(sdofs[j, v1]), int(sdofs[j, v2])))
                vals.extend((.5*dN[j, x1], -.5*dN[j, x2]))
    free = np.array([j for j in range(s.nm) if j not in eliminated], int)
    free_map = np.full(s.nm, -1, int)
    free_map[free] = np.arange(len(free))
    tr = list(free); tc = list(range(len(free))); tv = [1.] * len(free)
    tr.extend(rows)
    tc.extend(free_map[np.asarray(cols, int)].tolist())
    tv.extend(vals)
    T = sparse.coo_matrix((tv, (tr, tc)), shape=(s.nm, len(free))).tocsr()
    C = sparse.eye(s.nm, format="csr")[sorted(eliminated)] - sparse.coo_matrix(
        (vals, (np.searchsorted(sorted(eliminated), rows), cols)),
        shape=(len(eliminated), s.nm)).tocsr()
    if normmax(C @ T) > 1e-8:
        raise ValueError("Nodal tie algebraic residual")
    R = rigid_vectors(m, s)
    if normmax(C @ R) > 1e-7:
        raise ValueError("Nodal tie does not preserve rigid motion")
    return C, T, free, {
        "method": "nodal_displacement_and_solid_spin",
        "approximation": "local hard tie in place of finite-thickness mortar",
        "tied_nodes": len(tied), "rows": C.shape[0],
        "rigid_motion_error": normmax(C @ R)
    }


def interface_blocks(m: Mesh, s: Space, inc: dict, cfg: dict):
    ext = [bi for bi, tag in enumerate(m.tag)
           if 0 in m.adjacency[int(tag)] and tag not in SYMMETRY]
    panels, normals, areas = [], [], []
    for bi in ext:
        adj = inc[tuple(sorted(m.btri[bi]))]
        if len(adj) != 1:
            raise ValueError(f"Exterior boundary {m.tag[bi]} is not singly adjacent")
        ci, oriented = adj[0]
        p = m.x[oriented]
        n = np.cross(p[1]-p[0], p[2]-p[0])
        A = np.linalg.norm(n)/2
        panels.append(p)
        normals.append(n/(2*A))
        areas.append(A)
    ext = np.array(ext, int)
    panels, normals, areas = np.array(panels), np.array(normals), np.array(areas)
    nb = len(ext)
    edge_count = defaultdict(int)
    for bi in ext:
        q = m.btri[bi]
        for a, b in ((q[0], q[1]), (q[0], q[2]), (q[1], q[2])):
            edge_count[tuple(sorted((int(a), int(b))))] += 1
    for edge, count in edge_count.items():
        if count == 1:
            if np.max(np.abs(m.x[list(edge), 1])) > 1e-10:
                raise ValueError(f"Exterior half-boundary has a non-symmetry crack {edge}")
        elif count != 2:
            raise ValueError(f"Exterior nonmanifold edge {edge}, incidence {count}")
    Bi = Accumulator((s.nm, s.np), cfg["triplet_entries"])
    Be = Accumulator((s.nm, nb), cfg["triplet_entries"])
    Cp = Accumulator((s.np, nb), cfg["triplet_entries"])
    Ls, ws = tri_rule(4)
    NN = sum(w*np.outer(tri_shape(L), tri_shape(L)) for L, w in zip(Ls, ws))
    Nv = sum(w*tri_shape(L) for L, w in zip(Ls, ws))
    ext_index = {int(bi): j for j, bi in enumerate(ext)}
    kinds = defaultdict(set)

    def mechanics(bi, adjacent):
        if m.tag[bi] in SHELL_TAGS:
            return (s.offset + 6*s.hcell[bi, :, None] + np.arange(3)).ravel()
        solids = [ci for ci, _ in adjacent if m.domain[ci] in SOLID_DOMAINS]
        if len(solids) != 1:
            raise ValueError(f"Boundary {m.tag[bi]}: ambiguous mechanical side")
        pos = trace_positions(m, solids[0], m.b6[bi])
        return (3*s.scell[solids[0], pos, None] + np.arange(3)).ravel()

    for bi, tag in enumerate(m.tag):
        tag = int(tag)
        if tag in SYMMETRY:
            continue
        adjacent = inc[tuple(sorted(m.btri[bi]))]
        fluids = [ci for ci, _ in adjacent if m.domain[ci] in AIR_DOMAINS]
        solids = [ci for ci, _ in adjacent if m.domain[ci] in SOLID_DOMAINS]
        shell = tag in SHELL_TAGS
        if fluids and solids:
            kinds["interior_acoustic_solid"].add(tag)
        if fluids and shell:
            kinds["interior_acoustic_shell"].add(tag)
        p = m.x[m.btri[bi]]
        area2 = np.linalg.norm(np.cross(p[1]-p[0], p[2]-p[0]))
        for ci in fluids:
            moving = shell or bool(solids)
            if moving:
                face = next(q for c, q in adjacent if c == ci)
                q = m.x[face]
                n = np.cross(q[1]-q[0], q[2]-q[0])
                n /= np.linalg.norm(n)
                pd = s.pcell[ci, trace_positions(m, ci, m.b6[bi])]
                md = mechanics(bi, adjacent)
                B = np.einsum("ij,k->ikj", area2*NN, n).reshape(18, 6)
                Bi.add(md, pd, B)
            elif 0 in m.adjacency[tag] and tag != 277:
                raise ValueError(f"Unassigned open acoustic boundary {tag}")
        if bi in ext_index:
            j = ext_index[bi]
            if shell:
                kinds["exterior_shell"].add(tag)
            if solids:
                kinds["exterior_solid"].add(tag)
            if tag == 277:
                if len(fluids) != 1 or shell or solids:
                    raise ValueError("Boundary 277 must be a direct, shell-free aperture")
                ci = fluids[0]
                pd = s.pcell[ci, trace_positions(m, ci, m.b6[bi])]
                Cp.add(pd, [j], (area2*Nv)[:, None])
            else:
                md = mechanics(bi, adjacent)
                force = (area2*Nv[:, None]*normals[j]).ravel()
                Be.add(md, [j], force[:, None])
    ap = np.flatnonzero(m.tag[ext] == 277)
    wall = np.flatnonzero(m.tag[ext] != 277)
    if not len(ap):
        raise ValueError("No boundary-277 aperture panels")
    report = {k: sorted(v) for k, v in kinds.items()}
    report["counts"] = {k: len(v) for k, v in kinds.items()}
    report["overlap_loading_policy"] = "shell receives pressure once; connector transfers it"
    return (Bi.finish(), Be.finish(), Cp.finish(), ext, panels, normals,
            areas, wall, ap, report)


@dataclass
class Model:
    mesh: Mesh
    space: Space
    full: dict
    reduced: dict
    ak: dict
    am: dict
    Cp: sparse.csr_matrix
    ext: np.ndarray
    panels: np.ndarray
    normals: np.ndarray
    areas: np.ndarray
    wall: np.ndarray
    aperture: np.ndarray
    constraints: sparse.csr_matrix
    T: sparse.csr_matrix
    free: np.ndarray
    report: dict


def describe(m: Mesh, s: Space, cfg: dict) -> dict:
    volumes = defaultdict(float)
    quality = []
    for t, d in zip(m.tet, m.domain):
        p = m.x[t]
        V = np.linalg.det(np.column_stack((p[1]-p[0], p[2]-p[0], p[3]-p[0]))) / 6
        h = max(np.linalg.norm(p[i]-p[j]) for i in range(4) for j in range(i))
        volumes[int(d)] += V
        quality.append(6*V/h**3)
    masses = {d: V*solid_material(d, cfg["temperature_K"]).rho
              for d, V in volumes.items() if d in SOLID_DOMAINS}
    shell_mass = 0.
    for q, tag in zip(m.btri, m.tag):
        if tag in SHELL_TAGS:
            mat = shell_material(int(tag), cfg["temperature_K"])
            shell_mass += shell_frame(m.x[q])[0]*mat.thickness*mat.rho
    nb = sum(0 in m.adjacency[int(tag)] and tag not in SYMMETRY for tag in m.tag)
    dense = 16*(10*nb*nb + 4*nb*cfg["bem"]["chief_count"])
    sparse_estimate = 8*2*(180*s.nm + 90*s.np)*cfg["factor_fill_estimate"]
    return {
        "native_counts": m.metadata.get("raw_counts", {}),
        "bounds_m": [m.x.min(axis=0), m.x.max(axis=0)],
        "converted_tetrahedra": len(m.tet), "quadratic_geometry_nodes": len(m.x),
        "solid_nodes": len(s.snode), "pressure_nodes": len(s.pnode),
        "shell_nodes": len(s.hnode), "mechanical_dofs_before_constraints": s.nm,
        "pressure_dofs": s.np, "bem_panels": nb,
        "domain_volumes_m3": dict(volumes), "solid_mass_half_kg": masses,
        "shell_mass_half_kg": shell_mass,
        "minimum_6V_over_h3": min(quality),
        "dense_bem_and_lstsq_estimate_bytes": dense,
        "sparse_factor_estimate_bytes": sparse_estimate,
        "estimate_is_not_measured_memory": True
    }


def build(m: Mesh, cfg: dict) -> Model:
    inc = incidence(m)
    s = number_spaces(m, inc)
    report = describe(m, s, cfg)
    guard(cfg, report["dense_bem_and_lstsq_estimate_bytes"]
          + report["sparse_factor_estimate_bytes"], "coupled reference backend")
    limit = cfg["triplet_entries"]
    accum = {name: Accumulator((s.nm, s.nm), limit)
             for name in ("K", "loss", "C", "M", "drill")}
    ak = {d: Accumulator((s.np, s.np), limit) for d in AIR_DOMAINS}
    am = {d: Accumulator((s.np, s.np), limit) for d in AIR_DOMAINS}
    gc, gm = np.zeros(s.nm), np.zeros(s.nm)
    vc, vm = 0., 0.
    for ci, (q, d) in enumerate(zip(m.tet, m.domain)):
        d = int(d)
        if d in AIR_DOMAINS:
            K, M, _ = scalar_element(m.x[q])
            pd = s.pcell[ci]
            ak[d].add(pd, pd, K)
            am[d].add(pd, pd, M)
        else:
            mat = solid_material(d, cfg["temperature_K"])
            K, M, V = solid_element(m.x[q], mat)
            dofs = s.sdofs(ci)
            accum["K"].add(dofs, dofs, K)
            accum["loss"].add(dofs, dofs, mat.loss*K)
            accum["M"].add(dofs, dofs, M)
            if d == 26:
                np.add.at(gc, 3*s.scell[ci], tetra_integrals(V))
                vc += V
            if d in (23, 27):
                np.add.at(gm, 3*s.scell[ci], tetra_integrals(V))
                vm += V
    if vc <= 0 or vm <= 0:
        raise ValueError("Coil domain 26 or combined steel domains 23/27 are absent")
    gc /= vc
    gm /= vm
    caps = []
    for bi, tag in enumerate(m.tag):
        if tag not in SHELL_TAGS:
            continue
        mat = shell_material(int(tag), cfg["temperature_K"])
        K, M, D, cap = shell_element(m.x[m.btri[bi]], mat, cfg["shell"])
        dofs = s.hdofs(bi)
        for name, value in (
            ("K", K), ("loss", mat.loss*K), ("C", mat.beta*K),
            ("M", M), ("drill", D)
        ):
            accum[name].add(dofs, dofs, value)
        caps.append(cap)
    full = {k: a.finish() for k, a in accum.items()}
    full.update(gc=gc, gm=gm, f=.5*Motor.BL*(gc-gm))
    if cfg["connection"].get("method") == "nodal_spin":
        Cconnect, Tconnect, connector_free, connection_report = nodal_spin_connector(m, s, inc)
        Csym = symmetry_constraints(m, s, inc)
        Csym_reduced = (Csym @ Tconnect).tocsr()
        _, Tsym, symmetry_free, constraint_report = reduce_constraints(Csym_reduced, cfg)
        T = (Tconnect @ Tsym).tocsr()
        free = connector_free[symmetry_free]
        C = sparse.vstack((Cconnect, Csym), format="csr")
        constraint_report["nodal_tie_rows"] = Cconnect.shape[0]
        constraint_report["max_C_times_T"] = normmax(C @ T)
    else:
        Cconnect, connection_report = thickness_connector(m, s, inc, cfg)
        Csym = symmetry_constraints(m, s, inc)
        C, T, free, constraint_report = reduce_constraints(
            sparse.vstack((Cconnect, Csym), format="csr"), cfg)
    (Bi, Be, Cp, ext, panels, normals, areas, wall, ap,
     interface_report) = interface_blocks(m, s, inc, cfg)
    full.update(Bi=Bi, Be=Be)
    reduced = {k: (T.T @ full[k] @ T).tocsr()
               for k in ("K", "loss", "C", "M", "drill")}
    reduced.update({k: np.asarray(T.T @ full[k]).ravel() for k in ("gc", "gm", "f")})
    reduced.update(Bi=(T.T @ Bi).tocsr(), Be=(T.T @ Be).tocsr())
    R = rigid_vectors(m, s)
    report.update(
        connection=connection_report, constraints=constraint_report,
        interfaces=interface_report,
        shear_residual_cap_range=[min(caps), max(caps)],
        coil_volume_half_m3=vc, magnetic_union_volume_half_m3=vm,
        coil_average_of_X_translation=float(gc @ R[:, 0]),
        magnetic_average_of_X_translation=float(gm @ R[:, 0]),
        motor_resultants=[float(full["f"] @ R[:, i]) for i in range(6)],
        stiffness_rigid_residual=normmax(full["K"] @ R) /
            max(normmax(full["K"]), 1e-300),
        matrix_symmetry={
            k: normmax(full[k]-full[k].T) / max(normmax(full[k]), 1e-300)
            for k in ("K", "loss", "C", "M")
        }
    )
    if abs(gc @ R[:, 0] - 1) > 1e-10 or abs(gm @ R[:, 0] - 1) > 1e-10:
        raise ValueError("Normalized motor integration invariant failed")
    if cfg["strict_bundle"] and cfg["connection"].get("method") != "nodal_spin":
        if len(connection_report["shared_shell_boundaries"]) != 12:
            raise ValueError("Expected twelve shell-labeled boundaries touching solids")
    return Model(m, s, full, reduced,
                 {k: v.finish() for k, v in ak.items()},
                 {k: v.finish() for k, v in am.items()},
                 Cp, ext, panels, normals, areas, wall, ap, C, T, free, report)
