from __future__ import annotations

import numpy as np

from .elements import unit_gauss, tri_rule

FOUR_PI = 4 * np.pi


def point_triangle_distance(x: np.ndarray, tri: np.ndarray) -> float:
    a, b, c = tri
    e1, e2 = b-a, c-a
    n = np.cross(e1, e2)
    nn = n @ n
    if nn == 0:
        raise ValueError("Degenerate BEM panel")
    projection = x - ((x-a) @ n)/nn*n
    uv = np.linalg.solve(np.array([[e1@e1, e1@e2], [e1@e2, e2@e2]]),
                         np.array([(projection-a)@e1, (projection-a)@e2]))
    if uv.min() >= 0 and uv.sum() <= 1:
        return float(abs((x-a)@n)/np.sqrt(nn))
    distances = []
    for p, q in ((a, b), (b, c), (c, a)):
        t = np.clip((x-p)@(q-p)/((q-p)@(q-p)), 0, 1)
        distances.append(np.linalg.norm(x-p-t*(q-p)))
    return float(min(distances))


def _angular_integral(fun, options: dict, a=0., b=1., depth=0):
    n = options["radial_order"]
    x, w = unit_gauss(n)
    xx, ww = unit_gauss(2*n)
    low = (b-a) * np.einsum("i,ij->j", w, fun(a+(b-a)*x))
    high = (b-a) * np.einsum("i,ij->j", ww, fun(a+(b-a)*xx))
    error = np.max(np.abs(high-low))
    scale = max(float(np.max(np.abs(high))), 1e-14)
    if error <= options["quadrature_rtol"]*scale:
        return high
    if depth >= options["max_depth"]:
        raise ArithmeticError(
            f"BEM angular quadrature unresolved: error={error:g}, scale={scale:g}"
        )
    mid = (a+b)/2
    return (_angular_integral(fun, options, a, mid, depth+1)
            + _angular_integral(fun, options, mid, b, depth+1))


def panel_integral(x: np.ndarray, tri: np.ndarray, normal: np.ndarray,
                   k: float, options: dict) -> tuple[complex, complex]:
    """
    Return (single layer, source-normal double layer).
    Radial integration is analytic after a signed projected-point Duffy split.
    No distance is replaced by epsilon.
    """
    raw_n = np.cross(tri[1]-tri[0], tri[2]-tri[0])
    geom_n = raw_n/np.linalg.norm(raw_n)
    height_geom = (x-tri[0])@geom_n
    projected = x-height_geom*geom_n
    signed_h = (x-projected)@normal
    h = abs(height_geom)
    size = max(np.linalg.norm(tri[i]-tri[j])
               for i in range(3) for j in range(i))
    on_plane = h <= 64*np.finfo(float).eps*size
    if on_plane:
        h, signed_h = 0., 0.
    total = np.zeros(2, complex)
    for j in range(3):
        a, b = tri[j]-projected, tri[(j+1)%3]-projected
        jac = np.cross(a, b)@geom_n
        if abs(jac) <= 32*np.finfo(float).eps*size**2:
            continue

        def integrand(v):
            ray = (1-v[:, None])*a + v[:, None]*b
            L2 = np.einsum("ij,ij->i", ray, ray)
            end = np.sqrt(h*h+L2)
            delta = L2/(end+h)  # end-h, cancellation-free
            if k == 0:
                S = delta/L2
            else:
                S = np.exp(-1j*k*h)*np.expm1(-1j*k*delta)/(-1j*k*L2)
            if on_plane:
                D = np.zeros_like(S)
            else:
                D = ((signed_h/h)*np.exp(-1j*k*h)
                     - signed_h*np.exp(-1j*k*end)/end)/L2
            return jac/FOUR_PI*np.column_stack((S, D))
        total += _angular_integral(integrand, options)
    return complex(total[0]), complex(total[1])


def operators(points: np.ndarray, panels: np.ndarray, normals: np.ndarray,
              k: float, options: dict, mirror: bool = True, progress=None):
    """Constant source panels, point tests. Returns D and S, without jump terms."""
    points, panels, normals = map(np.asarray, (points, panels, normals))
    nt, ns = len(points), len(panels)
    D, S = np.zeros((nt, ns), complex), np.zeros((nt, ns), complex)
    transforms = [np.array([1., 1., 1.])]
    if mirror:
        transforms.append(np.array([1., -1., 1.]))
    for reflection_index, reflection in enumerate(transforms):
        tri = panels*reflection
        n = normals*reflection  # polar normal, not the reflected corner cross product
        center = tri.mean(axis=1)
        radius = np.linalg.norm(tri-center[:, None], axis=2).max(axis=1)
        area2 = np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0],
                                        tri[:, 2]-tri[:, 0]), axis=1)
        # Increase far-panel quadrature for electrical-size-large panels.
        order = max(options["order"], int(np.ceil(k*radius.max(initial=0)))+2)
        if order > 64:
            raise ValueError("Panel electrical size requires refinement, not order >64")
        L, w = tri_rule(order)
        source = np.einsum("qv,svd->sqd", L, tri)
        weights = area2[:, None]*w
        for i, x in enumerate(points):
            distance = np.linalg.norm(center-x, axis=1)
            near = distance-radius < options["near_ratio"]*radius
            far_indices = np.flatnonzero(~near)
            for begin in range(0, len(far_indices), 256):
                ids = far_indices[begin:begin+256]
                delta = x-source[ids]
                r = np.linalg.norm(delta, axis=2)
                if np.any(r == 0):
                    raise ArithmeticError("Unexpected zero distance in far quadrature")
                G = np.exp(-1j*k*r)/(FOUR_PI*r)
                dot = np.einsum("sqd,sd->sq", delta, n[ids])
                S[i, ids] += np.sum(weights[ids]*G, axis=1)
                D[i, ids] += np.sum(
                    weights[ids]*G*(1/r+1j*k)*dot/r, axis=1)
            for j in np.flatnonzero(near):
                v, d = panel_integral(x, tri[j], n[j], k, options)
                S[i, j] += v
                D[i, j] += d
            if progress and (i == 0 or (i+1) % 32 == 0 or i+1 == nt):
                progress({"reflection": reflection_index,
                          "completed_rows": i+1, "total_rows": nt})
    if not np.isfinite(D).all() or not np.isfinite(S).all():
        raise FloatingPointError("Nonfinite BEM operator")
    return D, S


def chief_points(mesh, panels: np.ndarray, options: dict) -> np.ndarray:
    """Each point is a strict centroid of an actual finite-volume tetrahedron."""
    count = options["chief_count"]
    if count < 1:
        raise ValueError("At least one CHIEF point is required")
    candidates_count = min(options["chief_candidates"], len(mesh.tet))
    ids = np.unique(np.linspace(0, len(mesh.tet)-1, candidates_count, dtype=int))
    candidates = mesh.x[mesh.tet[ids]].mean(axis=1)
    reflection = np.array([1., -1., 1.])
    all_panels = np.concatenate((panels, panels*reflection))
    centers = all_panels.mean(axis=1)
    radii = np.linalg.norm(all_panels-centers[:, None], axis=2).max(axis=1)
    clearance = []
    for x in candidates:
        lower = np.linalg.norm(centers-x, axis=1)-radii
        best = np.inf
        for j in np.argsort(lower):
            if lower[j] >= best:
                break
            best = min(best, point_triangle_distance(x, all_panels[j]))
        clearance.append(best)
    clearance = np.asarray(clearance)
    valid = clearance >= options["chief_min_clearance_m"]
    candidates, clearance = candidates[valid], clearance[valid]
    if len(candidates) < count:
        raise ValueError("Insufficient interior CHIEF candidates with required clearance")
    selected = [int(np.argmax(clearance))]
    while len(selected) < count:
        distance = np.min(np.linalg.norm(
            candidates[:, None]-candidates[selected][None], axis=2), axis=1)
        score = distance*np.sqrt(clearance)
        score[selected] = -1
        selected.append(int(np.argmax(score)))
    return candidates[selected]


def coupled_operators(model, k: float, options: dict, progress=None):
    points = model.panels.mean(axis=1)
    D, S = operators(
        points, model.panels, model.normals, k, options, options.get("mirror", True),
        progress=(lambda info: progress({"operator": "boundary", **info}))
        if progress else None)
    chiefs = chief_points(model.mesh, model.panels, options)
    Dc, Sc = operators(
        chiefs, model.panels, model.normals, k, options, options.get("mirror", True),
        progress=(lambda info: progress({"operator": "chief", **info}))
        if progress else None)
    nb, nc = len(points), len(chiefs)
    Lp = np.vstack((.5*np.eye(nb)-D, Dc))
    Lq = np.vstack((S, -Sc))
    weights = np.r_[
        np.sqrt(model.areas/model.areas.mean()),
        np.full(nc, options["chief_weight"]*np.sqrt(nb/nc))
    ]
    return Lp, Lq, weights, chiefs


def sphere_mesh(level: int = 1, half: bool = False, radius: float = 1.):
    vertices = np.array([
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
    ], float)
    faces = np.array([
        [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
        [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]
    ])
    for _ in range(level):
        points, mid = vertices.tolist(), {}
        def midpoint(a, b):
            key = tuple(sorted((int(a), int(b))))
            if key not in mid:
                q = vertices[a]+vertices[b]
                q /= np.linalg.norm(q)
                mid[key] = len(points)
                points.append(q.tolist())
            return mid[key]
        new = []
        for a, b, c in faces:
            ab, ac, bc = midpoint(a, b), midpoint(a, c), midpoint(b, c)
            new.extend([[a, ab, ac], [b, bc, ab], [c, ac, bc], [ab, bc, ac]])
        vertices, faces = np.array(points), np.array(new)
    tri = radius*vertices[faces]
    n = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    n /= np.linalg.norm(n, axis=1)[:, None]
    wrong = np.einsum("ij,ij->i", n, tri.mean(axis=1)) < 0
    tri[wrong] = tri[wrong][:, [0, 2, 1]]
    n[wrong] *= -1
    if half:
        keep = tri.mean(axis=1)[:, 1] < 0
        tri, n = tri[keep], n[keep]
    return tri, n


def sphere_check(level: int, ka: float, options: dict, half: bool = False) -> dict:
    from scipy.linalg import lstsq
    tri, n = sphere_mesh(level, half)
    x = tri.mean(axis=1)
    r = np.linalg.norm(x, axis=1)
    k = ka
    p_exact = np.exp(-1j*k*(r-1))/r
    q = (-1/r-1j*k)*p_exact*np.einsum("ij,ij->i", n, x/r[:, None])
    D, S = operators(x, tri, n, k, options, half)
    chiefs = np.array([[.13, -.17, .11], [-.21, -.12, .07],
                       [.08, -.24, -.19], [.03, -.04, .02]])
    Dc, Sc = operators(chiefs, tri, n, k, options, half)
    A = np.vstack((.5*np.eye(len(x))-D, Dc))
    b = np.r_[-S@q, Sc@q]
    p, _, rank, singular = lstsq(A, b, cond=options["rcond"], lapack_driver="gelsd")
    obs = np.array([[2., 0, 0], [0, 0, 2.]])
    De, Se = operators(obs, tri, n, k, options, half)
    evaluated = De@p-Se@q
    expected = np.full(2, .5*np.exp(-1j*k))
    return {
        "level": level, "ka": ka, "half": half, "panels": len(tri),
        "trace_relative_error": np.linalg.norm(p-p_exact)/np.linalg.norm(p_exact),
        "exterior_relative_error": np.linalg.norm(evaluated-expected)/np.linalg.norm(expected),
        "chief_relative_residual": np.linalg.norm(Dc@p-Sc@q)/np.linalg.norm(p),
        "rank": rank, "condition_estimate": singular[0]/singular[-1],
        "evaluated_pressure": evaluated
    }
