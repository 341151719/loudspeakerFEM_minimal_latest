from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np

@dataclass(frozen=True)
class MeshTags:
    outer: int = 101
    speaker_front: int = 201
    speaker_back: int = 202
    baffle: int = 301
    back_wall: int = 302
    axis: int = 401
    air_front: int = 1
    air_rear: int = 2
    air_cavity: int = 3
    air_front_pml: int = 4
    air_rear_pml: int = 5
    hk_front: int = 501
    hk_rear: int = 502


def speaker_polyline_points(a: float = 0.12, n_bezier: int = 9, n_cap: int = 13) -> list[tuple[float, float]]:
    """Approximate the COMSOL speaker line: rational quadratic + cone line + 1 cm cap."""
    p0 = np.array([0.0, -0.03])
    p1 = np.array([0.018, -0.031])
    p2 = np.array([0.03, -0.04])
    w = np.array([1.0, 1.5, 1.0])
    pts = []
    for t in np.linspace(0.0, 1.0, n_bezier):
        b = np.array([(1-t)**2, 2*(1-t)*t, t**2]) * w
        q = (b[0]*p0 + b[1]*p1 + b[2]*p2) / b.sum()
        pts.append(tuple(q))
    pts.append((a - 0.01, 0.0))
    # upper half circle around (a, 0), from a-1cm to a+1cm
    for th in np.linspace(math.pi, 0.0, n_cap)[1:]:
        pts.append((a + 0.01 * math.cos(th), 0.01 * math.sin(th)))
    # remove near duplicates
    out = []
    for p in pts:
        if not out or (abs(out[-1][0]-p[0]) + abs(out[-1][1]-p[1]) > 1e-9):
            out.append(p)
    return out


def _add_polyline(gmsh, pts, lc, tag_prefix=None):
    p_tags = [gmsh.model.geo.addPoint(float(x), float(z), 0.0, lc) for x, z in pts]
    curves = []
    for p0, p1 in zip(p_tags[:-1], p_tags[1:]):
        curves.append(gmsh.model.geo.addLine(p0, p1))
    return p_tags, curves



def _add_polyline_with_endpoints(gmsh, pts, start_tag, end_tag, lc):
    """Create line segments through pts using supplied endpoint point tags."""
    mid_tags = [gmsh.model.geo.addPoint(float(x), float(z), 0.0, lc) for x, z in pts[1:-1]]
    p_tags = [start_tag] + mid_tags + [end_tag]
    curves = []
    for p0, p1 in zip(p_tags[:-1], p_tags[1:]):
        curves.append(gmsh.model.geo.addLine(p0, p1))
    return p_tags, curves

def _add_phys(gmsh, dim, entities, tag, name):
    gmsh.model.addPhysicalGroup(dim, list(entities), tag)
    gmsh.model.setPhysicalName(dim, tag, name)


def generate_gmsh_mesh(
    msh_path: str | Path,
    closed_back: bool = False,
    a: float = 0.12,
    Rair: float = 0.20,
    Rpml: float = 0.06,
    h: float = 0.010,
    h_speaker: float = 0.0035,
    order: int = 1,
):
    """Generate a cracked 2D axisymmetric mesh.

    Coincident curves are deliberately duplicated along the speaker, baffle and closed-box
    walls. This gives independent pressure DOFs on the two sides when using continuous P1.
    """
    import gmsh
    msh_path = Path(msh_path)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("lumped_loudspeaker_axisym_fem")
    tags = MeshTags()
    Rout = Rair + Rpml
    lc = h
    lc_sp = h_speaker

    sp = speaker_polyline_points(a=a)
    p_axis_sp = sp[0]
    p_sp_outer = sp[-1]
    rb = a + 0.01

    def P(x, z, l=lc):
        return gmsh.model.geo.addPoint(float(x), float(z), 0.0, l)
    def L(p0, p1):
        return gmsh.model.geo.addLine(p0, p1)
    def Arc(p0, pc, p1):
        return gmsh.model.geo.addCircleArc(p0, pc, p1)

    # ---------- front air surface ----------
    f_axis0 = P(0.0, p_axis_sp[1], lc_sp)
    f_top = P(0.0, Rout, lc)
    f_right = P(Rout, 0.0, lc)
    f_rb = P(rb, 0.0, lc_sp)
    f_center = P(0.0, 0.0, lc)
    f_axis = L(f_axis0, f_top)
    f_arc = Arc(f_top, f_center, f_right)
    f_baffle = L(f_rb, f_right)  # natural direction left->right; use - in front loop
    _, f_sp_curves = _add_polyline_with_endpoints(gmsh, sp, f_axis0, f_rb, lc_sp)
    # loop: axis up, outer front, baffle right->left, speaker outer->axis
    front_loop = gmsh.model.geo.addCurveLoop([f_axis, f_arc, -f_baffle] + [-c for c in reversed(f_sp_curves)])
    front_surf = gmsh.model.geo.addPlaneSurface([front_loop])

    surfaces = [front_surf]
    outer_curves = [f_arc]
    speaker_front_curves = f_sp_curves
    speaker_back_curves = []
    baffle_curves = [f_baffle]
    back_wall_curves = []
    axis_curves = [f_axis]

    if not closed_back:
        # ---------- rear air surface, transparent back volume ----------
        r_bot = P(0.0, -Rout, lc)
        r_axis0 = P(0.0, p_axis_sp[1], lc_sp)
        r_right = P(Rout, 0.0, lc)
        r_center = P(0.0, 0.0, lc)
        r_rb = P(rb, 0.0, lc_sp)
        r_axis = L(r_bot, r_axis0)
        _, r_sp_curves = _add_polyline_with_endpoints(gmsh, sp, r_axis0, r_rb, lc_sp)
        r_baffle = L(r_rb, r_right)
        r_arc = Arc(r_right, r_center, r_bot)
        rear_loop = gmsh.model.geo.addCurveLoop([r_axis] + r_sp_curves + [r_baffle, r_arc])
        rear_surf = gmsh.model.geo.addPlaneSurface([rear_loop])
        surfaces.append(rear_surf)
        outer_curves.append(r_arc)
        speaker_back_curves.extend(r_sp_curves)
        baffle_curves.append(r_baffle)
        axis_curves.append(r_axis)
    else:
        # ---------- closed cavity behind speaker ----------
        cb = 0.15
        zbox = -0.10
        c_axis_bottom = P(0.0, zbox, lc_sp)
        c_axis_sp = P(0.0, p_axis_sp[1], lc_sp)
        c_rb = P(rb, 0.0, lc_sp)
        c_cb0 = P(cb, 0.0, lc_sp)
        c_cbz = P(cb, zbox, lc_sp)
        c_axis = L(c_axis_bottom, c_axis_sp)
        _, c_sp_curves = _add_polyline_with_endpoints(gmsh, sp, c_axis_sp, c_rb, lc_sp)
        c_baffle_short = L(c_rb, c_cb0)
        c_vert = L(c_cb0, c_cbz)
        c_bottom = L(c_cbz, c_axis_bottom)
        cavity_loop = gmsh.model.geo.addCurveLoop([c_axis] + c_sp_curves + [c_baffle_short, c_vert, c_bottom])
        cavity_surf = gmsh.model.geo.addPlaneSurface([cavity_loop])
        surfaces.append(cavity_surf)
        speaker_back_curves.extend(c_sp_curves)
        baffle_curves.append(c_baffle_short)
        back_wall_curves.extend([c_vert, c_bottom])
        axis_curves.append(c_axis)

        # ---------- exterior rear domain outside the closed box ----------
        e_bot = P(0.0, -Rout, lc)
        e_box_axis = P(0.0, zbox, lc_sp)
        e_cbz = P(cb, zbox, lc_sp)
        e_cb0 = P(cb, 0.0, lc_sp)
        e_right = P(Rout, 0.0, lc)
        e_center = P(0.0, 0.0, lc)
        e_axis = L(e_bot, e_box_axis)
        e_bottom = L(e_box_axis, e_cbz)
        e_vert = L(e_cbz, e_cb0)
        e_baffle = L(e_cb0, e_right)
        e_arc = Arc(e_right, e_center, e_bot)
        exterior_loop = gmsh.model.geo.addCurveLoop([e_axis, e_bottom, e_vert, e_baffle, e_arc])
        exterior_surf = gmsh.model.geo.addPlaneSurface([exterior_loop])
        surfaces.append(exterior_surf)
        outer_curves.append(e_arc)
        baffle_curves.append(e_baffle)
        back_wall_curves.extend([e_bottom, e_vert])
        axis_curves.append(e_axis)

    gmsh.model.geo.synchronize()

    _add_phys(gmsh, 2, [front_surf], tags.air_front, "air_front")
    if not closed_back:
        _add_phys(gmsh, 2, [surfaces[1]], tags.air_rear, "air_rear_open")
    else:
        _add_phys(gmsh, 2, [surfaces[1]], tags.air_cavity, "air_cavity")
        _add_phys(gmsh, 2, [surfaces[2]], tags.air_rear, "air_rear_exterior")
    _add_phys(gmsh, 1, outer_curves, tags.outer, "outer_pml_boundary")
    _add_phys(gmsh, 1, speaker_front_curves, tags.speaker_front, "speaker_front")
    _add_phys(gmsh, 1, speaker_back_curves, tags.speaker_back, "speaker_back")
    _add_phys(gmsh, 1, baffle_curves, tags.baffle, "baffle_hard")
    if back_wall_curves:
        _add_phys(gmsh, 1, back_wall_curves, tags.back_wall, "closed_back_wall")
    _add_phys(gmsh, 1, axis_curves, tags.axis, "axis")

    gmsh.model.mesh.setOrder(order)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min(h_speaker, h) / 2)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.model.mesh.generate(2)
    gmsh.write(str(msh_path))
    gmsh.finalize()
    return msh_path


def generate_closed_back_hk_split_mesh(
    msh_path: str | Path,
    a: float = 0.12,
    Rair: float = 0.20,
    Rpml: float = 0.06,
    h: float = 0.010,
    h_speaker: float = 0.0035,
    order: int = 1,
):
    """Generate a closed-box mesh with a real HK interface at radius `Rair`.

    The original project evaluated HK data on an internal radius by interpolation.
    This mesh splits the physical domain and the PML annulus with physical line
    groups `hk_front` and `hk_rear`, enabling facet quadrature and FE-gradient
    normal flux on the HK surface.  The function deliberately covers the closed
    back geometry because that is the validated wood/HK project configuration.
    """
    import gmsh
    msh_path = Path(msh_path)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("loudspeaker_closed_back_hk_split")
    tags = MeshTags()
    Rout = Rair + Rpml
    rb = a + 0.01
    cb = 0.15
    zbox = -0.10
    sp = speaker_polyline_points(a=a)
    p_axis_sp = sp[0]

    def P(x, z, l=None):
        return gmsh.model.geo.addPoint(float(x), float(z), 0.0, float(h if l is None else l))
    def L(p0, p1):
        return gmsh.model.geo.addLine(p0, p1)
    def Arc(p0, pc, p1):
        return gmsh.model.geo.addCircleArc(p0, pc, p1)

    center = P(0.0, 0.0, h)

    # Shared front HK/PML endpoints (physical interface nodes are shared).
    f_axis0 = P(0.0, p_axis_sp[1], h_speaker)
    f_hk_top = P(0.0, Rair, h)
    f_hk_right = P(Rair, 0.0, h)
    f_out_top = P(0.0, Rout, h)
    f_out_right = P(Rout, 0.0, h)
    f_rb = P(rb, 0.0, h_speaker)

    f_axis_inner = L(f_axis0, f_hk_top)
    f_hk_arc = Arc(f_hk_top, center, f_hk_right)
    f_baffle_inner = L(f_rb, f_hk_right)
    _, f_sp_curves = _add_polyline_with_endpoints(gmsh, sp, f_axis0, f_rb, h_speaker)
    front_inner_loop = gmsh.model.geo.addCurveLoop([f_axis_inner, f_hk_arc, -f_baffle_inner] + [-c for c in reversed(f_sp_curves)])
    front_inner = gmsh.model.geo.addPlaneSurface([front_inner_loop])

    f_axis_pml = L(f_hk_top, f_out_top)
    f_outer_arc = Arc(f_out_top, center, f_out_right)
    f_baffle_pml = L(f_hk_right, f_out_right)
    front_pml_loop = gmsh.model.geo.addCurveLoop([f_axis_pml, f_outer_arc, -f_baffle_pml, -f_hk_arc])
    front_pml = gmsh.model.geo.addPlaneSurface([front_pml_loop])

    # Closed rear cavity uses duplicated wall/speaker nodes to allow pressure discontinuity.
    c_axis_bottom = P(0.0, zbox, h_speaker)
    c_axis_sp = P(0.0, p_axis_sp[1], h_speaker)
    c_rb = P(rb, 0.0, h_speaker)
    c_cb0 = P(cb, 0.0, h_speaker)
    c_cbz = P(cb, zbox, h_speaker)
    c_axis = L(c_axis_bottom, c_axis_sp)
    _, c_sp_curves = _add_polyline_with_endpoints(gmsh, sp, c_axis_sp, c_rb, h_speaker)
    c_baffle_short = L(c_rb, c_cb0)
    c_vert = L(c_cb0, c_cbz)
    c_bottom = L(c_cbz, c_axis_bottom)
    cavity_loop = gmsh.model.geo.addCurveLoop([c_axis] + c_sp_curves + [c_baffle_short, c_vert, c_bottom])
    cavity = gmsh.model.geo.addPlaneSurface([cavity_loop])

    # Rear exterior inner and PML surfaces, with shared HK interface at Rair.
    e_hk_bottom = P(0.0, -Rair, h)
    e_hk_right = P(Rair, 0.0, h)
    e_out_bottom = P(0.0, -Rout, h)
    e_out_right = P(Rout, 0.0, h)
    e_box_axis = P(0.0, zbox, h_speaker)
    e_cbz = P(cb, zbox, h_speaker)
    e_cb0 = P(cb, 0.0, h_speaker)
    e_axis_inner = L(e_hk_bottom, e_box_axis)
    e_bottom = L(e_box_axis, e_cbz)
    e_vert = L(e_cbz, e_cb0)
    e_baffle_inner = L(e_cb0, e_hk_right)
    e_hk_arc = Arc(e_hk_right, center, e_hk_bottom)
    rear_inner_loop = gmsh.model.geo.addCurveLoop([e_axis_inner, e_bottom, e_vert, e_baffle_inner, e_hk_arc])
    rear_inner = gmsh.model.geo.addPlaneSurface([rear_inner_loop])

    e_axis_pml = L(e_out_bottom, e_hk_bottom)
    e_outer_arc = Arc(e_out_right, center, e_out_bottom)
    e_baffle_pml = L(e_hk_right, e_out_right)
    rear_pml_loop = gmsh.model.geo.addCurveLoop([-e_hk_arc, e_baffle_pml, e_outer_arc, e_axis_pml])
    rear_pml = gmsh.model.geo.addPlaneSurface([rear_pml_loop])

    gmsh.model.geo.synchronize()

    _add_phys(gmsh, 2, [front_inner], tags.air_front, "air_front")
    _add_phys(gmsh, 2, [front_pml], tags.air_front_pml, "air_front_pml")
    _add_phys(gmsh, 2, [cavity], tags.air_cavity, "air_cavity")
    _add_phys(gmsh, 2, [rear_inner], tags.air_rear, "air_rear_exterior")
    _add_phys(gmsh, 2, [rear_pml], tags.air_rear_pml, "air_rear_pml")
    _add_phys(gmsh, 1, [f_outer_arc, e_outer_arc], tags.outer, "outer_pml_boundary")
    _add_phys(gmsh, 1, [f_hk_arc], tags.hk_front, "hk_front_Rair")
    _add_phys(gmsh, 1, [e_hk_arc], tags.hk_rear, "hk_rear_Rair")
    _add_phys(gmsh, 1, f_sp_curves, tags.speaker_front, "speaker_front")
    _add_phys(gmsh, 1, c_sp_curves, tags.speaker_back, "speaker_back")
    _add_phys(gmsh, 1, [f_baffle_inner, f_baffle_pml, c_baffle_short, e_baffle_inner, e_baffle_pml], tags.baffle, "baffle_hard")
    _add_phys(gmsh, 1, [c_vert, c_bottom, e_bottom, e_vert], tags.back_wall, "closed_back_wall")
    _add_phys(gmsh, 1, [f_axis_inner, f_axis_pml, c_axis, e_axis_inner, e_axis_pml], tags.axis, "axis")

    gmsh.model.mesh.setOrder(order)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min(h_speaker, h) / 2.0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.model.mesh.generate(2)
    gmsh.write(str(msh_path))
    gmsh.finalize()
    return msh_path
