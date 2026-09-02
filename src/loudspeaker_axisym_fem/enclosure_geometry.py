"""Deterministic, geometry-only reference meshes for the enclosure phase 2 audit.

The geometry in this module is deliberately independent from ``meshgen.py`` and
from the production loudspeaker polyline.  It is a small, explicit demonstrator
geometry whose pressure domains can be rotated about the r=0 axis.  No solver,
Helmholtz model, or FEM assembly is imported here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from .enclosure_schema import EnclosureConfig, load_enclosure_config


# Physical tags are part of the phase-2 file contract.  Entity tags below are
# allocated in deterministic blocks, but callers must use these physical tags
# and names rather than Gmsh's generated entity numbers.
DOMAIN_PHYSICAL_TAGS: dict[str, int] = {
    "air_front_free": 1001,
    "air_side_free": 1002,
    "air_rear_free": 1003,
    "air_cavity": 1004,
    "air_rear_opening": 1005,
    "air_port": 1006,
    "air_pml_front": 1007,
    "air_pml_rear": 1008,
    "rigid_driver_displacement": 1009,
    "rigid_comparison_equalizer": 1010,
    "rigid_pr_back_mechanism": 1011,
}

BOUNDARY_PHYSICAL_TAGS: dict[str, int] = {
    "outer_pml_boundary": 2001,
    "hk_front": 2002,
    "hk_rear": 2003,
    "axis": 2004,
    "cabinet_front_wall": 2005,
    "cabinet_side_wall": 2006,
    "cabinet_rear_wall": 2007,
    "reference_planar_piston_front": 2008,
    "reference_planar_piston_back": 2009,
    "driver_side_wall": 2010,
    "rigid_driver_cap": 2011,
    "comparison_equalizer_face": 2012,
    "comparison_equalizer_wall": 2013,
    "rear_opening_cavity": 2014,
    "rear_opening_exterior": 2015,
    "rear_opening_wall": 2016,
    "port_wall": 2017,
    "port_cavity_opening": 2018,
    "port_exterior_opening": 2019,
    "pr_cavity_face": 2020,
    "pr_exterior_face": 2021,
    "pr_side_wall": 2022,
    "freefield_front_side_interface": 2023,
    "freefield_rear_side_interface": 2024,
    "rear_tip_shell_interface": 2025,
    "pml_mid_interface": 2026,
}

PHYSICAL_TAGS: dict[str, int] = {
    **DOMAIN_PHYSICAL_TAGS,
    **BOUNDARY_PHYSICAL_TAGS,
}

REFERENCE_CONFIGS: dict[str, str] = {
    "A": "open_back.json",
    "B": "sealed_lossless.json",
    "C": "sealed_thermoviscous.json",
    "D": "vented_rear_coaxial.json",
    "E": "passive_radiator_rear_coaxial.json",
}


def case_id_for_config(case: str) -> str:
    """Return the comparison ID for a validated phase-1 case name."""

    return {
        "open_back": "A",
        "sealed_lossless": "B",
        "sealed_thermoviscous": "C",
        "vented_rear_coaxial": "D",
        "passive_radiator_rear_coaxial": "E",
        # The base file is a schema demonstrator, not a sixth comparison case.
        "base_axisym": "B",
    }[case]


def geometry_key_for_case(case_id: str) -> str:
    """Return the geometry family; C must be byte-for-byte geometry-equivalent to B."""

    if case_id in {"B", "C"}:
        return "sealed"
    return case_id.lower()


def expected_domain_names(case_id: str) -> tuple[str, ...]:
    names = [
        "air_front_free",
        "air_side_free",
        "air_rear_free",
        "air_cavity",
        "air_pml_front",
        "air_pml_rear",
        "rigid_driver_displacement",
    ]
    if case_id == "A":
        names += ["air_rear_opening", "rigid_comparison_equalizer"]
    elif case_id == "D":
        names += ["air_port", "rigid_comparison_equalizer"]
    elif case_id == "E":
        names += ["rigid_pr_back_mechanism"]
    else:
        names += ["rigid_comparison_equalizer"]
    return tuple(names)


def expected_boundary_names(case_id: str) -> tuple[str, ...]:
    names = [
        "outer_pml_boundary",
        "hk_front",
        "hk_rear",
        "axis",
        "cabinet_front_wall",
        "cabinet_side_wall",
        "cabinet_rear_wall",
        "reference_planar_piston_front",
        "reference_planar_piston_back",
        "driver_side_wall",
        "rigid_driver_cap",
        "freefield_front_side_interface",
        "freefield_rear_side_interface",
        "pml_mid_interface",
    ]
    if case_id == "A":
        names += [
            "comparison_equalizer_face",
            "comparison_equalizer_wall",
            "rear_opening_cavity",
            "rear_opening_exterior",
            "rear_opening_wall",
        ]
    elif case_id == "D":
        names += [
            "comparison_equalizer_face",
            "comparison_equalizer_wall",
            "port_wall",
            "port_cavity_opening",
            "port_exterior_opening",
            "rear_tip_shell_interface",
        ]
    elif case_id == "E":
        names += ["pr_cavity_face", "pr_exterior_face", "pr_side_wall"]
    else:
        names += ["comparison_equalizer_face", "comparison_equalizer_wall"]
    return tuple(names)


@dataclass(frozen=True)
class MeshGenerationResult:
    path: Path
    case_id: str
    level: str
    config_path: Path
    geometry_key: str
    physical_groups: dict[str, dict[str, int | str]]
    geometry: dict[str, float | str | bool]


class _GeoBuilder:
    """Small deterministic wrapper around the Gmsh built-in geometry kernel."""

    def __init__(self, gmsh: Any) -> None:
        self.gmsh = gmsh
        self._point_tag = 1
        self._curve_tag = 1001
        self._loop_tag = 3001
        self._surface_tag = 4001
        self.points: dict[str, tuple[int, float, float]] = {}
        self.curves: dict[str, tuple[int, int, int]] = {}
        self.groups: dict[tuple[int, str], set[int]] = {}

    def point(self, key: str, r: float, z: float, mesh_size: float) -> int:
        r = float(r)
        z = float(z)
        old = self.points.get(key)
        if old is not None:
            if not (math.isclose(old[1], r, rel_tol=0.0, abs_tol=1.0e-13) and math.isclose(old[2], z, rel_tol=0.0, abs_tol=1.0e-13)):
                raise ValueError(f"point key {key!r} was reused with different coordinates")
            return old[0]
        tag = self._point_tag
        self._point_tag += 1
        self.gmsh.model.geo.addPoint(r, z, 0.0, float(mesh_size), tag)
        self.points[key] = (tag, r, z)
        return tag

    def _oriented_existing(self, key: str, start: int, end: int) -> int | None:
        old = self.curves.get(key)
        if old is None:
            return None
        tag, old_start, old_end = old
        if (start, end) == (old_start, old_end):
            return tag
        if (start, end) == (old_end, old_start):
            return -tag
        raise ValueError(f"curve key {key!r} was reused with different endpoints")

    def line(self, key: str, start: int, end: int, *groups: str) -> int:
        old = self._oriented_existing(key, start, end)
        if old is not None:
            base = abs(old)
            for group in groups:
                self.add_group(1, group, base)
            return old
        tag = self._curve_tag
        self._curve_tag += 1
        self.gmsh.model.geo.addLine(start, end, tag)
        self.curves[key] = (tag, start, end)
        for group in groups:
            self.add_group(1, group, tag)
        return tag

    def arc(self, key: str, start: int, center: int, end: int, *groups: str) -> int:
        old = self._oriented_existing(key, start, end)
        if old is not None:
            base = abs(old)
            for group in groups:
                self.add_group(1, group, base)
            return old
        tag = self._curve_tag
        self._curve_tag += 1
        self.gmsh.model.geo.addCircleArc(start, center, end, tag)
        self.curves[key] = (tag, start, end)
        for group in groups:
            self.add_group(1, group, tag)
        return tag

    def add_group(self, dimension: int, name: str, entity: int) -> None:
        self.groups.setdefault((int(dimension), name), set()).add(abs(int(entity)))

    def surface(self, key: str, loop_curves: Iterable[int], domain_name: str) -> int:
        loop_tag = self._loop_tag
        self._loop_tag += 1
        self.gmsh.model.geo.addCurveLoop(list(loop_curves), loop_tag)
        tag = self._surface_tag
        self._surface_tag += 1
        self.gmsh.model.geo.addPlaneSurface([loop_tag], tag)
        self.add_group(2, domain_name, tag)
        return tag


def _point(builder: _GeoBuilder, key: str, r: float, z: float, lc: float) -> int:
    return builder.point(key, r, z, lc)


def _build_outer_field(
    builder: _GeoBuilder,
    geometry: Mapping[str, float],
    case_id: str,
    lc_global: float,
) -> dict[str, float]:
    """Build front/side/rear free field and the two closed PML half-annuli."""

    gm = builder.gmsh
    ri = float(geometry["inner_radius_m"])
    ro = float(geometry["outer_radius_m"])
    wall = float(geometry["wall_thickness_m"])
    depth = float(geometry["inner_depth_m"])
    h = float(geometry["pml_inner_radius_m"])
    rout = h + float(geometry["pml_thickness_m"])
    z_front = wall
    z_rear = -depth - wall
    if abs(z_front) >= h or abs(z_rear) >= h:
        raise ValueError("the configured HK radius does not contain the cabinet")

    x_front = math.sqrt(h * h - z_front * z_front)
    x_rear = math.sqrt(h * h - z_rear * z_rear)
    center = _point(builder, "outer_circle_center", 0.0, 0.0, lc_global)

    # Inner and outer circle vertices.  The centre is only a circle-arc control
    # point and is not part of a pressure domain boundary.
    p_it = _point(builder, "hk_top", 0.0, h, lc_global)
    p_im = _point(builder, "hk_mid", h, 0.0, lc_global)
    p_ib = _point(builder, "hk_bottom", 0.0, -h, lc_global)
    p_ot = _point(builder, "outer_top", 0.0, rout, lc_global)
    p_om = _point(builder, "outer_mid", rout, 0.0, lc_global)
    p_ob = _point(builder, "outer_bottom", 0.0, -rout, lc_global)
    p_fc = _point(builder, "hk_front_cabinet_intersection", x_front, z_front, lc_global)
    p_rc = _point(builder, "hk_rear_cabinet_intersection", x_rear, z_rear, lc_global)
    p_axis_front = _point(builder, "front_field_axis", 0.0, z_front, lc_global)
    p_axis_rear = _point(builder, "rear_field_axis", 0.0, z_rear, lc_global)
    p_front_side = _point(builder, "front_outer_cabinet_corner", ro, z_front, lc_global)
    p_rear_side = _point(builder, "rear_outer_cabinet_corner", ro, z_rear, lc_global)
    p_driver_front = _point(builder, "driver_front_outer_radius", float(geometry["driver_radius_m"]), z_front, lc_global)

    builder.arc("hk_front_cabinet", p_it, center, p_fc, "hk_front")
    builder.arc("hk_front_side", p_fc, center, p_im, "hk_front")
    builder.arc("hk_rear_side", p_im, center, p_rc, "hk_rear")

    # The lower HK arc is split around a D-port end so the port remains wholly
    # inside free field and never reaches the PML domain.
    if case_id == "D":
        port_radius = float(geometry["port_radius_m"])
        port_length = float(geometry["port_length_m"])
        p_port_circle = _point(builder, "hk_port_intersection", math.sqrt(h * h - (-depth - port_length) ** 2), -depth - port_length, lc_global)
        builder.arc("hk_rear_shell", p_rc, center, p_port_circle, "hk_rear")
        builder.arc("hk_rear_tip", p_port_circle, center, p_ib, "hk_rear")
    else:
        p_port_circle = None
        builder.arc("hk_rear_cap", p_rc, center, p_ib, "hk_rear")

    builder.arc("outer_front", p_ot, center, p_om, "outer_pml_boundary")
    builder.arc("outer_rear", p_om, center, p_ob, "outer_pml_boundary")
    builder.line("pml_front_axis", p_ot, p_it, "axis")
    builder.line("pml_rear_axis", p_ib, p_ob, "axis")
    builder.line("pml_mid", p_im, p_om, "pml_mid_interface")

    builder.surface(
        "pml_front_surface",
        [
            builder.curves["pml_front_axis"][0],
            builder.curves["hk_front_cabinet"][0],
            builder.curves["hk_front_side"][0],
            builder.curves["pml_mid"][0],
            -builder.curves["outer_front"][0],
        ],
        "air_pml_front",
    )
    lower_hk = [
        builder.curves["hk_rear_side"][0],
        builder.curves["hk_rear_shell"][0] if case_id == "D" else builder.curves["hk_rear_cap"][0],
    ]
    if case_id == "D":
        lower_hk.append(builder.curves["hk_rear_tip"][0])
    builder.surface(
        "pml_rear_surface",
        [
            builder.curves["outer_rear"][0],
            -builder.curves["pml_rear_axis"][0],
            *[-x for x in reversed(lower_hk)],
            builder.curves["pml_mid"][0],
        ],
        "air_pml_rear",
    )

    # Front cap: driver front trace, rigid baffle, and the shared side-field
    # interface are all distinct named curves.
    p_front_side_hk = p_fc
    builder.line("front_axis_free", p_axis_front, p_it, "axis")
    builder.line("free_front_side", p_front_side_hk, p_front_side, "freefield_front_side_interface")
    builder.line("cabinet_front_outer", p_front_side, p_driver_front, "cabinet_front_wall")
    builder.line(
        "reference_planar_piston_front",
        p_driver_front,
        p_axis_front,
        "reference_planar_piston_front",
    )
    builder.surface(
        "front_free_surface",
        [
            builder.curves["front_axis_free"][0],
            builder.curves["hk_front_cabinet"][0],
            builder.curves["free_front_side"][0],
            builder.curves["cabinet_front_outer"][0],
            builder.curves["reference_planar_piston_front"][0],
        ],
        "air_front_free",
    )

    # The side field runs between the outer cabinet side and the right-hand
    # inner HK arc and connects the front and rear exterior components.
    builder.line("cabinet_side_outer", p_rear_side, p_front_side, "cabinet_side_wall")
    builder.line("free_rear_side", p_rear_side, p_rc, "freefield_rear_side_interface")
    builder.surface(
        "side_free_surface",
        [
            -builder.curves["cabinet_side_outer"][0],
            builder.curves["free_rear_side"][0],
            -builder.curves["hk_rear_side"][0],
            -builder.curves["hk_front_side"][0],
            builder.curves["free_front_side"][0],
        ],
        "air_side_free",
    )

    # Rear exterior is case-specific because D has a port shell and A has a
    # through-wall opening.  The function returns the points needed by the
    # feature connector builder.
    feature = {
        "p_rear_circle": p_rc,
        "p_axis_rear": p_axis_rear,
        "p_rear_side": p_rear_side,
        "p_inner_bottom": p_ib,
        "p_port_circle": p_port_circle,
        "p_axis_bottom": p_ib,
        "z_front_outer": z_front,
        "z_rear_outer": z_rear,
        "h": h,
        "rout": rout,
        "x_front": x_front,
        "x_rear": x_rear,
    }
    return feature


def _build_driver_and_cavity(
    builder: _GeoBuilder,
    geometry: Mapping[str, float],
    volume: Mapping[str, float],
    case_id: str,
    lc_feature: float,
) -> dict[str, float]:
    ri = float(geometry["inner_radius_m"])
    depth = float(geometry["inner_depth_m"])
    rd = float(geometry["driver_radius_m"])
    z_inner_rear = -depth
    driver_length = float(volume["driver_displacement_m3"]) / (math.pi * rd * rd)
    z_driver_back = -driver_length

    # Driver exclusion body: its portion between z=0 and z_driver_back is a
    # real rigid 2-D area, so its rotated volume is not an algebraic report-only
    # correction.  The front trace is on a duplicated exterior line.
    p_db_axis = _point(builder, "driver_back_axis", 0.0, z_driver_back, lc_feature)
    p_db = _point(builder, "driver_back_radius", rd, z_driver_back, lc_feature)
    p_dside = _point(builder, "driver_side_front", rd, 0.0, lc_feature)
    p_daxis_front = _point(builder, "driver_body_axis_front", 0.0, 0.0, lc_feature)
    builder.line(
        "reference_planar_piston_back",
        p_db_axis,
        p_db,
        "reference_planar_piston_back",
    )
    builder.line("driver_side", p_db, p_dside, "driver_side_wall")
    builder.line("driver_body_cap", p_dside, p_daxis_front, "rigid_driver_cap")
    builder.line("driver_body_axis", p_daxis_front, p_db_axis, "axis")
    builder.surface(
        "rigid_driver_surface",
        [
            builder.curves["reference_planar_piston_back"][0],
            builder.curves["driver_side"][0],
            builder.curves["driver_body_cap"][0],
            builder.curves["driver_body_axis"][0],
        ],
        "rigid_driver_displacement",
    )

    # Rear comparison occupancy.  For D the declared equalizer is the annulus
    # left outside the explicit circular port; for A it is outside the open
    # rear aperture.  For B/C it is a full rear disk.  E uses the PR mechanism
    # below instead.
    z_eq_top = z_inner_rear
    z_pr_face = z_inner_rear
    feature_radius = 0.0
    equalizer_volume = 0.0
    if case_id == "A":
        feature_radius = float(geometry["rear_opening_radius_m"])
        equalizer_volume = float(volume["reserved_rear_feature_m3"])
    elif case_id == "D":
        feature_radius = float(geometry["port_radius_m"])
        equalizer_volume = float(volume["fair_comparison_equalization_m3"])
    elif case_id in {"B", "C"}:
        equalizer_volume = float(volume["reserved_rear_feature_m3"])

    if case_id in {"A", "D"}:
        annulus_area = math.pi * (ri * ri - feature_radius * feature_radius)
        if annulus_area <= 0.0:
            raise ValueError("rear equalizer annulus has no positive area")
        equalizer_depth = equalizer_volume / annulus_area
        z_eq_top = z_inner_rear + equalizer_depth
    elif case_id in {"B", "C"}:
        equalizer_depth = equalizer_volume / (math.pi * ri * ri)
        z_eq_top = z_inner_rear + equalizer_depth
    else:
        equalizer_depth = 0.0

    # Common cavity front and side boundary.
    p_c_dside = p_dside
    p_c_front_outer = _point(builder, "cavity_front_outer_radius", ri, 0.0, lc_feature)
    p_c_side_split = _point(builder, "cavity_side_rear_inner_wall", ri, z_inner_rear, lc_feature)
    builder.line("cavity_baffle", p_c_dside, p_c_front_outer, "cabinet_front_wall")

    if case_id in {"B", "C"}:
        p_eq_top_axis = _point(builder, "equalizer_top_axis", 0.0, z_eq_top, lc_feature)
        p_eq_top_outer = _point(builder, "equalizer_top_outer", ri, z_eq_top, lc_feature)
        builder.line("cavity_side_front", p_c_front_outer, p_eq_top_outer, "cabinet_side_wall")
        builder.line("equalizer_outer_wall", p_c_side_split, p_eq_top_outer, "comparison_equalizer_wall")
        builder.line("equalizer_top", p_eq_top_outer, p_eq_top_axis, "comparison_equalizer_face")
        p_axis_cavity = p_eq_top_axis
        cavity_loop = [
            builder.curves["reference_planar_piston_back"][0],
            builder.curves["driver_side"][0],
            builder.curves["cavity_baffle"][0],
            builder.curves["cavity_side_front"][0],
            builder.curves["equalizer_top"][0],
            builder.line("cavity_axis", p_axis_cavity, p_db_axis, "axis"),
        ]
        builder.surface("cavity_surface", cavity_loop, "air_cavity")

        p_eq_bottom_axis = _point(builder, "equalizer_bottom_axis", 0.0, z_inner_rear, lc_feature)
        builder.line("equalizer_bottom", p_eq_bottom_axis, p_c_side_split, "cabinet_rear_wall")
        builder.line("equalizer_axis", p_eq_bottom_axis, p_eq_top_axis, "axis")
        builder.surface(
            "rigid_equalizer_surface",
            [
                builder.curves["equalizer_bottom"][0],
                builder.curves["equalizer_outer_wall"][0],
                builder.curves["equalizer_top"][0],
                -builder.curves["equalizer_axis"][0],
            ],
            "rigid_comparison_equalizer",
        )

    elif case_id in {"A", "D"}:
        a = feature_radius
        p_eq_top_inner = _point(builder, "equalizer_top_inner", a, z_eq_top, lc_feature)
        p_eq_inner_top = p_eq_top_inner
        p_eq_inner_bottom = _point(builder, "equalizer_inner_bottom", a, z_inner_rear, lc_feature)
        if case_id == "D":
            z_port_in = z_inner_rear + float(geometry["port_penetration_m"])
            p_open_axis = _point(builder, "rear_feature_inner_axis", 0.0, z_port_in, lc_feature)
        else:
            p_open_axis = _point(builder, "rear_feature_inner_axis", 0.0, z_inner_rear, lc_feature)
        builder.line("equalizer_outer_wall", p_c_side_split, p_eq_top_outer := _point(builder, "equalizer_top_outer", ri, z_eq_top, lc_feature), "comparison_equalizer_wall")
        builder.line("equalizer_top", p_eq_top_outer, p_eq_top_inner, "comparison_equalizer_face")

        if case_id == "A":
            builder.line("equalizer_inner_wall", p_eq_inner_bottom, p_eq_inner_top, "comparison_equalizer_wall")
            builder.line("rear_opening_inner", p_open_axis, p_eq_inner_bottom, "rear_opening_cavity")
            inner_feature_curve = builder.curves["equalizer_inner_wall"][0]
        else:
            z_port_in = z_inner_rear + float(geometry["port_penetration_m"])
            p_port_inner = _point(builder, "port_inner_end", a, z_port_in, lc_feature)
            builder.line("port_wall_cavity", p_eq_inner_bottom, p_port_inner, "port_wall")
            builder.line("equalizer_inner_upper", p_port_inner, p_eq_inner_top, "comparison_equalizer_wall")
            inner_feature_curve = builder.curves["equalizer_inner_upper"][0]
            # The port opening is above its explicitly occupied penetration.
            builder.line("port_cavity_top", p_open_axis, p_port_inner, "port_cavity_opening")

        builder.line("cavity_side_front", p_c_front_outer, p_eq_top_outer, "cabinet_side_wall")

        cavity_loop = [
            builder.curves["reference_planar_piston_back"][0],
            builder.curves["driver_side"][0],
            builder.curves["cavity_baffle"][0],
            builder.curves["cavity_side_front"][0],
            builder.curves["equalizer_top"][0],
            -inner_feature_curve,
            -builder.curves["rear_opening_inner"][0] if case_id == "A" else -builder.curves["port_cavity_top"][0],
            builder.line("cavity_axis", p_open_axis, p_db_axis, "axis"),
        ]
        builder.surface("cavity_surface", cavity_loop, "air_cavity")

        p_eq_bottom_outer = p_c_side_split
        builder.line("equalizer_bottom", p_eq_bottom_outer, p_eq_inner_bottom, "cabinet_rear_wall")
        body_loop = [
            builder.curves["equalizer_bottom"][0],
            builder.curves["equalizer_inner_wall"][0] if case_id == "A" else builder.curves["port_wall_cavity"][0],
            builder.curves["equalizer_inner_upper"][0] if case_id == "D" else 0,
            -builder.curves["equalizer_top"][0],
            -builder.curves["equalizer_outer_wall"][0],
        ]
        body_loop = [x for x in body_loop if x != 0]
        builder.surface("rigid_equalizer_surface", body_loop, "rigid_comparison_equalizer")

    elif case_id == "E":
        sd = float(geometry["pr_area_m2"])
        pr_radius = math.sqrt(sd / math.pi)
        pr_clearance = float(geometry["pr_rear_clearance_m"])
        z_pr_face = z_inner_rear + pr_clearance
        p_pr_wall_inner = _point(builder, "pr_cavity_wall_start", pr_radius, z_inner_rear, lc_feature)
        p_pr_wall_top = _point(builder, "pr_cavity_wall_top", pr_radius, z_pr_face, lc_feature)
        p_pr_face_axis = _point(builder, "pr_cavity_face_axis", 0.0, z_pr_face, lc_feature)
        builder.line("cavity_side_inner", p_c_front_outer, p_c_side_split, "cabinet_side_wall")
        builder.line("cavity_rear_wall", p_c_side_split, p_pr_wall_inner, "cabinet_rear_wall")
        builder.line("pr_side", p_pr_wall_inner, p_pr_wall_top, "pr_side_wall")
        builder.line("pr_face_cavity", p_pr_wall_top, p_pr_face_axis, "pr_cavity_face")
        cavity_loop = [
            builder.curves["reference_planar_piston_back"][0],
            builder.curves["driver_side"][0],
            builder.curves["cavity_baffle"][0],
            builder.curves["cavity_side_inner"][0],
            builder.curves["cavity_rear_wall"][0],
            builder.curves["pr_side"][0],
            builder.curves["pr_face_cavity"][0],
            builder.line("cavity_axis", p_pr_face_axis, p_db_axis, "axis"),
        ]
        builder.surface("cavity_surface", cavity_loop, "air_cavity")

        z_outer_rear = -depth - float(geometry["wall_thickness_m"])
        p_pr_outer_axis = _point(builder, "pr_exterior_face_axis", 0.0, z_outer_rear, lc_feature)
        p_pr_outer = _point(builder, "pr_exterior_face_radius", pr_radius, z_outer_rear, lc_feature)
        builder.line("pr_body_lower_side", p_pr_outer, p_pr_wall_inner, "cabinet_rear_wall")
        builder.line("pr_side", p_pr_wall_inner, p_pr_wall_top, "pr_side_wall")
        builder.line("pr_face_cavity", p_pr_wall_top, p_pr_face_axis, "pr_cavity_face")
        builder.line("pr_body_axis", p_pr_outer_axis, p_pr_face_axis, "axis")
        builder.line("pr_face_exterior", p_pr_outer_axis, p_pr_outer, "pr_exterior_face")
        builder.surface(
            "rigid_pr_surface",
            [
                builder.curves["pr_body_axis"][0],
                -builder.curves["pr_face_cavity"][0],
                -builder.curves["pr_side"][0],
                -builder.curves["pr_body_lower_side"][0],
                -builder.curves["pr_face_exterior"][0],
            ],
            "rigid_pr_back_mechanism",
        )

    else:  # pragma: no cover - guarded by case mapping
        raise ValueError(f"unsupported reference case {case_id}")

    return {
        "driver_occupancy_length_m": driver_length,
        "driver_back_z_m": z_driver_back,
        "equalizer_depth_m": equalizer_depth,
        "equalizer_volume_m3": equalizer_volume,
        "reference_planar_piston_radius_m": rd,
        "pr_radius_m": math.sqrt(float(geometry["pr_area_m2"]) / math.pi) if case_id == "E" else 0.0,
    }


def _build_rear_connector(
    builder: _GeoBuilder,
    geometry: Mapping[str, float],
    case_id: str,
    outer: Mapping[str, float | int | None],
    lc_feature: float,
) -> None:
    """Build the through-wall opening, the D port, or the E rear exterior face."""

    z_outer = float(outer["z_rear_outer"])
    p_rc = int(outer["p_rear_circle"])
    p_rear_side = int(outer["p_rear_side"])
    p_axis_rear = int(outer["p_axis_rear"])
    p_ib = int(outer["p_inner_bottom"])
    p_axis_bottom = int(outer["p_axis_bottom"])
    z_inner = -float(geometry["inner_depth_m"])
    ri = float(geometry["inner_radius_m"])
    ro = float(geometry["outer_radius_m"])

    if case_id == "A":
        a = float(geometry["rear_opening_radius_m"])
        p_i0 = builder.points["rear_feature_inner_axis"][0]
        p_i1 = builder.points["equalizer_inner_bottom"][0]
        p_o0 = p_axis_rear
        p_o1 = _point(builder, "rear_opening_radius_outer", a, z_outer, lc_feature)
        builder.line("rear_opening_inner", p_i0, p_i1, "rear_opening_cavity")
        builder.line("rear_opening_exterior_line", p_o0, p_o1, "rear_opening_exterior")
        builder.line("rear_opening_wall_line", p_o1, p_i1, "rear_opening_wall")
        builder.line("rear_opening_axis", p_o0, p_i0, "axis")
        builder.surface(
            "rear_opening_surface",
            [
                builder.curves["rear_opening_axis"][0],
                builder.curves["rear_opening_inner"][0],
                -builder.curves["rear_opening_wall_line"][0],
                -builder.curves["rear_opening_exterior_line"][0],
            ],
            "air_rear_opening",
        )
        builder.line("rear_exterior_axis", p_axis_bottom, p_axis_rear, "axis")
        builder.line("rear_exterior_wall", p_o1, p_rear_side, "cabinet_rear_wall")
        builder.line("free_rear_side", p_rear_side, p_rc, "freefield_rear_side_interface")
        builder.surface(
            "rear_free_surface",
            [
                builder.curves["rear_exterior_axis"][0],
                builder.curves["rear_opening_exterior_line"][0],
                builder.curves["rear_exterior_wall"][0],
                builder.curves["free_rear_side"][0],
                builder.curves["hk_rear_cap"][0],
            ],
            "air_rear_free",
        )
        return

    if case_id == "D":
        a = float(geometry["port_radius_m"])
        penetration = float(geometry["port_penetration_m"])
        port_length = float(geometry["port_length_m"])
        z_port_in = z_inner + penetration
        z_port_out = z_inner - port_length
        if z_port_out <= -float(outer["h"]):
            raise ValueError("port end reaches beyond the configured HK radius")
        p_i0 = builder.points["rear_feature_inner_axis"][0]
        p_i1 = builder.points["port_inner_end"][0]
        p_wall_inner = builder.points["equalizer_inner_bottom"][0]
        p_wall_outer = _point(builder, "port_wall_outer", a, z_outer, lc_feature)
        p_o1 = _point(builder, "port_outer_radius", a, z_port_out, lc_feature)
        p_o0 = _point(builder, "port_axis_outer", 0.0, z_port_out, lc_feature)
        # The upper section is shared with the comparison equalizer; the lower
        # section is the physical port wall through the cabinet and exterior.
        builder.line("port_wall_through", p_wall_outer, p_wall_inner, "port_wall")
        builder.line("port_wall_external", p_o1, p_wall_outer, "port_wall")
        builder.line("port_exterior_line", p_o0, p_o1, "port_exterior_opening")
        builder.line("port_axis", p_o0, p_i0, "axis")
        builder.surface(
            "port_surface",
            [
                builder.curves["port_axis"][0],
                builder.curves["port_cavity_top"][0],
                -builder.curves["port_wall_cavity"][0],
                -builder.curves["port_wall_through"][0],
                -builder.curves["port_wall_external"][0],
                -builder.curves["port_exterior_line"][0],
            ],
            "air_port",
        )

        p_tip_axis = p_o0
        p_tip_radius = int(outer["p_port_circle"])
        builder.line("rear_shell_wall_top", p_wall_outer, p_rear_side, "cabinet_rear_wall")
        builder.line("free_rear_side", p_rear_side, p_rc, "freefield_rear_side_interface")
        builder.line("rear_shell_tip_interface", p_tip_radius, p_o1, "rear_tip_shell_interface")
        builder.surface(
            "rear_port_shell_surface",
            [
                builder.curves["rear_shell_wall_top"][0],
                builder.curves["free_rear_side"][0],
                builder.curves["hk_rear_shell"][0],
                builder.curves["rear_shell_tip_interface"][0],
                builder.curves["port_wall_external"][0],
            ],
            "air_rear_free",
        )
        builder.line("rear_tip_axis", p_axis_bottom, p_tip_axis, "axis")
        builder.surface(
            "rear_port_tip_surface",
            [
                builder.curves["rear_tip_axis"][0],
                builder.curves["port_exterior_line"][0],
                -builder.curves["rear_shell_tip_interface"][0],
                builder.curves["hk_rear_tip"][0],
            ],
            "air_rear_free",
        )
        return

    # B/C have a closed rear cabinet wall.  E has a PR exterior face at the
    # outer rear plane; the PR rigid body itself occupies the opening.
    if case_id == "E":
        pr_radius = math.sqrt(float(geometry["pr_area_m2"]) / math.pi)
        p_pr_outer_axis = _point(builder, "pr_exterior_face_axis", 0.0, z_outer, lc_feature)
        p_pr_outer = _point(builder, "pr_exterior_face_radius", pr_radius, z_outer, lc_feature)
        p_wall = p_rear_side
        builder.line("pr_face_exterior", p_pr_outer_axis, p_pr_outer, "pr_exterior_face")
        builder.line("rear_pr_wall", p_pr_outer, p_wall, "cabinet_rear_wall")
        builder.line("free_rear_side", p_wall, p_rc, "freefield_rear_side_interface")
        builder.line("rear_pr_axis", p_axis_bottom, p_pr_outer_axis, "axis")
        builder.surface(
            "rear_pr_free_surface",
            [
                builder.curves["rear_pr_axis"][0],
                builder.curves["pr_face_exterior"][0],
                builder.curves["rear_pr_wall"][0],
                builder.curves["free_rear_side"][0],
                builder.curves["hk_rear_cap"][0],
            ],
            "air_rear_free",
        )
        return

    p_wall = p_rear_side
    builder.line("rear_closed_axis", p_axis_bottom, p_axis_rear, "axis")
    builder.line("rear_closed_wall", p_axis_rear, p_wall, "cabinet_rear_wall")
    builder.line("free_rear_side", p_wall, p_rc, "freefield_rear_side_interface")
    builder.surface(
        "rear_closed_free_surface",
        [
            builder.curves["rear_closed_axis"][0],
            builder.curves["rear_closed_wall"][0],
            builder.curves["free_rear_side"][0],
            builder.curves["hk_rear_cap"][0],
        ],
        "air_rear_free",
    )


def _validated_geometry_values(config: EnclosureConfig, case_id: str) -> dict[str, float]:
    raw = config.raw
    g = raw["geometry"]
    volume = raw["volume_contract"]
    port = raw["port"]
    pr = raw["passive_radiator"]
    return {
        "inner_radius_m": float(g["inner_radius_m"]),
        "inner_depth_m": float(g["inner_depth_m"]),
        "wall_thickness_m": float(g["wall_thickness_m"]),
        "outer_radius_m": float(g["outer_radius_m"]),
        "outer_depth_m": float(g["outer_depth_m"]),
        "driver_radius_m": float(g["driver_radius_m"]),
        "rear_opening_radius_m": float(g["rear_opening_radius_m"] or 0.0),
        "pml_inner_radius_m": float(g["pml_inner_radius_m"]),
        "pml_thickness_m": float(g["pml_thickness_m"]),
        "port_radius_m": float(port["radius_m"]),
        "port_length_m": float(port["length_m"]),
        "port_penetration_m": float(g["port_penetration_into_box_m"]),
        "pr_area_m2": float(pr["Sd_m2"]),
        "pr_rear_clearance_m": float(pr["rear_clearance_m"]),
        "driver_displacement_m3": float(volume["driver_displacement_m3"]),
        "reserved_rear_feature_m3": float(volume["reserved_rear_feature_m3"]),
        "fair_comparison_equalization_m3": float(volume["fair_comparison_equalization_m3"]),
        "net_volume_target_m3": float(raw["net_volume_target_m3"]),
    }


def generate_reference_mesh(
    config_path: str | Path,
    level: str,
    msh_path: str | Path,
    *,
    terminal: bool = False,
) -> MeshGenerationResult:
    """Generate one deterministic A--E reference mesh at L0/L1/L2.

    This function only creates a Gmsh mesh and physical labels.  In particular,
    it never calls a Helmholtz/FEM assembly or a frequency-domain solve.
    """

    config_path = Path(config_path)
    msh_path = Path(msh_path)
    config = load_enclosure_config(config_path)
    case_id = case_id_for_config(config.case)
    level = str(level).upper()
    if level not in {"L0", "L1", "L2"}:
        raise ValueError("level must be L0, L1, or L2")
    geometry_key = geometry_key_for_case(case_id)
    values = _validated_geometry_values(config, case_id)
    mesh_cfg = config.raw["mesh"]
    h = float(mesh_cfg[f"global_size_{level}_m"])
    h_port = float(mesh_cfg[f"port_local_size_{level}_m"])
    lc_feature = min(h * 0.5, h_port)
    msh_path.parent.mkdir(parents=True, exist_ok=True)

    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if terminal else 0)
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.option.setNumber("Mesh.RandomFactor", 0.0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min(lc_feature, h) * 0.35)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        # B and C intentionally use the exact same model name and geometry key.
        gmsh.model.add(f"luna_enclosure_{geometry_key}")
        builder = _GeoBuilder(gmsh)
        outer = _build_outer_field(builder, values, case_id, h)
        _build_driver_and_cavity(builder, values, config.raw["volume_contract"], case_id, lc_feature)
        _build_rear_connector(builder, values, case_id, outer, lc_feature)
        gmsh.model.geo.synchronize()

        for (dimension, name), entities in sorted(builder.groups.items()):
            if not entities:
                continue
            tag = PHYSICAL_TAGS[name]
            gmsh.model.addPhysicalGroup(dimension, sorted(entities), tag)
            gmsh.model.setPhysicalName(dimension, tag, name)

        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(int(mesh_cfg["element_order"]))
        gmsh.write(str(msh_path))
    finally:
        gmsh.finalize()

    group_info: dict[str, dict[str, int | str]] = {}
    for (dimension, name), entities in sorted(builder.groups.items()):
        group_info[name] = {
            "tag": int(PHYSICAL_TAGS[name]),
            "dimension": int(dimension),
            "entity_count": int(len(entities)),
        }
    geometry_report = _validated_geometry_values(config, case_id)
    geometry_report.update(
        {
            "case_id": case_id,
            "geometry_key": geometry_key,
            "level": level,
            "global_size_m": h,
            "feature_size_m": lc_feature,
            "reference_planar_piston_radius_m": float(values["driver_radius_m"]),
            "pml_inner_radius_m": float(values["pml_inner_radius_m"]),
            "pml_outer_radius_m": float(values["pml_inner_radius_m"] + values["pml_thickness_m"]),
            "mirror": False,
        }
    )
    return MeshGenerationResult(
        path=msh_path,
        case_id=case_id,
        level=level,
        config_path=config_path,
        geometry_key=geometry_key,
        physical_groups=group_info,
        geometry=geometry_report,
    )


# Descriptive aliases for callers and phase-2 scripts.
generate_enclosure_mesh = generate_reference_mesh
