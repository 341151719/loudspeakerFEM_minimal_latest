"""Low-level passive thermoviscous boundary-layer reference utilities.

This module deliberately does not assemble or solve an enclosure model.  It
implements the phase-4A material, applicability, weak-coefficient, power, and
circular-duct reference contracts for the ``exp(+i*omega*t)`` convention.

For a fixed, isothermal, no-slip wall the BLI correction added to the project's
pressure weak residual is

    integral_Gamma 2*pi*r * (
        c_v * grad_t(p).grad_t(q) + c_t * p*q
    ) ds,

where ``c_v = -delta_v/(rho*(1+i))`` and
``c_t = -omega**2*(gamma-1)*delta_t/(K_s*(1+i))``.  Both coefficients have
positive imaginary part.  Consequently ``Im(p^H A_BLI p)/(2*omega)`` is the
nonnegative time-average wall loss for peak phasors.  The minus signs are not
empirical: they result when the official inward-velocity condition is put on
the left of the pressure weak equation and its surface Laplacian is integrated
by parts.  The axisymmetric ``2*pi*r`` factor belongs in the boundary matrices,
not in these scalar coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.special import jve

from .enclosure_schema import EnclosureConfig, load_enclosure_config


HARMONIC_CONVENTION = "exp(+i*omega*t)"
REFERENCE_MODEL = "fixed_isothermal_no_slip_BLI"
ALLOWED_ROUTES = frozenset(("BLI", "reject", "NRA", "full-TV"))


def _positive(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


@dataclass(frozen=True)
class ThermoviscousAirProperties:
    """SI air properties required by the BLI and circular-duct references."""

    rho0_kg_m3: float
    c0_m_s: float
    temperature_K: float
    pressure_Pa: float
    dynamic_viscosity_Pa_s: float
    heat_capacity_cp_J_kgK: float
    thermal_conductivity_W_mK: float
    gamma: float
    prandtl: float

    def __post_init__(self) -> None:
        for field in (
            "rho0_kg_m3",
            "c0_m_s",
            "temperature_K",
            "pressure_Pa",
            "dynamic_viscosity_Pa_s",
            "heat_capacity_cp_J_kgK",
            "thermal_conductivity_W_mK",
            "prandtl",
        ):
            _positive(field, getattr(self, field))
        if not math.isfinite(float(self.gamma)) or float(self.gamma) <= 1.0:
            raise ValueError("gamma must be finite and greater than one")

    @property
    def bulk_modulus_isentropic_Pa(self) -> float:
        return self.rho0_kg_m3 * self.c0_m_s * self.c0_m_s

    @classmethod
    def from_config(
        cls, config: EnclosureConfig | str | Path
    ) -> "ThermoviscousAirProperties":
        validated = (
            config
            if isinstance(config, EnclosureConfig)
            else load_enclosure_config(config)
        )
        air = validated.raw["air"]
        return cls(
            rho0_kg_m3=air["rho0_kg_m3"],
            c0_m_s=air["c0_m_s"],
            temperature_K=air["temperature_K"],
            pressure_Pa=air["pressure_Pa"],
            dynamic_viscosity_Pa_s=air["dynamic_viscosity_Pa_s"],
            heat_capacity_cp_J_kgK=air["heat_capacity_cp_J_kgK"],
            thermal_conductivity_W_mK=air["thermal_conductivity_W_mK"],
            gamma=air["gamma"],
            prandtl=air["prandtl"],
        )

    def with_changes(self, **changes: float) -> "ThermoviscousAirProperties":
        """Return a validated changed copy, useful for sensitivity references."""

        return replace(self, **changes)

    def to_dict(self) -> dict[str, float]:
        return {
            key: float(getattr(self, key))
            for key in self.__dataclass_fields__
        } | {"bulk_modulus_isentropic_Pa": self.bulk_modulus_isentropic_Pa}


@dataclass(frozen=True)
class BoundaryLayerThicknesses:
    frequency_Hz: float
    omega_rad_s: float
    delta_v_m: float
    delta_t_m: float
    loss_scale: float

    def to_dict(self) -> dict[str, float]:
        return {
            key: float(getattr(self, key))
            for key in self.__dataclass_fields__
        }


def boundary_layer_thicknesses(
    air: ThermoviscousAirProperties,
    frequency_Hz: float,
    *,
    loss_scale: float = 1.0,
) -> BoundaryLayerThicknesses:
    """Return ``delta_v`` and ``delta_t`` in metres.

    ``loss_scale`` scales the integrated boundary layer and exists only for
    deterministic zero-loss and continuity checks.  It must be nonnegative.
    """

    frequency = _positive("frequency_Hz", frequency_Hz)
    scale = float(loss_scale)
    if not math.isfinite(scale) or scale < 0.0:
        raise ValueError("loss_scale must be finite and nonnegative")
    omega = 2.0 * math.pi * frequency
    delta_v = math.sqrt(
        2.0 * air.dynamic_viscosity_Pa_s / (air.rho0_kg_m3 * omega)
    )
    delta_t = math.sqrt(
        2.0
        * air.thermal_conductivity_W_mK
        / (air.rho0_kg_m3 * air.heat_capacity_cp_J_kgK * omega)
    )
    return BoundaryLayerThicknesses(
        frequency_Hz=frequency,
        omega_rad_s=omega,
        delta_v_m=scale * delta_v,
        delta_t_m=scale * delta_t,
        loss_scale=scale,
    )


@dataclass(frozen=True)
class BLIBilinearCoefficients:
    """Scalar coefficients for axisymmetric boundary mass/stiffness matrices."""

    frequency_Hz: float
    omega_rad_s: float
    delta_v_m: float
    delta_t_m: float
    viscous_tangential_gradient_m4_kg: complex
    thermal_pressure_m2_kg: complex
    loss_scale: float

    @property
    def passive(self) -> bool:
        return bool(
            self.viscous_tangential_gradient_m4_kg.imag >= 0.0
            and self.thermal_pressure_m2_kg.imag >= 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        def encoded(value: complex) -> dict[str, float]:
            return {"real": float(value.real), "imag": float(value.imag)}

        return {
            "frequency_Hz": self.frequency_Hz,
            "omega_rad_s": self.omega_rad_s,
            "delta_v_m": self.delta_v_m,
            "delta_t_m": self.delta_t_m,
            "loss_scale": self.loss_scale,
            "viscous_tangential_gradient_m4_kg": encoded(
                self.viscous_tangential_gradient_m4_kg
            ),
            "thermal_pressure_m2_kg": encoded(self.thermal_pressure_m2_kg),
            "passive_exp_plus_iwt": self.passive,
        }


def bli_bilinear_coefficients(
    air: ThermoviscousAirProperties,
    frequency_Hz: float,
    *,
    loss_scale: float = 1.0,
) -> BLIBilinearCoefficients:
    """Return fixed-wall BLI weak coefficients for ``exp(+i*omega*t)``.

    The boundary stiffness and mass matrices must include ``2*pi*r``.  The
    units shown in the returned field names make each assembled residual term
    compatible with the bulk pressure weak equation.
    """

    layers = boundary_layer_thicknesses(
        air, frequency_Hz, loss_scale=loss_scale
    )
    divisor = 1.0 + 1.0j
    viscous = -layers.delta_v_m / (air.rho0_kg_m3 * divisor)
    thermal = -(
        layers.omega_rad_s**2
        * (air.gamma - 1.0)
        * layers.delta_t_m
        / (air.bulk_modulus_isentropic_Pa * divisor)
    )
    return BLIBilinearCoefficients(
        frequency_Hz=layers.frequency_Hz,
        omega_rad_s=layers.omega_rad_s,
        delta_v_m=layers.delta_v_m,
        delta_t_m=layers.delta_t_m,
        viscous_tangential_gradient_m4_kg=complex(viscous),
        thermal_pressure_m2_kg=complex(thermal),
        loss_scale=layers.loss_scale,
    )


@dataclass(frozen=True)
class BLIDissipation:
    frequency_Hz: float
    viscous_quadratic: float
    thermal_quadratic: float
    P_visc_W: float
    P_thermal_W: float
    P_total_W: float
    passive: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            key: (bool(value) if isinstance(value, bool) else float(value))
            for key, value in self.__dict__.items()
        }


def _real_quadratic(vector: np.ndarray, matrix: Any, name: str) -> float:
    values = np.asarray(vector, dtype=np.complex128).reshape(-1)
    if hasattr(matrix, "getH"):
        skew = matrix - matrix.getH()
        skew_max = float(np.max(np.abs(skew.data))) if skew.nnz else 0.0
        matrix_max = float(np.max(np.abs(matrix.data))) if matrix.nnz else 0.0
    else:
        dense = np.asarray(matrix)
        skew_max = float(np.max(np.abs(dense - dense.conj().T)))
        matrix_max = float(np.max(np.abs(dense))) if dense.size else 0.0
    hermitian_tolerance = 1.0e-12 * max(1.0, matrix_max)
    if skew_max > hermitian_tolerance:
        raise ValueError(f"{name} must be Hermitian for a physical quadratic form")
    product = matrix @ values
    result = complex(np.vdot(values, product))
    roundoff_scale = float(np.linalg.norm(values) * np.linalg.norm(product))
    tolerance = 1.0e-12 * max(1.0, roundoff_scale)
    if abs(result.imag) > tolerance:
        raise ValueError(f"{name} must be Hermitian for a physical quadratic form")
    positivity_tolerance = 1.0e-12 * max(1.0, roundoff_scale, abs(result.real))
    if result.real < -positivity_tolerance:
        raise ValueError(f"{name} quadratic form must be nonnegative")
    return float(max(result.real, 0.0))


def bli_dissipation(
    pressure: np.ndarray,
    tangential_gradient_matrix: Any,
    boundary_mass_matrix: Any,
    coefficients: BLIBilinearCoefficients,
) -> BLIDissipation:
    """Evaluate independent peak-phasor BLI loss quadratics.

    Both supplied matrices must already represent the axisymmetric integrals
    with ``2*pi*r``.  No clipping of a coefficient's loss sign is performed.
    """

    viscous_q = _real_quadratic(
        pressure, tangential_gradient_matrix, "tangential_gradient_matrix"
    )
    thermal_q = _real_quadratic(
        pressure, boundary_mass_matrix, "boundary_mass_matrix"
    )
    omega = coefficients.omega_rad_s
    p_visc = (
        coefficients.viscous_tangential_gradient_m4_kg.imag
        * viscous_q
        / (2.0 * omega)
    )
    p_thermal = (
        coefficients.thermal_pressure_m2_kg.imag
        * thermal_q
        / (2.0 * omega)
    )
    tolerance = 1.0e-14 * max(1.0, abs(p_visc) + abs(p_thermal))
    passive = p_visc >= -tolerance and p_thermal >= -tolerance
    if not passive:
        raise ValueError("BLI coefficient sign violates exp(+i*omega*t) passivity")
    return BLIDissipation(
        frequency_Hz=coefficients.frequency_Hz,
        viscous_quadratic=viscous_q,
        thermal_quadratic=thermal_q,
        P_visc_W=float(p_visc),
        P_thermal_W=float(p_thermal),
        P_total_W=float(p_visc + p_thermal),
        passive=passive,
    )


def _validated_config(config: EnclosureConfig | str | Path) -> EnclosureConfig:
    return config if isinstance(config, EnclosureConfig) else load_enclosure_config(config)


def assess_bli_applicability(
    config: EnclosureConfig | str | Path,
    frequencies_Hz: Iterable[float],
    *,
    local_curvature_radii_m: Mapping[str, float] | None = None,
    cross_section_kind: str = "general_enclosure",
) -> dict[str, Any]:
    """Assess the declared BLI gap/curvature/non-overlap contract.

    ``NRA`` is returned only for an explicitly identified constant parallel
    slit.  Circular ducts and general enclosures route to ``full-TV`` when
    their layers overlap; this function never calls the repository's slit NRA.
    """

    validated = _validated_config(config)
    tv = validated.raw["thermoviscous"]
    declaration = tv["applicability"]
    air = ThermoviscousAirProperties.from_config(validated)
    minimum_gap = _positive("minimum_gap_m", declaration["minimum_gap_m"])
    frequency_min = _positive("frequency_min_Hz", declaration["frequency_min_Hz"])
    frequency_max = _positive("frequency_max_Hz", declaration["frequency_max_Hz"])
    if frequency_max < frequency_min:
        raise ValueError("BLI applicability frequency interval is reversed")
    gap_limit = _positive(
        "boundary_layer_to_gap_max", declaration["boundary_layer_to_gap_max"]
    )
    radius_limit = _positive(
        "boundary_layer_to_radius_max",
        declaration["boundary_layer_to_radius_max"],
    )

    geometry = validated.raw["geometry"]
    radii = dict(
        local_curvature_radii_m
        or {
            "cabinet_inner_cylindrical_wall": geometry["inner_radius_m"],
            "reference_piston_edge_cylindrical_wall": geometry["driver_radius_m"],
        }
    )
    if not radii:
        raise ValueError("at least one local curvature radius is required")
    normalized_radii = {name: _positive(name, value) for name, value in radii.items()}
    conservative_radius = min(normalized_radii.values())
    frequency_values = tuple(float(value) for value in frequencies_Hz)
    if not frequency_values:
        raise ValueError("at least one frequency is required")

    rows: list[dict[str, Any]] = []
    for frequency in frequency_values:
        layers = boundary_layer_thicknesses(air, frequency)
        maximum_delta = max(layers.delta_v_m, layers.delta_t_m)
        in_band = frequency_min <= frequency <= frequency_max
        delta_gap = maximum_delta / minimum_gap
        two_delta_gap = 2.0 * maximum_delta / minimum_gap
        delta_radius = maximum_delta / conservative_radius
        overlap = two_delta_gap >= 1.0
        gap_ok = delta_gap <= gap_limit
        curvature_ok = delta_radius <= radius_limit
        if not bool(tv["enabled"]) or not in_band:
            route = "reject"
        elif overlap:
            route = (
                "NRA"
                if cross_section_kind == "constant_parallel_slit"
                else "full-TV"
            )
        elif not gap_ok or not curvature_ok:
            route = "full-TV"
        else:
            route = "BLI"
        rows.append(
            {
                **layers.to_dict(),
                "in_declared_frequency_band": in_band,
                "minimum_gap_m": minimum_gap,
                "conservative_curvature_radius_m": conservative_radius,
                "max_delta_over_gap": delta_gap,
                "two_max_delta_over_gap": two_delta_gap,
                "max_delta_over_radius": delta_radius,
                "boundary_layers_overlap": overlap,
                "gap_ratio_pass": gap_ok,
                "curvature_ratio_pass": curvature_ok,
                "route": route,
            }
        )

    overall_route = "BLI"
    if any(row["route"] == "reject" for row in rows):
        overall_route = "reject"
    elif any(row["route"] == "full-TV" for row in rows):
        overall_route = "full-TV"
    elif any(row["route"] == "NRA" for row in rows):
        overall_route = "NRA"
    assert overall_route in ALLOWED_ROUTES
    return {
        "status": "pass" if overall_route == "BLI" else "fail",
        "case": validated.case,
        "demonstrator": validated.demonstrator,
        "model": REFERENCE_MODEL,
        "harmonic_convention": HARMONIC_CONVENTION,
        "cross_section_kind": cross_section_kind,
        "frequency_band_Hz": [frequency_min, frequency_max],
        "minimum_gap_m": minimum_gap,
        "local_curvature_radii_m": normalized_radii,
        "conservative_curvature_radius_m": conservative_radius,
        "limits": {
            "boundary_layer_to_gap_max": gap_limit,
            "boundary_layer_to_radius_max": radius_limit,
            "boundary_layers_must_not_overlap": bool(
                declaration["boundary_layers_must_not_overlap"]
            ),
        },
        "sharp_corner_note": (
            "BLI applies on smooth wall segments; unresolved corner neighborhoods "
            "remain a local full-TV risk and are not represented as zero curvature."
        ),
        "frequencies": rows,
        "route": overall_route,
        "nra_guard": (
            "NRA routing is allowed only for constant_parallel_slit; circular "
            "duct references use their own Bessel solution."
        ),
    }


@dataclass(frozen=True)
class CircularDuctReference:
    frequency_Hz: float
    radius_m: float
    wavenumber_1_m: complex
    wavenumber_ratio: complex
    characteristic_impedance_Pa_s_m: complex
    attenuation_Np_m: float
    model: str
    loss_scale: float

    def to_dict(self) -> dict[str, Any]:
        def encoded(value: complex) -> dict[str, float]:
            return {"real": float(value.real), "imag": float(value.imag)}

        return {
            "frequency_Hz": self.frequency_Hz,
            "radius_m": self.radius_m,
            "wavenumber_1_m": encoded(self.wavenumber_1_m),
            "wavenumber_ratio": encoded(self.wavenumber_ratio),
            "characteristic_impedance_Pa_s_m": encoded(
                self.characteristic_impedance_Pa_s_m
            ),
            "attenuation_Np_m": self.attenuation_Np_m,
            "model": self.model,
            "loss_scale": self.loss_scale,
            "harmonic_convention": HARMONIC_CONVENTION,
        }


def circular_duct_wide_asymptotic(
    air: ThermoviscousAirProperties,
    frequency_Hz: float,
    radius_m: float,
    *,
    loss_scale: float = 1.0,
) -> CircularDuctReference:
    """Kirchhoff/BLI wide-circular-duct first-order reference."""

    radius = _positive("radius_m", radius_m)
    layers = boundary_layer_thicknesses(
        air, frequency_Hz, loss_scale=loss_scale
    )
    correction = (
        (1.0 - 1.0j)
        * (layers.delta_v_m + (air.gamma - 1.0) * layers.delta_t_m)
        / (2.0 * radius)
    )
    ratio = 1.0 + correction
    k0 = layers.omega_rad_s / air.c0_m_s
    wavenumber = k0 * ratio
    # First-order impedance correction follows the density/compliance split.
    impedance_ratio = 1.0 + (
        (1.0 - 1.0j)
        * (layers.delta_v_m - (air.gamma - 1.0) * layers.delta_t_m)
        / (2.0 * radius)
    )
    impedance = air.rho0_kg_m3 * air.c0_m_s * impedance_ratio
    return CircularDuctReference(
        frequency_Hz=layers.frequency_Hz,
        radius_m=radius,
        wavenumber_1_m=complex(wavenumber),
        wavenumber_ratio=complex(ratio),
        characteristic_impedance_Pa_s_m=complex(impedance),
        attenuation_Np_m=float(-wavenumber.imag),
        model="Kirchhoff_BLI_wide_circular_duct",
        loss_scale=layers.loss_scale,
    )


def _circular_field_function(argument: complex) -> complex:
    denominator = jve(0, argument)
    if abs(denominator) == 0.0:
        raise ValueError("circular duct Bessel denominator is zero")
    # COMSOL 6.1 circular LRF field function: Upsilon_j=-J2(k_j*a)/J0(k_j*a).
    return complex(-jve(2, argument) / denominator)


def circular_duct_zwikker_kosten(
    air: ThermoviscousAirProperties,
    frequency_Hz: float,
    radius_m: float,
    *,
    loss_scale: float = 1.0,
) -> CircularDuctReference:
    """Non-asymptotic circular LRF (Zwikker--Kosten/Stinson) reference.

    This is a circular Bessel solution and does not call the parallel-slit NRA
    implementation.  For ``exp(+i*omega*t)``, outgoing waves use
    ``exp(-i*k*z)`` and passive attenuation therefore has ``Im(k) <= 0``.
    """

    radius = _positive("radius_m", radius_m)
    layers = boundary_layer_thicknesses(
        air, frequency_Hz, loss_scale=loss_scale
    )
    k0 = layers.omega_rad_s / air.c0_m_s
    if layers.loss_scale == 0.0:
        ratio = 1.0 + 0.0j
        impedance = complex(air.rho0_kg_m3 * air.c0_m_s)
    else:
        zv = (1.0 - 1.0j) * radius / layers.delta_v_m
        zt = (1.0 - 1.0j) * radius / layers.delta_t_m
        upsilon_v = _circular_field_function(zv)
        upsilon_t = _circular_field_function(zt)
        compliance_ratio = air.gamma - (air.gamma - 1.0) * upsilon_t
        ratio_squared = compliance_ratio / upsilon_v
        ratio = complex(np.sqrt(ratio_squared))
        if ratio.real < 0.0:
            ratio = -ratio
        if ratio.imag > 1.0e-12:
            raise ValueError("circular duct branch violates exp(+i*omega*t) passivity")
        impedance_ratio = complex(np.sqrt(1.0 / (upsilon_v * compliance_ratio)))
        if impedance_ratio.real < 0.0:
            impedance_ratio = -impedance_ratio
        impedance = air.rho0_kg_m3 * air.c0_m_s * impedance_ratio
    wavenumber = k0 * ratio
    attenuation = float(-wavenumber.imag)
    if attenuation < -1.0e-12:
        raise ValueError("circular duct attenuation is negative")
    return CircularDuctReference(
        frequency_Hz=layers.frequency_Hz,
        radius_m=radius,
        wavenumber_1_m=complex(wavenumber),
        wavenumber_ratio=complex(ratio),
        characteristic_impedance_Pa_s_m=complex(impedance),
        attenuation_Np_m=max(attenuation, 0.0),
        model="Zwikker_Kosten_Stinson_circular_Bessel_LRF",
        loss_scale=layers.loss_scale,
    )


__all__ = [
    "ALLOWED_ROUTES",
    "BLIBilinearCoefficients",
    "BLIDissipation",
    "BoundaryLayerThicknesses",
    "CircularDuctReference",
    "HARMONIC_CONVENTION",
    "REFERENCE_MODEL",
    "ThermoviscousAirProperties",
    "assess_bli_applicability",
    "bli_bilinear_coefficients",
    "bli_dissipation",
    "boundary_layer_thicknesses",
    "circular_duct_wide_asymptotic",
    "circular_duct_zwikker_kosten",
]
