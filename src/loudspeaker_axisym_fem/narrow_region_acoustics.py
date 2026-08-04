from __future__ import annotations

from dataclasses import dataclass, asdict
import cmath
import math
from typing import Any


@dataclass(frozen=True)
class NarrowSlit:
    domain_id: int
    height_m: float
    label: str


COMSOL_NARROW_REGIONS = {
    "nra1": NarrowSlit(domain_id=8, height_m=0.4e-3, label="magnetic gap slit 0.4 mm"),
    "nra2": NarrowSlit(domain_id=22, height_m=0.2e-3, label="magnetic gap slit 0.2 mm"),
}


@dataclass(frozen=True)
class AirThermophysicalProperties:
    temperature_K: float
    pressure_Pa: float
    molar_mass_kg_mol: float
    gamma: float
    density_kg_m3: float
    dynamic_viscosity_Pa_s: float
    heat_capacity_cp_J_kgK: float
    thermal_conductivity_W_mK: float
    sound_speed_m_s: float
    prandtl: float
    bulk_modulus_Pa: float

    def to_jsonable(self) -> dict[str, float]:
        return asdict(self)


def comsol_air_properties(
    temperature_K: float = 293.15,
    pressure_Pa: float = 101325.0,
    *,
    molar_mass_kg_mol: float = 0.02897,
    gamma: float = 1.4,
) -> AirThermophysicalProperties:
    """Air properties copied from the COMSOL model material polynomials.

    The source model defines eta(T), Cp(T), k(T), rho(pA,T), and cs(T) in the
    Air material.  Evaluating the same functions removes the former hard-coded
    approximate Prandtl number and keeps the native NRA coefficients traceable.
    """
    T = float(temperature_K)
    p = float(pressure_Pa)
    R = 8.31446261815324
    eta = (
        -8.38278e-7
        + 8.35717342e-8 * T
        - 7.69429583e-11 * T**2
        + 4.6437266e-14 * T**3
        - 1.06585607e-17 * T**4
    )
    cp = (
        1047.63657
        - 0.372589265 * T
        + 9.45304214e-4 * T**2
        - 6.02409443e-7 * T**3
        + 1.2858961e-10 * T**4
    )
    kappa = (
        -0.00227583562
        + 1.15480022e-4 * T
        - 7.90252856e-8 * T**2
        + 4.11702505e-11 * T**3
        - 7.43864331e-15 * T**4
    )
    rho = p * molar_mass_kg_mol / (R * T)
    c0 = math.sqrt(gamma * R * T / molar_mass_kg_mol)
    pr = eta * cp / kappa
    return AirThermophysicalProperties(
        temperature_K=T,
        pressure_Pa=p,
        molar_mass_kg_mol=float(molar_mass_kg_mol),
        gamma=float(gamma),
        density_kg_m3=float(rho),
        dynamic_viscosity_Pa_s=float(eta),
        heat_capacity_cp_J_kgK=float(cp),
        thermal_conductivity_W_mK=float(kappa),
        sound_speed_m_s=float(c0),
        prandtl=float(pr),
        bulk_modulus_Pa=float(rho * c0 * c0),
    )


def viscous_boundary_layer_thickness(
    f_Hz: float,
    rho0: float = 1.2043175745358388,
    eta: float = 1.8139686307339444e-5,
) -> float:
    omega = 2 * math.pi * max(float(f_Hz), 1e-30)
    return math.sqrt(2 * eta / (rho0 * omega))


def thermal_boundary_layer_thickness(
    f_Hz: float,
    rho0: float = 1.2043175745358388,
    cp: float = 1005.4220711271306,
    kappa: float = 0.02576818523619657,
) -> float:
    omega = 2 * math.pi * max(float(f_Hz), 1e-30)
    return math.sqrt(2 * kappa / (rho0 * cp * omega))


def slit_loss_scale(
    f_Hz: float,
    slit_height_m: float,
    *,
    properties: AirThermophysicalProperties | None = None,
) -> dict[str, float]:
    props = properties or comsol_air_properties()
    dv = viscous_boundary_layer_thickness(
        f_Hz, props.density_kg_m3, props.dynamic_viscosity_Pa_s
    )
    dt = thermal_boundary_layer_thickness(
        f_Hz,
        props.density_kg_m3,
        props.heat_capacity_cp_J_kgK,
        props.thermal_conductivity_W_mK,
    )
    half = max(float(slit_height_m) / 2.0, 1e-300)
    return {
        "f_Hz": float(f_Hz),
        "slit_height_m": float(slit_height_m),
        "viscous_delta_m": float(dv),
        "thermal_delta_m": float(dt),
        "viscous_delta_over_half_height": float(dv / half),
        "thermal_delta_over_half_height": float(dt / half),
    }


def _safe_tanh_over_x(x: complex) -> complex:
    ax = abs(x)
    if ax < 1e-6:
        # tanh(x)/x = 1 - x^2/3 + 2*x^4/15 - 17*x^6/315 + ...
        x2 = x * x
        return 1.0 - x2 / 3.0 + 2.0 * x2 * x2 / 15.0 - 17.0 * x2**3 / 315.0
    return cmath.tanh(x) / x


@dataclass(frozen=True)
class NarrowRegionCoefficients:
    frequency_Hz: float
    slit_height_m: float
    rho_eq_over_rho0: complex
    bulk_eq_over_bulk0: complex
    stiffness_factor: complex
    mass_factor: complex
    complex_wavenumber_over_k0: complex
    characteristic_impedance_over_Z0: complex
    boundary_layer: dict[str, float]
    air_properties: AirThermophysicalProperties
    model: str = "native_parallel_plate_thermoviscous"
    harmonic_convention: str = "exp(+i*omega*t)"

    def to_jsonable(self) -> dict[str, Any]:
        def z(v: complex) -> dict[str, float]:
            return {"real": float(v.real), "imag": float(v.imag)}

        d = asdict(self)
        for k in (
            "rho_eq_over_rho0",
            "bulk_eq_over_bulk0",
            "stiffness_factor",
            "mass_factor",
            "complex_wavenumber_over_k0",
            "characteristic_impedance_over_Z0",
        ):
            d[k] = z(getattr(self, k))
        return d

    @property
    def passive(self) -> bool:
        # For exp(+i wt), Im(rho_eff) <= 0, Im(1/K_eff) <= 0.
        return bool(
            self.rho_eq_over_rho0.imag <= 1e-12
            and self.mass_factor.imag <= 1e-12
            and self.stiffness_factor.imag >= -1e-12
        )


def equivalent_narrow_region_coefficients(
    f_Hz: float,
    slit_height_m: float,
    *,
    rho0: float | None = None,
    c0: float | None = None,
    eta: float | None = None,
    gamma: float = 1.4,
    prandtl: float | None = None,
    temperature_K: float = 293.15,
    pressure_Pa: float = 101325.0,
) -> NarrowRegionCoefficients:
    """Native parallel-plate thermoviscous effective properties.

    This is the non-calibrated Zwikker-Kosten/Stinson slit model used as a
    replacement for COMSOL Narrow Region Acoustics.  The former implementation
    used ``sqrt(i) * a/delta``.  Since ``delta=sqrt(2*nu/omega)``, the exact
    transverse argument is ``a*sqrt(i*omega/nu)=(1+i)*a/delta``.  The missing
    sqrt(2) factor materially under-estimated losses and is corrected here.

    The project pressure weak form is ``K - k0^2 M``.  Consequently
    ``rho0/rho_eff`` multiplies the gradient term and ``K0/K_eff`` multiplies
    the pressure mass term.
    """
    props0 = comsol_air_properties(
        temperature_K=temperature_K, pressure_Pa=pressure_Pa, gamma=gamma
    )
    rho = float(props0.density_kg_m3 if rho0 is None else rho0)
    c = float(props0.sound_speed_m_s if c0 is None else c0)
    mu = float(props0.dynamic_viscosity_Pa_s if eta is None else eta)
    cp = float(props0.heat_capacity_cp_J_kgK)
    kappa = float(props0.thermal_conductivity_W_mK)
    if prandtl is not None:
        # Preserve thermodynamic consistency if an explicit Pr is requested.
        kappa = mu * cp / float(prandtl)
    props = AirThermophysicalProperties(
        temperature_K=props0.temperature_K,
        pressure_Pa=props0.pressure_Pa,
        molar_mass_kg_mol=props0.molar_mass_kg_mol,
        gamma=float(gamma),
        density_kg_m3=rho,
        dynamic_viscosity_Pa_s=mu,
        heat_capacity_cp_J_kgK=cp,
        thermal_conductivity_W_mK=kappa,
        sound_speed_m_s=c,
        prandtl=mu * cp / kappa,
        bulk_modulus_Pa=rho * c * c,
    )
    f = max(float(f_Hz), 1e-30)
    h = float(slit_height_m)
    a = h / 2.0
    omega = 2.0 * math.pi * f
    # Exact parallel-plate transverse wave arguments for exp(+i omega t).
    x_v = a * cmath.sqrt(1j * omega * rho / mu)
    x_t = a * cmath.sqrt(1j * omega * rho * cp / kappa)
    Fv = _safe_tanh_over_x(x_v)
    Ft = _safe_tanh_over_x(x_t)
    rho_ratio = 1.0 / (1.0 - Fv)
    # beta_eff/beta0 = 1 + (gamma-1) Ft; hence K_eff/K0 is reciprocal.
    mass_factor = 1.0 + (gamma - 1.0) * Ft
    bulk_ratio = 1.0 / mass_factor
    stiffness_factor = 1.0 / rho_ratio
    k_ratio = cmath.sqrt(rho_ratio / bulk_ratio)
    z_ratio = cmath.sqrt(rho_ratio * bulk_ratio)
    # Choose the passive branch for exp(+i wt): attenuation has Im(k) <= 0.
    if k_ratio.imag > 0:
        k_ratio = -k_ratio
    if z_ratio.real < 0:
        z_ratio = -z_ratio
    result = NarrowRegionCoefficients(
        frequency_Hz=f,
        slit_height_m=h,
        rho_eq_over_rho0=rho_ratio,
        bulk_eq_over_bulk0=bulk_ratio,
        stiffness_factor=stiffness_factor,
        mass_factor=mass_factor,
        complex_wavenumber_over_k0=k_ratio,
        characteristic_impedance_over_Z0=z_ratio,
        boundary_layer=slit_loss_scale(f, h, properties=props),
        air_properties=props,
    )
    if not result.passive:
        raise FloatingPointError(
            f"non-passive native NRA coefficients at {f:g} Hz, h={h:g} m: {result}"
        )
    return result


def narrow_region_dissipation(
    pressure: Any,
    stiffness_matrix: Any,
    mass_matrix: Any,
    omega: float,
    coefficients: NarrowRegionCoefficients,
) -> dict[str, float]:
    """Return time-average viscous and thermal loss for one NRA domain.

    Matrices are the unscaled axisymmetric P1/P2 integrals
    ``K=integral grad(N).grad(N) dV`` and ``M=integral N*N dV``.  For the
    exp(+i wt) convention, passivity gives Im(rho0/rho_eff)>=0 and
    Im(K0/K_eff)<=0.
    """
    import numpy as np

    p = np.asarray(pressure, dtype=complex)
    grad_energy = float(np.real(np.vdot(p, stiffness_matrix @ p)))
    pressure_energy = float(np.real(np.vdot(p, mass_matrix @ p)))
    props = coefficients.air_properties
    visc = 0.5 * coefficients.stiffness_factor.imag * grad_energy / (
        max(float(omega), 1e-300) * props.density_kg_m3
    )
    thermal = -0.5 * float(omega) * coefficients.mass_factor.imag * pressure_energy / (
        props.bulk_modulus_Pa
    )
    return {
        "viscous_W": float(max(visc, 0.0)),
        "thermal_W": float(max(thermal, 0.0)),
        "total_W": float(max(visc + thermal, 0.0)),
        "gradient_quadratic_Pa2_m": grad_energy,
        "pressure_quadratic_Pa2_m3": pressure_energy,
    }
