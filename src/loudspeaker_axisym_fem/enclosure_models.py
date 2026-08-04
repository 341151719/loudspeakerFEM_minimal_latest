from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import numpy as np


@dataclass(frozen=True)
class AirProperties:
    rho0: float = 1.21
    c0: float = 343.0


@dataclass(frozen=True)
class ClosedBox:
    volume_m3: float
    loss_resistance_Pa_s_m3: float | None = None

    def compliance(self, air: AirProperties = AirProperties()) -> float:
        return float(self.volume_m3) / (air.rho0 * air.c0 ** 2)

    def input_admittance(self, f: float, air: AirProperties = AirProperties()) -> complex:
        w = 2.0 * np.pi * max(float(f), 1e-9)
        Y = 1j * w * self.compliance(air)
        if self.loss_resistance_Pa_s_m3 and self.loss_resistance_Pa_s_m3 > 0:
            Y += 1.0 / self.loss_resistance_Pa_s_m3
        return Y

    def input_impedance(self, f: float, air: AirProperties = AirProperties()) -> complex:
        return 1.0 / self.input_admittance(f, air)


@dataclass(frozen=True)
class LeakPath:
    radius_m: float
    length_m: float
    resistance_Pa_s_m3: float | None = None

    def impedance(self, f: float, air: AirProperties = AirProperties()) -> complex:
        a = max(float(self.radius_m), 1e-9)
        S = np.pi * a * a
        leff = float(self.length_m) + 1.7 * a
        M = air.rho0 * leff / S
        R = self.resistance_Pa_s_m3
        if R is None:
            # low-Re engineering placeholder; should be calibrated from impedance data
            eta_air = 1.84e-5
            R = 8.0 * eta_air * max(float(self.length_m), 1e-6) / (np.pi * a ** 4)
        return complex(R) + 1j * 2.0 * np.pi * f * M


@dataclass(frozen=True)
class Port:
    radius_m: float
    length_m: float
    resistance_Pa_s_m3: float = 0.0
    end_correction_radii: float = 1.46

    def area(self) -> float:
        return float(np.pi * self.radius_m ** 2)

    def effective_length(self) -> float:
        return float(self.length_m + self.end_correction_radii * self.radius_m)

    def acoustic_mass(self, air: AirProperties = AirProperties()) -> float:
        return float(air.rho0 * self.effective_length() / self.area())

    def impedance(self, f: float, air: AirProperties = AirProperties()) -> complex:
        return complex(self.resistance_Pa_s_m3) + 1j * 2.0 * np.pi * float(f) * self.acoustic_mass(air)

    def first_pipe_resonance_Hz(self, air: AirProperties = AirProperties(), closed_open: bool = True) -> float:
        # Most loudspeaker ports are closer to open-open for longitudinal pipe modes;
        # closed_open=True is provided for ducts flush at one reactive termination.
        denom = 4.0 * self.effective_length() if closed_open else 2.0 * self.effective_length()
        return float(air.c0 / denom)


@dataclass(frozen=True)
class VentedBox:
    box: ClosedBox
    port: Port
    leak: LeakPath | None = None

    def helmholtz_frequency_Hz(self, air: AirProperties = AirProperties()) -> float:
        return float(1.0 / (2.0 * np.pi * math.sqrt(self.port.acoustic_mass(air) * self.box.compliance(air))))

    def input_admittance(self, f: float, air: AirProperties = AirProperties()) -> complex:
        Y = self.box.input_admittance(f, air) + 1.0 / self.port.impedance(f, air)
        if self.leak is not None:
            Y += 1.0 / self.leak.impedance(f, air)
        return Y

    def input_impedance(self, f: float, air: AirProperties = AirProperties()) -> complex:
        return 1.0 / self.input_admittance(f, air)

    def port_volume_velocity(self, f: float, box_pressure_Pa: complex, air: AirProperties = AirProperties()) -> complex:
        return complex(box_pressure_Pa) / self.port.impedance(f, air)

    def port_velocity(self, f: float, box_pressure_Pa: complex, air: AirProperties = AirProperties()) -> complex:
        return self.port_volume_velocity(f, box_pressure_Pa, air) / self.port.area()


@dataclass(frozen=True)
class PassiveRadiator:
    Sd_m2: float
    Mms_kg: float
    Cms_m_N: float
    Rms_N_s_m: float

    def mechanical_impedance(self, f: float) -> complex:
        w = 2.0 * np.pi * max(float(f), 1e-9)
        return self.Rms_N_s_m + 1j * w * self.Mms_kg + 1.0 / (1j * w * self.Cms_m_N)

    def acoustic_impedance(self, f: float) -> complex:
        S = max(float(self.Sd_m2), 1e-12)
        return self.mechanical_impedance(f) / (S * S)

    def resonance_Hz(self) -> float:
        return float(1.0 / (2.0 * np.pi * math.sqrt(self.Mms_kg * self.Cms_m_N)))


@dataclass(frozen=True)
class PassiveRadiatorBox:
    box: ClosedBox
    radiator: PassiveRadiator
    leak: LeakPath | None = None

    def input_admittance(self, f: float, air: AirProperties = AirProperties()) -> complex:
        Y = self.box.input_admittance(f, air) + 1.0 / self.radiator.acoustic_impedance(f)
        if self.leak is not None:
            Y += 1.0 / self.leak.impedance(f, air)
        return Y

    def input_impedance(self, f: float, air: AirProperties = AirProperties()) -> complex:
        return 1.0 / self.input_admittance(f, air)

    def radiator_velocity(self, f: float, box_pressure_Pa: complex) -> complex:
        U = complex(box_pressure_Pa) / self.radiator.acoustic_impedance(f)
        return U / self.radiator.Sd_m2


@dataclass(frozen=True)
class PorousRegionSpec:
    name: str
    mode: str  # empty | lining | fullfill | custom_mask
    sigma_Pa_s_m2: float | None = None
    thickness_m: float = 0.03
    notes: str = ""


def loudspeaker_side_mechanical_impedance_from_acoustic(Za_Pa_s_m3: complex, Sd_m2: float) -> complex:
    """Convert box acoustic impedance p/U into diaphragm mechanical impedance F/v."""
    return complex(Za_Pa_s_m3) * float(Sd_m2) ** 2


def acoustic_impedance_from_mechanical(Zm_N_s_m: complex, Sd_m2: float) -> complex:
    return complex(Zm_N_s_m) / max(float(Sd_m2) ** 2, 1e-300)
