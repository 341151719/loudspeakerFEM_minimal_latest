from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import math
import numpy as np


@dataclass(frozen=True)
class LocalPanelMode:
    frequency_Hz: float
    mass_per_area_kg_m2: float
    loss_factor: float = 0.05
    participation: float = 1.0

    def specific_impedance(self, f: float) -> complex:
        """Specific panel impedance p/v_n for one locally reacting SDOF mode.

        This is a calibrated local-reaction wall model, not a full plate PDE.  It
        captures the first-order effect of a panel mobility peak and loss, which is
        the intended intermediate layer before a full 3D shell-acoustic FSI model.
        """
        w = 2.0 * np.pi * max(float(f), 1e-9)
        w0 = 2.0 * np.pi * max(float(self.frequency_Hz), 1e-9)
        m = float(self.mass_per_area_kg_m2) / max(float(self.participation), 1e-12)
        k = m * w0 * w0
        c = float(self.loss_factor) * k / w0
        return c + 1j * w * m + k / (1j * w)


@dataclass(frozen=True)
class LocalPanelImpedance:
    modes: Sequence[LocalPanelMode]
    residual_resistance_Pa_s_m: float = 0.0

    def specific_impedance(self, f: float) -> complex:
        # modal mobilities add in parallel; residual resistance is a parallel sink
        Y = 0.0 + 0.0j
        for mode in self.modes:
            Y += 1.0 / mode.specific_impedance(f)
        if self.residual_resistance_Pa_s_m > 0:
            Y += 1.0 / self.residual_resistance_Pa_s_m
        if abs(Y) < 1e-300:
            return complex(np.inf)
        return 1.0 / Y


@dataclass(frozen=True)
class DiaphragmMode:
    frequency_Hz: float
    q_factor: float
    amplitude: float
    radial_order: int = 0

    def response(self, f: float) -> complex:
        w = 2.0 * np.pi * max(float(f), 1e-9)
        w0 = 2.0 * np.pi * max(float(self.frequency_Hz), 1e-9)
        # dimensionless second-order modal transfer function normalized to 1 at low f
        return self.amplitude * (w0 * w0) / (w0 * w0 - w * w + 1j * w * w0 / max(self.q_factor, 1e-9))

    def shape(self, r: np.ndarray, radius_m: float) -> np.ndarray:
        x = np.clip(np.asarray(r, dtype=float) / max(float(radius_m), 1e-12), 0.0, 1.0)
        if self.radial_order == 0:
            return 1.0 - 2.0 * x * x
        if self.radial_order == 1:
            return x * (1.0 - x) * np.cos(np.pi * x)
        # generic smooth alternating radial family
        return (1.0 - x) * np.cos((self.radial_order + 0.5) * np.pi * x)


@dataclass(frozen=True)
class ModalDiaphragmVelocity:
    radius_m: float
    modes: Sequence[DiaphragmMode]
    normalize_area_average: bool = True

    def velocity_profile(self, f: float, r: np.ndarray, piston_velocity: complex = 1.0 + 0.0j) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        profile = np.ones_like(r, dtype=complex)
        for mode in self.modes:
            profile += mode.response(f) * mode.shape(r, self.radius_m)
        if self.normalize_area_average:
            # keep volume velocity equal to piston_velocity * Sd so the modal model
            # changes radiation/directivity without silently changing low-frequency SPL.
            rr = np.linspace(0.0, self.radius_m, 512)
            pp = np.ones_like(rr, dtype=complex)
            for mode in self.modes:
                pp += mode.response(f) * mode.shape(rr, self.radius_m)
            avg = 2.0 * np.trapz(pp * rr, rr) / (self.radius_m ** 2)
            if abs(avg) > 1e-12:
                profile = profile / avg
        return complex(piston_velocity) * profile

    def effective_volume_velocity(self, f: float, piston_velocity: complex = 1.0 + 0.0j) -> complex:
        rr = np.linspace(0.0, self.radius_m, 512)
        vv = self.velocity_profile(f, rr, piston_velocity)
        return complex(2.0 * np.pi * np.trapz(vv * rr, rr))
