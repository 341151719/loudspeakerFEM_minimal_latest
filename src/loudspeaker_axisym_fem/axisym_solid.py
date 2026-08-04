from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class LinearElasticMaterial:
    name: str
    rho: float
    E: float
    nu: float
    loss_factor: float | None = None
    beta_dK: float | None = None

    @property
    def lame_lambda(self) -> float:
        return self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

    @property
    def lame_mu(self) -> float:
        return self.E / (2.0 * (1.0 + self.nu))


def rayleigh_beta_from_loss_factor(eta: float, omega_loss: float) -> float:
    return eta / omega_loss


def comsol_solid_materials(omega_loss: float) -> Dict[str, LinearElasticMaterial]:
    return {
        "composite": LinearElasticMaterial("Composite", 1200.0, 2e9, 0.42, loss_factor=0.04),
        "cloth": LinearElasticMaterial("Cloth", 650.0, 0.58e9, 0.30, beta_dK=rayleigh_beta_from_loss_factor(0.14, omega_loss)),
        "foam": LinearElasticMaterial("Foam", 67.0, 5e6, 0.40, beta_dK=rayleigh_beta_from_loss_factor(0.46, omega_loss)),
        "coil": LinearElasticMaterial("Coil", 4500.0, 110e9, 0.35, loss_factor=0.05),
        "glass_fiber": LinearElasticMaterial("Glass Fiber", 2000.0, 70e9, 0.33, loss_factor=0.04),
    }


class SolidAssemblyNotYetImplemented(RuntimeError):
    pass


def assemble_axisymmetric_solid(*args, **kwargs):
    """Placeholder for axisymmetric structural FEM assembly.

    Required unknowns: u_r and u_z.  Strain terms include epsilon_rr,
    epsilon_zz, epsilon_rz, and epsilon_phi_phi = u_r/r.  The next stage will
    assemble M, K, damping, pressure load coupling, and eigenfrequency solve.
    """
    raise SolidAssemblyNotYetImplemented("axisymmetric solid mechanics assembly is pending")
