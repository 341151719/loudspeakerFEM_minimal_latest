from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import math


@dataclass(frozen=True)
class ComsolDriverParameters:
    N0: int = 100
    V0_peak_V: float = 3.55
    f_loss_Hz: float = 40.0
    fmax_Hz: float = 8000.0
    c0_m_s: float = 343.0

    @property
    def omega_loss(self) -> float:
        return 2.0 * math.pi * self.f_loss_Hz

    @property
    def lam0_m(self) -> float:
        return self.c0_m_s / self.fmax_Hz


DOMAIN_GROUPS: Dict[str, List[int]] = {
    "pml": [1, 5],
    "soft_iron": [6, 23],
    "composite": [3, 21],
    "cloth": [20],
    "foam": [25],
    "coil": [17, 18, 19],
    "glass_fiber": [9, 10, 11, 12, 13, 14, 15, 16],
    "generic_ferrite": [24],
    "air": [2, 4, 7, 8, 22],
    "structural": [3, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 25],
    "magnetic": [6, 17, 18, 19, 23, 24],
    "narrow_region_0p4mm": [8],
    "narrow_region_0p2mm": [22],
}

BOUNDARY_GROUPS: Dict[str, List[int]] = {
    "exterior_field_boundary": [93],
    "fixed_spider_surround": [81, 85],
    "mesh_distribution_2": [22, 38, 41, 45],
    "pml_distribution_8": [87, 88],
    "boundary_layer_core": [12, 53, 95, 96, 97, 98],
    "boundary_layer_exterior": [93],
}

MATERIALS = {
    "Composite": {"domains": DOMAIN_GROUPS["composite"], "E_Pa": 2e9, "nu": 0.42, "rho_kg_m3": 1200.0, "lossfactor": 0.04},
    "Cloth": {"domains": DOMAIN_GROUPS["cloth"], "E_Pa": 0.58e9, "nu": 0.30, "rho_kg_m3": 650.0, "beta_dK": "0.14/omega_loss"},
    "Foam": {"domains": DOMAIN_GROUPS["foam"], "E_Pa": 5e6, "nu": 0.40, "rho_kg_m3": 67.0, "beta_dK": "0.46/omega_loss"},
    "Coil": {"domains": DOMAIN_GROUPS["coil"], "E_Pa": 110e9, "nu": 0.35, "rho_kg_m3": 4500.0, "lossfactor": 0.05},
    "Glass Fiber": {"domains": DOMAIN_GROUPS["glass_fiber"], "E_Pa": 70e9, "nu": 0.33, "rho_kg_m3": 2000.0, "lossfactor": 0.04},
    "Generic Ferrite": {"domains": DOMAIN_GROUPS["generic_ferrite"], "E_Pa": 200e9, "nu": 0.30, "rho_kg_m3": 5000.0, "Br_T": 0.4},
    "Soft Iron": {"domains": DOMAIN_GROUPS["soft_iron"], "sigma_S_m": 1.12e7, "BH_curve": "see SOFT_IRON_BH_TABLE"},
}

SOFT_IRON_BH_TABLE: Tuple[Tuple[float, float], ...] = (
    (0.0, 0.0),
    (663.146, 1.0),
    (1067.5, 1.1),
    (1705.23, 1.2),
    (2463.11, 1.3),
    (3841.67, 1.4),
    (5425.74, 1.5),
    (7957.75, 1.6),
    (12298.3, 1.7),
    (20462.8, 1.8),
    (32169.6, 1.9),
    (61213.4, 2.0),
    (111408.0, 2.1),
    (188487.757, 2.2),
    (267930.364, 2.3),
    (347507.836, 2.4),
)

COMSOL_TARGETS = {
    "BL_N_per_A": 10.48,
    "dc_resistance_ohm": 5.6,
    "nominal_impedance_ohm": 6.3,
    "mechanical_impedance_peak_Hz_approx": 50.0,
    "flat_operating_range_Hz": [100.0, 1500.0],
    "first_breakup_Hz_approx": 2350.0,
    "back_cavity_mode_lossless_Hz_approx": 600.0,
}
