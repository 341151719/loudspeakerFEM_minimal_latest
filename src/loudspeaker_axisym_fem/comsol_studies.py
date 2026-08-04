from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ComsolStudySpec:
    tag: str
    label: str
    solves_mf: bool
    solves_acpr: bool
    solves_solid: bool
    narrow_region_enabled: bool
    eigenfrequency: bool = False


COMSOL_STUDIES = [
    ComsolStudySpec("std1", "Study 1 - Magnetic Fields", True, False, False, True, False),
    ComsolStudySpec("std2", "Study 2 - Complete Model", True, True, True, True, False),
    ComsolStudySpec("std3", "Study 3 - Complete Model, Without Narrow Region Acoustics", True, True, True, False, False),
    ComsolStudySpec("std4", "Study 4 - Eigenfrequency", False, False, True, False, True),
]


def iso_1_12_frequencies_10_to_8000() -> List[float]:
    return [
        10, 10.6, 11.2, 11.8, 12.5, 13.2, 14, 15, 16, 17, 18, 19, 20, 21.2,
        22.4, 23.6, 25, 26.5, 28, 30, 31.5, 33.5, 35.5, 37.5, 40, 42.5,
        45, 47.5, 50, 53, 56, 60, 63, 67, 71, 75, 80, 85, 90, 95, 100,
        106, 112, 118, 125, 132, 140, 150, 160, 170, 180, 190, 200, 212,
        224, 236, 250, 265, 280, 300, 315, 335, 355, 375, 400, 425, 450,
        475, 500, 530, 560, 600, 630, 670, 710, 750, 800, 850, 900, 950,
        1000, 1060, 1120, 1180, 1250, 1320, 1400, 1500, 1600, 1700, 1800,
        1900, 2000, 2120, 2240, 2360, 2500, 2650, 2800, 3000, 3150, 3350,
        3550, 3750, 4000, 4250, 4500, 4750, 5000, 5300, 5600, 6000, 6300,
        6700, 7100, 7500, 8000,
    ]
