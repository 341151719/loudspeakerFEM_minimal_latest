"""Build representative FR10 rocking GIFs for three root causes and mode forms."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import meshio
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "fr10_full360_cyclic"
sys.path.insert(0, str(MODULE))

from animate_full360_results import write_membrane_surface_gif  # noqa: E402
from cyclic_full360_solver import run_phase_diagnostics  # noqa: E402


TARGET = Path("/mnt/c/Users/Administrator/Documents/扬声器摇摆模态")
DATA = TARGET / "fr10_rocking_fem_data" / "phase_diagnostics"
DERIVED = TARGET / "fr10_rocking_fem_data" / "derived_cause_fields"


def source_path(frequency: float) -> Path:
    tag = f"{frequency:g}Hz"
    return DATA / f"{tag}_k1" / f"structure_full360_{tag}_diagnostic_k1_P2.vtu"


def ensure_field(frequency: float) -> Path:
    path = source_path(frequency)
    if not path.exists():
        run_phase_diagnostics(frequency, (1,), outroot=DATA)
    return path


def write_variant(source: Path, output: Path, form: str, angle_deg: float = 0.0) -> None:
    mesh = meshio.read(source)
    field = np.asarray(mesh.point_data["u_real_m"]) + 1j * np.asarray(
        mesh.point_data["u_imag_m"]
    )
    angle = np.deg2rad(angle_deg)
    a = field.real
    b = field.imag
    if form == "standing":
        spatial = a * np.cos(angle) + b * np.sin(angle)
        result = spatial.astype(complex)
    elif form == "elliptical":
        result = a + 0.45j * b
    elif form == "circular":
        result = field
    else:
        raise ValueError(form)
    mesh.point_data["u_real_m"] = result.real
    mesh.point_data["u_imag_m"] = result.imag
    mesh.point_data["u_abs_m"] = np.linalg.norm(result, axis=1)
    mesh.point_data["u_z_phase_deg"] = np.degrees(np.angle(result[:, 2]))
    output.parent.mkdir(parents=True, exist_ok=True)
    mesh.write(output)


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    jobs = [
        (80.0, "standing", 0.0, "Suspension stiffness asymmetry - fixed-axis rock", "01_悬挂刚度不均_80Hz_定轴摇摆.gif"),
        (300.0, "standing", 45.0, "Bl asymmetry - 45-deg fixed-axis rock", "02_Bl不均_300Hz_45度定轴摇摆.gif"),
        (2000.0, "standing", 0.0, "Mass eccentricity - fixed-axis rock", "03_质量偏心_2000Hz_定轴摇摆.gif"),
        (2000.0, "elliptical", 0.0, "Combined defects - elliptical precession", "04_复合不对称_2000Hz_椭圆进动.gif"),
        (2000.0, "circular", 0.0, "m=1 doublet - circular precession", "05_m1双重态_2000Hz_圆形进动.gif"),
    ]
    manifest = []
    for frequency, form, angle, title, filename in jobs:
        source = ensure_field(frequency)
        derived = DERIVED / f"{Path(filename).stem}.vtu"
        write_variant(source, derived, form, angle)
        result = write_membrane_surface_gif(
            derived,
            TARGET / filename,
            frequency,
            title,
            frames=36,
            fps=12,
            kclass="k1_m1",
            physical_electrical_drive=False,
        )
        result.update(
            {
                "cause_or_form": title,
                "standing_axis_angle_deg": angle if form == "standing" else None,
                "normalization_note": "Shape-normalized k=1 diagnostic; deformation is exaggerated and is not a measured FR10 amplitude.",
            }
        )
        manifest.append(result)
    (TARGET / "摇摆模态GIF说明.json").write_text(
        json.dumps({"status": "complete", "items": manifest}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
