from __future__ import annotations

import hashlib
from pathlib import Path

import meshio
import numpy as np

from loudspeaker_axisym_fem.enclosure_geometry import (
    expected_boundary_names,
    expected_domain_names,
    generate_reference_mesh,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "A": ROOT / "configs/enclosures/open_back.json",
    "B": ROOT / "configs/enclosures/sealed_lossless.json",
    "C": ROOT / "configs/enclosures/sealed_thermoviscous.json",
    "D": ROOT / "configs/enclosures/vented_rear_coaxial.json",
    "E": ROOT / "configs/enclosures/passive_radiator_rear_coaxial.json",
}


def _mesh_signature(path: Path) -> str:
    mesh = meshio.read(path)
    digest = hashlib.sha256()

    def add_array(value: np.ndarray) -> None:
        arr = np.ascontiguousarray(value)
        digest.update(str(arr.shape).encode("ascii"))
        digest.update(arr.dtype.str.encode("ascii"))
        digest.update(arr.tobytes())

    add_array(mesh.points)
    for cell_type in ("line", "triangle"):
        cells = [block.data for block in mesh.cells if block.type == cell_type]
        assert cells, f"missing {cell_type} cells"
        add_array(np.concatenate(cells, axis=0))
        tags = mesh.cell_data_dict["gmsh:physical"][cell_type]
        add_array(np.asarray(tags, dtype=np.int64))
    for name in sorted(mesh.field_data):
        digest.update(name.encode("utf-8"))
        add_array(np.asarray(mesh.field_data[name], dtype=np.int64))
    return digest.hexdigest()


def test_all_reference_cases_generate_l0_with_stable_named_groups_and_nonnegative_r(tmp_path):
    generated = {}
    for case_id, config_path in CONFIGS.items():
        result = generate_reference_mesh(config_path, "L0", tmp_path / f"{case_id}.msh")
        generated[case_id] = result.path
        mesh = meshio.read(result.path)
        assert any(block.type == "triangle" and len(block.data) > 0 for block in mesh.cells)
        assert np.min(mesh.points[:, 0]) >= -1.0e-12

        required = set(expected_domain_names(case_id)) | set(expected_boundary_names(case_id))
        assert required <= set(mesh.field_data)
        for name in required:
            tag, dimension = map(int, mesh.field_data[name])
            assert tag > 0
            assert dimension in {1, 2}

    # C differs from B only in the declared loss/applicability configuration;
    # its deterministic reference geometry and all physical labels are equal.
    assert _mesh_signature(generated["B"]) == _mesh_signature(generated["C"])
