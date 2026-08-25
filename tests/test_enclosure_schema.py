from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path[:0] = [str(ROOT / "src")]

from loudspeaker_axisym_fem.enclosure_schema import (  # noqa: E402
    EnclosureSchemaError,
    load_enclosure_config,
    validate_enclosure_config,
)


CONFIG_NAMES = (
    "base_axisym",
    "open_back",
    "sealed_lossless",
    "sealed_thermoviscous",
    "vented_rear_coaxial",
    "passive_radiator_rear_coaxial",
)
CONFIG_DIR = ROOT / "configs" / "enclosures"


def _raw(name: str) -> dict:
    return json.loads((CONFIG_DIR / f"{name}.json").read_text(encoding="utf-8"))


def test_all_phase1_demonstrators_validate_and_report_volume_contract():
    validated = [load_enclosure_config(CONFIG_DIR / f"{name}.json") for name in CONFIG_NAMES]

    assert all(item.demonstrator for item in validated)
    assert {item.case for item in validated} == set(CONFIG_NAMES)
    assert all(item.computed_net_volume_m3 > 0.0 for item in validated)

    comparable = {
        item.case: item.net_volume_target_m3
        for item in validated
        if item.case in {"sealed_lossless", "sealed_thermoviscous", "vented_rear_coaxial", "passive_radiator_rear_coaxial"}
    }
    assert len(set(comparable.values())) == 1


def test_schema_requires_explicit_demonstrator_provenance_and_units():
    cfg = _raw("sealed_lossless")

    missing_flag = copy.deepcopy(cfg)
    del missing_flag["demonstrator"]
    with pytest.raises(EnclosureSchemaError):
        validate_enclosure_config(missing_flag)

    bare_numeric = copy.deepcopy(cfg)
    bare_numeric["air"]["rho0"] = 1.2
    with pytest.raises(EnclosureSchemaError, match="unit"):
        validate_enclosure_config(bare_numeric)

    not_demo = copy.deepcopy(cfg)
    not_demo["demonstrator"] = False
    with pytest.raises(EnclosureSchemaError, match="demonstrator"):
        validate_enclosure_config(not_demo)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("net_volume_target_m3",), 0.0),
        (("volume_contract", "gross_internal_volume_m3"), -1.0),
        (("passive_radiator", "Mms_kg"), 0.0),
        (("passive_radiator", "Cms_m_N"), -1.0),
        (("passive_radiator", "Rms_N_s_m"), 0.0),
        (("geometry", "inner_radius_m"), 0.0),
        (("geometry", "outer_radius_m"), 0.05),
    ],
)
def test_schema_rejects_nonpositive_materials_and_invalid_geometry(path, value):
    cfg = _raw("passive_radiator_rear_coaxial")
    target = cfg
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(EnclosureSchemaError):
        validate_enclosure_config(cfg)


def test_schema_rejects_nonfinite_values_and_unknown_case():
    cfg = _raw("sealed_lossless")

    nonfinite = copy.deepcopy(cfg)
    nonfinite["air"]["rho0_kg_m3"] = float("nan")
    with pytest.raises(EnclosureSchemaError, match="finite"):
        validate_enclosure_config(nonfinite)

    unknown = copy.deepcopy(cfg)
    unknown["case"] = "unapproved_case"
    with pytest.raises(EnclosureSchemaError, match="case"):
        validate_enclosure_config(unknown)


def test_schema_rejects_topology_and_geometry_conflicts():
    both_rear_features = _raw("passive_radiator_rear_coaxial")
    both_rear_features["port"]["enabled"] = True
    with pytest.raises(EnclosureSchemaError, match="port.*PR|PR.*port"):
        validate_enclosure_config(both_rear_features)

    pml_crosses_entity = _raw("sealed_lossless")
    pml_crosses_entity["geometry"]["pml_inner_radius_m"] = 0.01
    with pytest.raises(EnclosureSchemaError, match="PML"):
        validate_enclosure_config(pml_crosses_entity)

    port_hits_driver = _raw("vented_rear_coaxial")
    port_hits_driver["geometry"]["port_penetration_into_box_m"] = port_hits_driver["geometry"]["inner_depth_m"]
    with pytest.raises(EnclosureSchemaError, match="intersect|相交"):
        validate_enclosure_config(port_hits_driver)


def test_schema_rejects_loss_model_cross_section_mismatch():
    cfg = _raw("vented_rear_coaxial")
    cfg["port"]["cross_section"] = "rectangular"

    with pytest.raises(EnclosureSchemaError, match="截面|cross.section"):
        validate_enclosure_config(cfg)


def test_explicit_fem_radiation_cannot_reuse_lumped_port_corrections():
    cfg = _raw("vented_rear_coaxial")
    cfg["port"]["explicit_fem"]["reuse_end_correction"] = True
    with pytest.raises(EnclosureSchemaError, match="端部修正|end.correction"):
        validate_enclosure_config(cfg)

    cfg = _raw("vented_rear_coaxial")
    cfg["port"]["explicit_fem"]["reuse_radiation_mass"] = True
    with pytest.raises(EnclosureSchemaError, match="辐射质量|radiation.mass"):
        validate_enclosure_config(cfg)


def test_json_loader_rejects_nonstandard_nonfinite_literal(tmp_path):
    source = (CONFIG_DIR / "sealed_lossless.json").read_text(encoding="utf-8")
    target = tmp_path / "bad.json"
    target.write_text(source.replace("1.2043175745358388", "NaN", 1), encoding="utf-8")

    with pytest.raises(EnclosureSchemaError, match="finite|NaN"):
        load_enclosure_config(target)


def test_phase1_thermoviscous_choice_is_a_declared_applicability_only():
    cfg = _raw("sealed_thermoviscous")
    validated = validate_enclosure_config(cfg)

    assert validated.raw["thermoviscous"]["implemented_in_phase1"] is False
    assert validated.raw["thermoviscous"]["model"] in {
        "BLI_boundary_operator_declared",
        "narrow_region_lrf_declared",
    }
