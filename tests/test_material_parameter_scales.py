import pytest

from coupled_solver import solid_materials_from_config
from p2_axisym_solid import default_stage4_materials


def test_material_parameter_scales_change_only_requested_domain_and_property():
    defaults = default_stage4_materials()
    materials = solid_materials_from_config({
        "structure": {
            "material_parameter_scales": {
                "21": {"density": 0.5, "youngs_modulus": 2.0},
            }
        }
    })

    assert materials[21].rho == pytest.approx(0.5 * defaults[21].rho)
    assert materials[21].E == pytest.approx(2.0 * defaults[21].E)
    assert materials[21].nu == defaults[21].nu
    assert materials[25] == defaults[25]
    assert defaults[21] == default_stage4_materials()[21]


def test_material_parameter_scales_preserve_default_path_when_absent():
    assert solid_materials_from_config({}) is None


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan")])
def test_material_parameter_scales_reject_invalid_values(value):
    with pytest.raises(ValueError):
        solid_materials_from_config({
            "structure": {"material_parameter_scales": {"21": {"density": value}}}
        })
