import numpy as np
from loudspeaker_axisym_fem.narrow_region_acoustics import equivalent_narrow_region_coefficients, comsol_air_properties

def test_comsol_air_properties_at_model_state():
    p=comsol_air_properties()
    assert abs(p.density_kg_m3-1.2043175745358388)<1e-12
    assert abs(p.dynamic_viscosity_Pa_s-1.8139686307339444e-5)<1e-16
    assert abs(p.prandtl-0.7077735901674092)<1e-12

def test_native_nra_passive_and_high_frequency_limit():
    for h in (0.4e-3,0.2e-3):
        for f in (1,50,600,8000,1e7):
            c=equivalent_narrow_region_coefficients(f,h)
            assert c.passive
            assert c.stiffness_factor.imag >= -1e-12
            assert c.mass_factor.imag <= 1e-12
        c=equivalent_narrow_region_coefficients(1e7,h)
        assert abs(c.stiffness_factor-1)<0.05
        assert abs(c.mass_factor-1)<0.05

def test_narrower_slit_has_stronger_loss_at_600Hz():
    wide=equivalent_narrow_region_coefficients(600,0.4e-3)
    narrow=equivalent_narrow_region_coefficients(600,0.2e-3)
    assert narrow.stiffness_factor.imag > wide.stiffness_factor.imag
    assert -narrow.mass_factor.imag > -wide.mass_factor.imag


if __name__ == "__main__":
    test_comsol_air_properties_at_model_state()
    test_native_nra_passive_and_high_frequency_limit()
    test_narrower_slit_has_stronger_loss_at_600Hz()
    print("NATIVE_NRA_TEST: PASS")
