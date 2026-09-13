import numpy as np

from loudspeaker_axisym_fem.exterior_field import (
    directivity_index_halfspace,
    halfspace_power_from_directivity,
)


def test_uniform_halfspace_power_and_directivity_numpy2():
    theta = np.linspace(0.0, np.pi / 2.0, 1001)
    pressure = np.full(theta.shape, 2.0 + 0j)
    rho0, c0, radius = 1.2, 340.0, 3.0
    expected = 0.5 * 4.0 / (rho0 * c0) * 2.0 * np.pi * radius**2
    assert np.isclose(
        halfspace_power_from_directivity(pressure, theta, rho0, c0, radius),
        expected,
        rtol=1e-6,
    )
    assert abs(directivity_index_halfspace(pressure, theta)) < 1e-12
