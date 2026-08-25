from __future__ import annotations

from dataclasses import dataclass
import math


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive(value: float, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return result


def _nonnegative(value: float, name: str) -> float:
    result = _finite(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return result


def _omega(frequency_Hz: float) -> float:
    frequency = _finite(frequency_Hz, "frequency_Hz")
    if frequency < 0.0:
        raise ValueError("frequency_Hz must be >= 0")
    # Keep the zero-frequency limit numerically usable without pretending that
    # a finite DC acoustic impedance exists.
    return 2.0 * math.pi * max(frequency, 1.0e-30)


@dataclass(frozen=True)
class AirProperties:
    """Small-signal air properties used by the lumped references."""

    rho0: float = 1.21
    c0: float = 343.0

    def __post_init__(self) -> None:
        _positive(self.rho0, "rho0")
        _positive(self.c0, "c0")


@dataclass(frozen=True, init=False)
class ClosedBox:
    """A sealed cavity compliance with an optional *leak* branch.

    The old ``loss_resistance_Pa_s_m3`` spelling is accepted as a compatibility
    alias.  It is not a wall thermal-loss parameter: a finite value adds a
    parallel conductance ``1/R_leak`` and therefore represents pressure/volume
    flow through a leakage path.  A strictly sealed box leaves it as ``None``.
    """

    volume_m3: float
    leak_resistance_Pa_s_m3: float | None = None

    def __init__(
        self,
        volume_m3: float,
        leak_resistance_Pa_s_m3: float | None = None,
        *,
        loss_resistance_Pa_s_m3: float | None = None,
    ) -> None:
        if leak_resistance_Pa_s_m3 is not None and loss_resistance_Pa_s_m3 is not None:
            raise TypeError(
                "specify leak_resistance_Pa_s_m3 or its legacy alias, not both"
            )
        resistance = (
            leak_resistance_Pa_s_m3
            if leak_resistance_Pa_s_m3 is not None
            else loss_resistance_Pa_s_m3
        )
        volume = _positive(volume_m3, "volume_m3")
        if resistance is not None:
            resistance = _positive(resistance, "leak_resistance_Pa_s_m3")
        object.__setattr__(self, "volume_m3", volume)
        object.__setattr__(self, "leak_resistance_Pa_s_m3", resistance)

    @property
    def loss_resistance_Pa_s_m3(self) -> float | None:
        """Deprecated compatibility alias; the branch is a physical leak."""

        return self.leak_resistance_Pa_s_m3

    @property
    def strictly_sealed(self) -> bool:
        return self.leak_resistance_Pa_s_m3 is None

    def compliance(self, air: AirProperties = AirProperties()) -> float:
        return self.volume_m3 / (air.rho0 * air.c0**2)

    def input_admittance(self, f: float, air: AirProperties = AirProperties()) -> complex:
        """Return ``U/p`` for the compliance plus optional leakage branch."""

        omega = _omega(f)
        admittance = 1j * omega * self.compliance(air)
        if self.leak_resistance_Pa_s_m3 is not None:
            admittance += 1.0 / self.leak_resistance_Pa_s_m3
        return admittance

    def input_impedance(self, f: float, air: AirProperties = AirProperties()) -> complex:
        return 1.0 / self.input_admittance(f, air)


@dataclass(frozen=True)
class LeakPath:
    """A low-order leakage path, with resistance in ``Pa s/m^3``."""

    radius_m: float
    length_m: float
    resistance_Pa_s_m3: float | None = None

    def __post_init__(self) -> None:
        _positive(self.radius_m, "radius_m")
        _positive(self.length_m, "length_m")
        if self.resistance_Pa_s_m3 is not None:
            _positive(self.resistance_Pa_s_m3, "resistance_Pa_s_m3")

    def impedance(self, f: float, air: AirProperties = AirProperties()) -> complex:
        area = math.pi * self.radius_m**2
        effective_length = self.length_m + 1.7 * self.radius_m
        mass = air.rho0 * effective_length / area
        resistance = self.resistance_Pa_s_m3
        if resistance is None:
            # Low-Re engineering placeholder.  It is a leakage reference, not
            # a replacement for a measured or resolved enclosure loss model.
            eta_air = 1.84e-5
            resistance = 8.0 * eta_air * self.length_m / (math.pi * self.radius_m**4)
        return complex(resistance, _omega(f) * mass)


@dataclass(frozen=True)
class Port:
    """Uniform circular-port lumped reference.

    ``end_correction_radii`` belongs only to this lumped reference.  If an
    explicit FEM air domain resolves the port opening and its radiation, the
    FEM path must use no copy of this correction or its radiation mass.
    """

    radius_m: float
    length_m: float
    resistance_Pa_s_m3: float = 0.0
    end_correction_radii: float = 1.46
    terminal_condition: str = "open-open"

    def __post_init__(self) -> None:
        _positive(self.radius_m, "radius_m")
        _positive(self.length_m, "length_m")
        _nonnegative(self.resistance_Pa_s_m3, "resistance_Pa_s_m3")
        _nonnegative(self.end_correction_radii, "end_correction_radii")
        if self.terminal_condition not in {
            "open-open",
            "closed-open",
            "open-closed",
            "closed-closed",
        }:
            raise ValueError(f"unknown terminal condition: {self.terminal_condition}")

    def area(self) -> float:
        return math.pi * self.radius_m**2

    def effective_length(self) -> float:
        """Length for the lumped Helmholtz reference, including end correction."""

        return self.length_m + self.end_correction_radii * self.radius_m

    def acoustic_mass(self, air: AirProperties = AirProperties()) -> float:
        return air.rho0 * self.effective_length() / self.area()

    def impedance(self, f: float, air: AirProperties = AirProperties()) -> complex:
        return complex(self.resistance_Pa_s_m3, _omega(f) * self.acoustic_mass(air))

    def first_pipe_resonance_Hz(
        self,
        air: AirProperties = AirProperties(),
        terminal_condition: str | bool | None = None,
        *,
        closed_open: bool | None = None,
    ) -> float:
        """Return the first uniform-pipe longitudinal mode.

        The default is ``open-open`` (half-wave), appropriate for an ordinary
        open-ended port reference.  ``closed-open`` is an explicit quarter-wave
        choice.  The legacy positional/keyword ``closed_open`` boolean remains
        supported so existing callers do not silently change meaning.
        """

        selected: str
        if isinstance(terminal_condition, bool):
            if closed_open is not None:
                raise TypeError("terminal_condition and closed_open both supplied")
            selected = "closed-open" if terminal_condition else "open-open"
        elif terminal_condition is None:
            selected = self.terminal_condition
        else:
            selected = terminal_condition
        if closed_open is not None:
            if terminal_condition is not None:
                raise TypeError("terminal_condition and closed_open both supplied")
            selected = "closed-open" if closed_open else "open-open"
        if selected not in {
            "open-open",
            "closed-open",
            "open-closed",
            "closed-closed",
        }:
            raise ValueError(f"unknown terminal condition: {selected}")
        denominator = 4.0 if selected in {"closed-open", "open-closed"} else 2.0
        return air.c0 / (denominator * self.effective_length())


@dataclass(frozen=True)
class VentedBox:
    """Parallel compliance/port network for the low-frequency reference."""

    box: ClosedBox
    port: Port
    leak: LeakPath | None = None

    def helmholtz_frequency_Hz(self, air: AirProperties = AirProperties()) -> float:
        return 1.0 / (
            2.0
            * math.pi
            * math.sqrt(self.port.acoustic_mass(air) * self.box.compliance(air))
        )

    def input_admittance(self, f: float, air: AirProperties = AirProperties()) -> complex:
        # The cavity compliance, port inertance, and optional leakage path all
        # see the same cavity pressure, hence their admittances are parallel.
        admittance = self.box.input_admittance(f, air) + 1.0 / self.port.impedance(f, air)
        if self.leak is not None:
            admittance += 1.0 / self.leak.impedance(f, air)
        return admittance

    def input_impedance(self, f: float, air: AirProperties = AirProperties()) -> complex:
        return 1.0 / self.input_admittance(f, air)

    def port_volume_velocity(
        self,
        f: float,
        box_pressure_Pa: complex,
        air: AirProperties = AirProperties(),
    ) -> complex:
        """Return flow from the box into the port for the chosen pressure sign."""

        return complex(box_pressure_Pa) / self.port.impedance(f, air)

    def port_velocity(
        self,
        f: float,
        box_pressure_Pa: complex,
        air: AirProperties = AirProperties(),
    ) -> complex:
        return self.port_volume_velocity(f, box_pressure_Pa, air) / self.port.area()


@dataclass(frozen=True)
class PassiveRadiator:
    """Rigid-piston SDOF PR; radiation loading is external to ``Mms/Rms``."""

    Sd_m2: float
    Mms_kg: float
    Cms_m_N: float
    Rms_N_s_m: float

    def __post_init__(self) -> None:
        _positive(self.Sd_m2, "Sd_m2")
        _positive(self.Mms_kg, "Mms_kg")
        _positive(self.Cms_m_N, "Cms_m_N")
        _nonnegative(self.Rms_N_s_m, "Rms_N_s_m")

    def mechanical_impedance(self, f: float) -> complex:
        omega = _omega(f)
        return complex(
            self.Rms_N_s_m,
            omega * self.Mms_kg - 1.0 / (omega * self.Cms_m_N),
        )

    def acoustic_impedance(self, f: float) -> complex:
        """Convert ``F/v`` to ``p/U`` as ``Zm / Sd^2``."""

        return self.mechanical_impedance(f) / self.Sd_m2**2

    def resonance_Hz(self) -> float:
        return 1.0 / (2.0 * math.pi * math.sqrt(self.Mms_kg * self.Cms_m_N))


@dataclass(frozen=True)
class PassiveRadiatorBox:
    """Parallel sealed-box/PR network using one coherent PR SDOF."""

    box: ClosedBox
    radiator: PassiveRadiator
    leak: LeakPath | None = None

    def input_admittance(self, f: float, air: AirProperties = AirProperties()) -> complex:
        admittance = self.box.input_admittance(f, air) + 1.0 / self.radiator.acoustic_impedance(f)
        if self.leak is not None:
            admittance += 1.0 / self.leak.impedance(f, air)
        return admittance

    def input_impedance(self, f: float, air: AirProperties = AirProperties()) -> complex:
        return 1.0 / self.input_admittance(f, air)

    def radiator_volume_velocity(self, f: float, box_pressure_Pa: complex) -> complex:
        return complex(box_pressure_Pa) / self.radiator.acoustic_impedance(f)

    def radiator_velocity(self, f: float, box_pressure_Pa: complex) -> complex:
        # F = p Sd and v = F/Zm.  This is the same conversion as U/Sd, with
        # the area appearing exactly once in the pressure-to-force step.
        return (
            complex(box_pressure_Pa)
            * self.radiator.Sd_m2
            / self.radiator.mechanical_impedance(f)
        )


@dataclass(frozen=True)
class PorousRegionSpec:
    name: str
    mode: str  # empty | lining | fullfill | custom_mask
    sigma_Pa_s_m2: float | None = None
    thickness_m: float = 0.03
    notes: str = ""

    def __post_init__(self) -> None:
        if self.mode not in {"empty", "lining", "fullfill", "custom_mask"}:
            raise ValueError(f"unknown porous region mode: {self.mode}")
        _positive(self.thickness_m, "thickness_m")
        if self.sigma_Pa_s_m2 is not None:
            _nonnegative(self.sigma_Pa_s_m2, "sigma_Pa_s_m2")


def loudspeaker_side_mechanical_impedance_from_acoustic(
    Za_Pa_s_m3: complex,
    Sd_m2: float,
) -> complex:
    """Convert acoustic ``p/U`` to mechanical ``F/v`` using ``Zm = Za Sd^2``."""

    area = _positive(Sd_m2, "Sd_m2")
    return complex(Za_Pa_s_m3) * area**2


def acoustic_impedance_from_mechanical(
    Zm_N_s_m: complex,
    Sd_m2: float,
) -> complex:
    """Convert mechanical ``F/v`` to acoustic ``p/U`` using ``Za = Zm/Sd^2``."""

    area = _positive(Sd_m2, "Sd_m2")
    return complex(Zm_N_s_m) / area**2
