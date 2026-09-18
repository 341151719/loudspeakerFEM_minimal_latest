from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import base_p2_local_solver as base
import cyclic_full360_solver as cyclic


DEFAULT_FREQUENCIES_HZ = (
    50.0,
    63.0,
    80.0,
    90.0,
    100.0,
    125.0,
    160.0,
    200.0,
    250.0,
    315.0,
    400.0,
    500.0,
    630.0,
    800.0,
    1000.0,
    1250.0,
    1600.0,
    2000.0,
)


def _complex_pair(value):
    return complex(float(value[0]), float(value[1]))


def _row_from_baseline(row):
    electrical = row["electrical"]
    mechanical = row["mechanical"]
    acoustic = row["acoustic"]
    frequency = float(row["frequency_Hz"])
    z_total = _complex_pair(electrical["Z_total_ohm"])
    z_motional = _complex_pair(electrical["Z_motional_ohm"])
    current = _complex_pair(electrical["current_A_peak"])
    coil = _complex_pair(mechanical["coil_displacement_peak_m"])
    return {
        "frequency_Hz": frequency,
        "front_SPL_1m_dB_at_1V_peak": float(acoustic["front_SPL_1m_axis_dB"]),
        "rear_SPL_1m_dB_at_1V_peak": float(acoustic["rear_SPL_1m_axis_dB"]),
        "Z_total_ohm": [z_total.real, z_total.imag],
        "Z_motional_ohm": [z_motional.real, z_motional.imag],
        "current_A_peak": [current.real, current.imag],
        "coil_displacement_m_peak": [coil.real, coil.imag],
        "solver": row.get("cyclic_solver"),
        "source": "reused validated final_baseline result",
    }


def _solve_row(cfg, model, front, rear, coupling, frequency, suspension_scale):
    displacement, p_front, p_rear, solver = cyclic.solve_phase(
        cfg, model, front, rear, coupling, frequency, suspension_scale, 0
    )
    omega = 2.0 * math.pi * frequency
    bl = float(cfg["electrical"]["Bl_Tm"])
    coil_per_amp = complex(model["gcoil"] @ displacement)
    z_motional = 1j * omega * bl * coil_per_amp
    z_blocked = complex(
        float(cfg["electrical"]["Rdc_ohm"]),
        omega * float(cfg["electrical"]["Le_H"]),
    )
    z_total = z_blocked + z_motional
    current = 1.0 / z_total
    coil = coil_per_amp * current
    p_front = p_front * current
    p_rear = p_rear * current
    radius = float(front["outer_radius_m"])
    front_axis = int(
        np.argmin(np.linalg.norm(front["p"] - np.array([0.0, 0.0, radius]), axis=1))
    )
    rear_axis = int(
        np.argmin(np.linalg.norm(rear["p"] - np.array([0.0, 0.0, -radius]), axis=1))
    )
    wave_number = omega / float(cfg["air"]["c_m_s"])
    propagation = radius * np.exp(-1j * wave_number * (1.0 - radius))
    p_front_1m = complex(p_front[front_axis]) * propagation
    p_rear_1m = complex(p_rear[rear_axis]) * propagation
    reference = float(cfg["air"]["p_ref_Pa"])

    def spl(value):
        return 20.0 * math.log10(max(abs(value) / math.sqrt(2.0) / reference, 1e-300))

    return {
        "frequency_Hz": float(frequency),
        "front_SPL_1m_dB_at_1V_peak": spl(p_front_1m),
        "rear_SPL_1m_dB_at_1V_peak": spl(p_rear_1m),
        "front_pressure_1m_Pa_peak": [p_front_1m.real, p_front_1m.imag],
        "rear_pressure_1m_Pa_peak": [p_rear_1m.real, p_rear_1m.imag],
        "Z_total_ohm": [z_total.real, z_total.imag],
        "Z_motional_ohm": [z_motional.real, z_motional.imag],
        "current_A_peak": [current.real, current.imag],
        "coil_displacement_m_peak": [coil.real, coil.imag],
        "solver": solver,
        "source": "new full-360 k=0 solve",
    }


def _add_normalizations(rows):
    offset_2p83 = 20.0 * math.log10(2.83 / (1.0 / math.sqrt(2.0)))
    for row in rows:
        row["front_SPL_1m_dB_at_2p83Vrms"] = (
            row["front_SPL_1m_dB_at_1V_peak"] + offset_2p83
        )
        row["rear_SPL_1m_dB_at_2p83Vrms"] = (
            row["rear_SPL_1m_dB_at_1V_peak"] + offset_2p83
        )


def _write_outputs(output, rows, suspension_scale):
    rows = sorted(rows, key=lambda item: item["frequency_Hz"])
    _add_normalizations(rows)
    payload = {
        "method": "FR10 full-360 cyclic k=0 FEM; spherical outgoing-wave extrapolation from the 0.3 m boundary to 1 m",
        "drive": "1 V peak (0.70710678 V RMS); 2.83 V RMS columns are linear normalization",
        "validity": "validated/issued band 50-2000 Hz; first-order Sommerfeld boundary, not PML",
        "suspension_scale": float(suspension_scale),
        "frequencies": rows,
    }
    json_path = output / "frequency_response_1m.json"
    json_path.write_text(json.dumps(payload, indent=2))
    csv_path = output / "frequency_response_1m.csv"
    fields = (
        "frequency_Hz",
        "front_SPL_1m_dB_at_1V_peak",
        "rear_SPL_1m_dB_at_1V_peak",
        "front_SPL_1m_dB_at_2p83Vrms",
        "rear_SPL_1m_dB_at_2p83Vrms",
        "Z_total_magnitude_ohm",
        "Z_total_phase_deg",
        "coil_displacement_m_peak_magnitude",
        "source",
    )
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            impedance = _complex_pair(row["Z_total_ohm"])
            coil = _complex_pair(row["coil_displacement_m_peak"])
            writer.writerow(
                {
                    **{key: row[key] for key in fields[:5]},
                    "Z_total_magnitude_ohm": abs(impedance),
                    "Z_total_phase_deg": math.degrees(math.atan2(impedance.imag, impedance.real)),
                    "coil_displacement_m_peak_magnitude": abs(coil),
                    "source": row["source"],
                }
            )

    frequency = np.asarray([row["frequency_Hz"] for row in rows])
    front = np.asarray([row["front_SPL_1m_dB_at_1V_peak"] for row in rows])
    rear = np.asarray([row["rear_SPL_1m_dB_at_1V_peak"] for row in rows])
    impedance = np.asarray([abs(_complex_pair(row["Z_total_ohm"])) for row in rows])
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 7.5), sharex=True)
    axes[0].semilogx(frequency, front, "o-", label="front axis")
    axes[0].semilogx(frequency, rear, "s-", label="rear axis")
    axes[0].set_ylabel("SPL at 1 m / dB")
    axes[0].set_title("FR10 full-360 1 m frequency response — 1 V peak drive")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()
    axes[1].semilogx(frequency, impedance, "o-", color="tab:purple")
    axes[1].set_xlabel("frequency / Hz")
    axes[1].set_ylabel("|Z total| / ohm")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].set_xlim(float(frequency.min()), float(frequency.max()))
    fig.tight_layout()
    png_path = output / "frequency_response_1m.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    front_2p83 = np.asarray([row["front_SPL_1m_dB_at_2p83Vrms"] for row in rows])
    rear_2p83 = np.asarray([row["rear_SPL_1m_dB_at_2p83Vrms"] for row in rows])
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.semilogx(frequency, front_2p83, "o-", label="front axis")
    ax.semilogx(frequency, rear_2p83, "s-", label="rear axis")
    ax.set_xlabel("frequency / Hz")
    ax.set_ylabel("SPL at 1 m / dB")
    ax.set_title("FR10 full-360 1 m frequency response — normalized to 2.83 V RMS")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    ax.set_xlim(float(frequency.min()), float(frequency.max()))
    fig.tight_layout()
    normalized_png_path = output / "frequency_response_1m_2p83Vrms.png"
    fig.savefig(normalized_png_path, dpi=180)
    plt.close(fig)
    return json_path, csv_path, png_path, normalized_png_path


def run_frequency_response(
    frequencies=DEFAULT_FREQUENCIES_HZ,
    output=None,
    reuse_summary=None,
    suspension_scale=None,
):
    cfg = cyclic.load_cfg()
    suspension_scale = float(
        suspension_scale or cfg["calibration"]["p2_local_asb_suspension_scale"]
    )
    output = Path(output or cyclic.default_output_root() / "frequency_response_1m")
    output.mkdir(parents=True, exist_ok=True)
    requested = sorted(set(float(value) for value in frequencies))
    rows_by_frequency = {}
    checkpoint = output / "frequency_response_1m.json"
    if checkpoint.exists():
        for row in json.loads(checkpoint.read_text()).get("frequencies", []):
            rows_by_frequency[float(row["frequency_Hz"])] = row
    if reuse_summary and Path(reuse_summary).exists():
        baseline = json.loads(Path(reuse_summary).read_text())
        for row in baseline.get("frequencies", []):
            frequency = float(row["frequency_Hz"])
            if frequency in requested:
                rows_by_frequency[frequency] = _row_from_baseline(row)

    missing = [value for value in requested if value not in rows_by_frequency]
    if missing:
        model = cyclic.build_sector_model(cfg)
        front, rear = base.build_acoustic_domains(cfg)
        coupling, _ = base.build_local_G(model, front, cfg)
        for index, frequency in enumerate(missing, 1):
            print(f"[response] {frequency:g} Hz ({index}/{len(missing)})", flush=True)
            rows_by_frequency[frequency] = _solve_row(
                cfg, model, front, rear, coupling, frequency, suspension_scale
            )
            current = [rows_by_frequency[value] for value in requested if value in rows_by_frequency]
            _write_outputs(output, current, suspension_scale)
    rows = [rows_by_frequency[value] for value in requested]
    paths = _write_outputs(output, rows, suspension_scale)
    return rows, paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FR10 full-360 1 m frequency response")
    parser.add_argument("--frequencies", type=float, nargs="+", default=DEFAULT_FREQUENCIES_HZ)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reuse-summary", type=Path)
    parser.add_argument("--scale", type=float)
    args = parser.parse_args()
    run_frequency_response(args.frequencies, args.output, args.reuse_summary, args.scale)
