#!/usr/bin/env python3
"""Compare the 2x2 structure/acoustic mesh audit used by the sum-rule study."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = {
    "base": "runs/sum_rules_mesh_base",
    "acoustic_refined": "runs/sum_rules_mesh_acoustic_refined",
    "structure_refined": "runs/sum_rules_mesh_structure_refined",
    "combined_finest": "runs/sum_rules_highest_accuracy_pilot",
}
METRICS = ("Zmot_real_ohm", "Zmot_imag_ohm", "Rac_far_ohm", "Rinternal_far_ohm")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="runs/sum_rules_mesh_convergence")
    parser.add_argument("--run", action="append", default=[], help="label=run_directory")
    args = parser.parse_args()
    run_specs = DEFAULT_RUNS if not args.run else dict(item.split("=", 1) for item in args.run)
    out = (ROOT / args.outdir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    frames: dict[str, pd.DataFrame] = {}
    summaries: dict[str, dict] = {}
    for label, directory in run_specs.items():
        path = (ROOT / directory).resolve()
        frames[label] = pd.read_csv(path / "sum_rule_frequency_audit.csv").sort_values("freq_Hz")
        summaries[label] = json.loads((path / "sum_rule_summary.json").read_text(encoding="utf-8"))
    for frame in frames.values():
        frame["frequency_key"] = frame["freq_Hz"].round(8)
    common_keys = set.intersection(*(set(frame["frequency_key"]) for frame in frames.values()))
    if len(common_keys) < 2:
        raise ValueError("fewer than two common audit frequencies across variants")
    reference = frames["combined_finest"]
    reference = reference[reference["frequency_key"].isin(common_keys)].copy()

    rows: list[dict[str, float | str]] = []
    aggregate: dict[str, dict[str, float]] = {}
    for label, frame in frames.items():
        frame = frame[frame["frequency_key"].isin(common_keys)].copy()
        joined = frame.merge(reference, on="frequency_key", suffixes=("", "_reference"), validate="one_to_one")
        if len(joined) != len(reference):
            raise ValueError(f"{label}: only {len(joined)}/{len(reference)} reference frequencies matched")
        for _, row in joined.iterrows():
            item: dict[str, float | str] = {"variant": label, "freq_Hz": float(row["freq_Hz"])}
            z = complex(row["Zmot_real_ohm"], row["Zmot_imag_ohm"])
            zref = complex(row["Zmot_real_ohm_reference"], row["Zmot_imag_ohm_reference"])
            item["Zmot_complex_relative_error"] = abs(z - zref) / max(abs(zref), 1e-300)
            for metric in ("Rac_far_ohm", "Rinternal_far_ohm"):
                item[f"{metric}_relative_error"] = abs(row[metric] - row[f"{metric}_reference"]) / max(abs(row[f"{metric}_reference"]), 1e-300)
            rows.append(item)
        errors = np.asarray([r["Zmot_complex_relative_error"] for r in rows if r["variant"] == label], float)
        rac_errors = np.asarray([r["Rac_far_ohm_relative_error"] for r in rows if r["variant"] == label], float)
        aggregate[label] = {
            "Zmot_complex_rms_relative_error": float(np.sqrt(np.mean(errors**2))),
            "Zmot_complex_max_relative_error": float(np.max(errors)),
            "Rac_far_rms_relative_error": float(np.sqrt(np.mean(rac_errors**2))),
            "Rac_far_max_relative_error": float(np.max(rac_errors)),
            "A_inf_ohm_per_s": float(summaries[label]["endpoints"]["A_inf_matrix_ohm_per_s"]),
            "A0_structure_ohm_s": float(summaries[label]["endpoints"]["A0_structure_only_ohm_s"]),
        }

    comparison = pd.DataFrame(rows)
    comparison.to_csv(out / "frequency_comparison.csv", index=False)
    result = {"reference": "combined_finest", "runs": run_specs, "aggregate": aggregate}
    (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for label in frames:
        subset = comparison[comparison.variant == label]
        axes[0].semilogx(subset.freq_Hz, 100 * subset.Zmot_complex_relative_error, marker="o", label=label)
        axes[1].semilogx(subset.freq_Hz, 100 * subset.Rac_far_ohm_relative_error, marker="o", label=label)
    axes[0].set_ylabel("Zmot complex error [%]")
    axes[1].set_ylabel("Far-field Rac error [%]")
    axes[1].set_xlabel("Frequency [Hz]")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "mesh_convergence.png", dpi=180)
    plt.close(fig)

    lines = ["# Sum rules FEM 离散收敛对照", "", "参考解为结构 L2 与声学 refined1 的组合最细网格。", ""]
    lines += ["| 变体 | Zmot RMS | Zmot max | Rac RMS | Rac max |", "|---|---:|---:|---:|---:|"]
    for label, values in aggregate.items():
        lines.append(
            f"| {label} | {100*values['Zmot_complex_rms_relative_error']:.3g}% | "
            f"{100*values['Zmot_complex_max_relative_error']:.3g}% | "
            f"{100*values['Rac_far_rms_relative_error']:.3g}% | {100*values['Rac_far_max_relative_error']:.3g}% |"
        )
    lines += ["", "逐频详细结果见 `frequency_comparison.csv`。", ""]
    (out / "VALIDATION_REPORT_CN.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
