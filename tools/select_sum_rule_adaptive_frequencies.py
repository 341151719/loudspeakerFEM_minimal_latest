#!/usr/bin/env python3
"""Select quarter points in intervals dominating a nested trapezoid error."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_csv")
    parser.add_argument("output_csv")
    parser.add_argument("--coverage", type=float, default=0.99)
    args = parser.parse_args()
    data = pd.read_csv(args.audit_csv).sort_values("omega_rad_s").reset_index(drop=True)
    if len(data) < 5 or len(data) % 2 != 1:
        raise ValueError("audit grid must have an odd number of nested logarithmic points")
    x = data["omega_rad_s"].to_numpy(float)
    y = data["Zmot_real_ohm"].to_numpy(float)
    candidates = []
    for i in range(0, len(data) - 2, 2):
        coarse = 0.5 * (x[i + 2] - x[i]) * (y[i] + y[i + 2])
        fine = 0.5 * (x[i + 1] - x[i]) * (y[i] + y[i + 1])
        fine += 0.5 * (x[i + 2] - x[i + 1]) * (y[i + 1] + y[i + 2])
        candidates.append({
            "left_Hz": float(data.loc[i, "freq_Hz"]),
            "mid_Hz": float(data.loc[i + 1, "freq_Hz"]),
            "right_Hz": float(data.loc[i + 2, "freq_Hz"]),
            "absolute_error_indicator": float(abs(fine - coarse) * 2.0 / np.pi),
        })
    ranked = sorted(candidates, key=lambda row: row["absolute_error_indicator"], reverse=True)
    total = sum(row["absolute_error_indicator"] for row in ranked)
    selected = []
    accumulated = 0.0
    for row in ranked:
        selected.append(row)
        accumulated += row["absolute_error_indicator"]
        if accumulated >= float(args.coverage) * total:
            break
    frequencies = sorted({
        value
        for row in selected
        for value in (
            float(np.sqrt(row["left_Hz"] * row["mid_Hz"])),
            float(np.sqrt(row["mid_Hz"] * row["right_Hz"])),
        )
    })
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"freq_Hz": frequencies}).to_csv(output, index=False)
    metadata = {
        "source": str(Path(args.audit_csv)),
        "coverage_target": float(args.coverage),
        "selected_parent_intervals": len(selected),
        "new_frequencies": len(frequencies),
        "covered_absolute_error_fraction": accumulated / total,
        "total_absolute_error_indicator_ohm_per_s": total,
        "selected_intervals": selected,
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({key: metadata[key] for key in metadata if key != "selected_intervals"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
