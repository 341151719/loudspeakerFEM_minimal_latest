# Curated frequency-domain benchmark snapshots

This directory contains a small, provenance-labelled subset of the historical evidence from
`loudspeakerFEM_current_20260717` (snapshot: **2026-07-19**). It is intended for review,
regression checks, and follow-up research; it is not a runtime input directory.

The main entry points are:

- `frequency_response/v3_126pt/`: 126 native Python sweep rows for 1–8 kHz plus the high-
  frequency triangle-split A/B outputs;
- `frequency_response/stage35_*.json`: aggregate Stage34/Stage35 results for 1–15 kHz;
- `blocked_impedance/`: native MQS mesh-convergence and layered metrics;
- `acoustic_structure/`, `eddy_current/`, `modal/`, and `pml/`: compact cross-checks for
  ASB power, eddy-current loss, eigenmodes, and PML geometry.

The V3 126-point comparison reports 1.2192% complex NRMSE, 0.1142 dB amplitude RMSE, and
0.7689° phase RMSE for the full coupled field. The 4–8 kHz complex NRMSE is 3.4761% and the
6.3–8 kHz value is 2.6439%. COMSOL is an independent reference in these comparisons; the
native Python runtime does not consume the published benchmark files.

Do not mix the blocked metrics: 0.4224% is the current native-field mesh-convergence result,
while 0.0272% is an offline COMSOL-identified embedded surrogate. The latter is not an
independent raw-field prediction. See [`README_CN.md`](README_CN.md) for the detailed scope.

Large point clouds, solved MPH files, logs, plots, checkpoints, and internal analysis paths
were deliberately excluded. The machine-readable catalog is [`manifest.json`](manifest.json).
