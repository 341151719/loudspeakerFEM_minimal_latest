# Agent entrypoint: loudspeakerFEM

This repository is the independent 2-D axisymmetric, frequency-domain loudspeaker FEM project. Treat the checked-in files and current branch as authoritative; do not assume another checkout is synchronized.

## Read first

1. [`docs/AGENT_MAP_CN.md`](docs/AGENT_MAP_CN.md) for task-to-file routing.
2. [`README_CN.md`](README_CN.md) for the production route, physical contracts, evidence limits, and required validation matrix.
3. [`docs/PROJECT_RELATIONSHIP_CN.md`](docs/PROJECT_RELATIONSHIP_CN.md) before working across the frequency and transient projects.
4. [`docs/IMPROVEMENT_ROADMAP_CN.md`](docs/IMPROVEMENT_ROADMAP_CN.md) when choosing project-wide improvements.
5. [`docs/LOW_MID_INTERFACE_AUDIT_CN.md`](docs/LOW_MID_INTERFACE_AUDIT_CN.md) and [`docs/TIME_SNAPSHOT_AUDIT_CN.md`](docs/TIME_SNAPSHOT_AUDIT_CN.md) for the completed low/mid geometry and cross-repository audits.

## Main route

- Production configuration: `configs/best_model.json`; frequency-dependent routing is part of the production contract.
- CLI: `cli.py`; primary coupled frequency solver: `best_model/coupled_solver.py`.
- Run `python cli.py plan --freqs 50,6300,12000` to inspect routing and required inputs before an expensive solve. Successful `solve` and `sweep` runs write `run_manifest.json` with effective config and input hashes.
- Shared axisymmetric FEM implementations: `src/loudspeaker_axisym_fem/`.
- The separate 3-D FR10 route lives in `fr10_full360_cyclic/`; consult `docs/FR10_FULL360_STATUS_CN.md` before changing it.

## Project boundary

Use this repository for harmonic response, frequency sweeps, native blocked-coil impedance, modal work, and enclosure FEM. Use [`loudspeakerTimeFEM`](https://github.com/341151719/loudspeakerTimeFEM_minimal_latest) for transient waveforms and large-signal time-domain behavior. The time-domain repository vendors a frequency-source snapshot under `inputs/frequency_mainline/`; that copy is neither a submodule nor automatically synchronized. A change to shared code must be reviewed and, if needed, applied and committed independently in each repository.

## Operating constraints

- Keep COMSOL as an offline reference/benchmark; do not introduce COMSOL-result runtime correction into the Python production route.
- Keep exploratory settings in a copied or dedicated diagnostic config. Do not overwrite `configs/best_model.json` for experiments.
- Write generated results to a new `runs/` location or outside the repository, never to `inputs/`.
- Treat historical metrics in `README_CN.md` as reported evidence, not as a fresh rerun. Follow its validation matrix for solver or physics changes.
