# Native 3-D vented enclosure FEM–shell–BEM

This is an independent route for the vented loudspeaker enclosure benchmark. It preserves the native quadratic acoustic tetrahedra, pyramids and hexahedra and their curved geometry. Both the solid and shell interfaces contribute where the COMSOL model selects both couplings. The existing axisymmetric and FR10 routes remain separate.

**The full model has not passed a uniform 3% COMSOL reproduction gate.** See [the saved evidence and remaining work](benchmarks/STATUS.md). Prescribed-boundary acoustic controls and independent full-model solves are reported separately.

## Run from this repository

Use Python 3.11 or newer on Linux/WSL, from the repository root. The current file locking and runtime loader use Linux interfaces:

```bash
python -m pip install -r vented3d/requirements.txt
python vented3d/run.py inspect
python vented3d/run.py prepare --out /path/to/local/vented-run
```

The included inputs contain the base mesh, native geometry/DOF topology, material settings and electrical impedance interpolation. There are 80,889 native acoustic pressure DOFs and 46,695 tet10, 2,169 pyr14 and 200 hex27 acoustic elements. Saved COMSOL response vectors, solved MPH files and preassembled matrices are not required for an independent solve.

The default core backend is oneMKL PARDISO. Use an installed compatible runtime:

```bash
python vented3d/run.py test --mkl-rt /path/to/libmkl_rt.so.3
python vented3d/run.py solve --freqs 40 31.5 315 3550 \
  --mkl-rt /path/to/libmkl_rt.so.3 --threads 8 \
  --out /path/to/local/vented-run --resume
```

Alternatively set `VENTED_MKL_RT` in the environment. The CLI adds the runtime's directory to the loader path before importing numerical modules. A separately installed `py-mkl-pardiso==0.0.6` wheel can supply the original single-thread fallback; editable installation with the optional dependency is `python -m pip install -e './vented3d[mkl]'`. No solver runtime is downloaded automatically. `--backend direct` selects SciPy LU and can require more memory for the full model.

The recorded 40 Hz full solve used about 15.8 GiB peak process memory. Serialize full solves. The 8-thread speed measurements for acoustic subproblems do not establish the same speedup for the entire FEM–BEM chain.

`--resume` checks source, input and configuration hashes and skips completed frequencies. All generated output goes to `--out`; the default is `runs/vented3d/`. Existing successful checkpoints from different source/input hashes are not overwritten. COMSOL references are used only by offline validation, never as runtime corrections or imposed boundaries in this CLI.

## What changed

The `feature/vented3d-native-shell-interfaces` branch also provides an experimental native shell route. It replaces the three remaining shell/interface mechanisms with published MITC6a/MITC9 tensor tying on curved tri6/quad9 geometry, independent fold directors with fixed-angle constraints, and solid–shell coupling selected by the actual connection branch. Use `--config vented3d/configs/native_shell_mitc.json` with the `solve` command. Source settings such as `UseMITCInterp`, the fold angle and the active shared-boundary connection type are checked against `live_settings.csv`; unsupported settings fail explicitly. See [the implementation and verification notes](NATIVE_SHELL_WORK.md). The four-point acoustic-repair table in benchmarks/STATUS.md remains the earlier baseline; native-shell results are reported separately there.

- Acoustic interpolation and geometry remain native tet10/pyr14/hex27. Splitting native mixed elements into straight P2 tetrahedra changed the high-frequency response substantially.
- Each selected acoustic–solid and acoustic–shell coupling is assembled, including overlapping selections. Choosing only the shell at these locations omitted a contribution present in the reference model.
- Curved interfaces use the same coupling matrix and its transpose for pressure work and structural-motion loading. The aperture uses a work-adjoint mortar into the existing P0 BEM.
- The native pressure topology preserves the two sides of internal shells. An ownership audit found no merging of distinct native pressure DOFs in the earlier Python mesh.

The prescribed-reference-boundary acoustic controls reached about **0.095% and 0.115% interior complex-pressure error at 3550 Hz** in domains 17 and 25. Those figures do not describe independent whole-model accuracy.

The corrected native-shell route has independently completed 315/3550 Hz regression. At 3550 Hz, coil velocity error is **0.876%** (previously 20.694%), +X/+Z pressure errors are **1.730% / 0.962%** (23.398% / 26.831%), and native pressure-field L2 error is **2.255%** (34.890%). See the separate native-shell table and provenance in [benchmarks/STATUS.md](benchmarks/STATUS.md). Both tested frequencies pass the recorded scalar and field L2 metrics under the user’s current 3% accuracy target. A complete repaired 46-point sweep remains unverified. See [the policy](benchmarks/acceptance_policy.json) and [offline reassessment](benchmarks/native_shell_acceptance_3percent.json).

## Remaining approximations

The default `native_mixed.json` configuration retains straight converted structural tetrahedra, faceted Reissner–Mindlin shells and nodal-spin ties. The experimental `native_shell_mitc.json` replaces these with curved native solids, published MITC shells, native fold directors and shared-boundary displacement connections. Its published MITC variants and connection enforcement have not been certified identical to COMSOL internals. Both routes use faceted P0 BEM with CHIEF stabilization. Modal matching and mesh convergence remain open. See [benchmarks/STATUS.md](benchmarks/STATUS.md) for measured independent regression errors.

The bundled `vented1` implementation is the reviewed local solver snapshot used by this route. `src/vented3d/model.py` attaches the repaired acoustic space and interfaces. No original axisymmetric production configuration is replaced.

## Inputs, exports and evidence

- [inputs/vented_enclosure/DATASET.json](inputs/vented_enclosure/DATASET.json) records file hashes and coordinate units. Native coordinates use metres; base `vertices.csv` uses millimetres.
- [benchmarks/](benchmarks/) contains compact numeric evidence. It excludes raw field archives, solved MPH files, large matrix caches and solver binaries.
- [comsol_exports/](comsol_exports/) contains optional read-only Java exports. They require a local solved model and an appropriate COMSOL installation. They do not run studies or save changes to the model.

For installed wheels, supply `--input-dir` and `--config` pointing to a source checkout; the documented default data layout is provided by the checkout/editable installation.
