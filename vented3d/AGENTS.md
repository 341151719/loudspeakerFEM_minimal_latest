# Vented 3-D route

Read README.md and benchmarks/STATUS.md. This folder is independent of the 2-D production route and the FR10 model.

Use `python vented3d/run.py inspect`, `prepare`, `solve`, and `test` from the repository root. Keep runs outside inputs; default runs/ is ignored. Inputs include native mesh geometry and DOF connectivity, never saved response vectors. Reference data must not enter assembly or solve.

Preserve native tet10/pyr14/hex27 acoustic interpolation and curved geometry. Preserve both selected solid and shell couplings on overlapping faces and the work-adjoint interfaces. The default native_mixed config keeps the earlier straight structural solids and faceted shell/connector. The experimental native_shell_mitc config rebuilds native curved tet10/pyr13/hex20 solids, MITC shells, fold directors and shared-boundary displacement connections. Keep the applicable implementation limits and P0 BEM limitation explicit; see NATIVE_SHELL_WORK.md.

Run relevant tests and inspect residuals after changes. Numerical success is not COMSOL reproduction acceptance. Do not describe prescribed-reference-boundary acoustic controls as independent full-model accuracy. Serialize large factorizations and reuse completed frequency checkpoints.
