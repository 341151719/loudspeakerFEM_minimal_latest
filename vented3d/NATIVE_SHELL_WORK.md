# Native shell/interface implementation work

User request (2026-10-20): implement all three missing mechanisms in Python: MITC shell interpolation, fold-line director DOFs/constraints, and finite-thickness/offset solid–shell coupling. Preserve the completed baseline and its provenance. Current branch: `feature/vented3d-native-shell-interfaces`, based on `7cb7c1d`.

Confirmed local settings in `inputs/vented_enclosure/live_settings.csv`: `shell/prop/ShellAdvancedSettings.UseMITCInterp=1`; `shell/prop/FoldlineAngle.FoldlineAngle=0.001`; pointwise elemental fold constraints; `coupling/sshc1.connectionSettings=sharedBnd`, `connectionType=Hard`, shell-thickness distance, no offset in thickness features. Shell translations/directors are quadratic: tri6 / quad9. Solid displacement is tet10 / pyr13 / hex20. Existing Python shells use faceted P2/mean shear and shared axial rotations; connector uses solid curl.

Work sequence:

1. Implement and test curved tri6/quad9 director shell strain operators and MITC tying interpolation. MITC6-a shear uses eight coefficients (two Gauss points on each edge plus two center components); membrane tying uses nine samples. MITC9 covariant 11/13 components use 2x3 tying grids, 22/23 use 3x2, 12 uses 2x2. Preserve quadratic geometry. Do not rename the old stabilization as MITC.
2. Explicit director tangency, smooth-boundary continuity, separate fold directors and linearized fixed-angle fold constraints. Director is normal displacement `a`, not axial rotation `theta`; use `a=theta cross n` only for compatible conversions/tests. Shared shell translation must stay continuous. No artificial drilling stiffness is needed with explicit tangency.
3. Physical hard connection operators `u_solid = u_mid + z*a`, with thickness/offset integration and bounded solid ownership. Validate translation and all three rigid rotations, force/moment transfer, and constraint rank. Unsupported/missing geometry must fail visibly.
4. Integrate behind a separate config/CLI mode into the isolated native acoustic route. Keep the completed acoustic-only repair baseline intact. Validate small patch, thin-limit, fold, offset and work-adjoint tests, then run 3550 Hz and a low/mid control before claiming improved COMSOL accuracy.

Primary references:

- COMSOL 6.3 shell theory: https://doc.comsol.com/6.3/doc/com.comsol.help.sme/sme_ug_theory.06.130.html
- COMSOL fold settings: https://doc.comsol.com/6.3/doc/com.comsol.help.sme/sme_ug_shell_plate.08.02.html
- COMSOL connection: https://doc.comsol.com/6.3/doc/com.comsol.help.sme/sme_ug_multiphysics.18.74.html
- Lee & Bathe 2004: https://web.mit.edu/kjb/www/Principal_Publications/Development_of_MITC_Isotropic_Triangular_Shell_Finite_Elements.pdf
- Kim & Bathe 2009: https://web.mit.edu/kjb/www/Principal_Publications/A_Triangular_Six-Node_Shell_Element.pdf
- MITC9: https://web.mit.edu/kjb/www/Principal_Publications/Towards_Improving_the_MITC9_Shell_Element.pdf

Offline diagnostics already created in the migrated project, `continuation_20261019/diagnosis_3550/`: response attribution and native parametric surface motion. At 3550 Hz current error 1.076%, velocity/current 20.365%; 74-direction radiation error 22.593%, after coil normalization 5.548%. Composite boundaries 200/202 dominate surface displacement error. Native pressure control with reference motion is ~0.1%, but independent full pressure error is 34.89%. These are diagnostics, not production gain corrections.

Other settings audited: damping reference 40 Hz, thickness and quadratic order agree. Exported `massMatrix=lumped` is under StructuralTransientBehavior and its applicability to the frequency-domain study is not established; do not switch Python mass blindly. Study geometric nonlinearity is off.
