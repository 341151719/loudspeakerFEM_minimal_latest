# Optional offline exports

These Java sources use the COMSOL 6.3 API. Compile with the local COMSOL compiler. Both accept two Java arguments: the existing solved MPH path and a new output directory.

- `ExportNativeGeometry`: node coordinates, element topology, reference coordinates and element DOF ownership; these are mesh/physical-model inputs.
- `ExportStoredVectors`: stored frequency solution vectors at indices 5, 7, 25 and 46, for offline comparison only. Do not place these vectors in solve inputs or commit the raw outputs.

The sources call neither a study run nor a model save. COMSOL batch argument forwarding depends on the installed launcher; invoking the compiled class from the COMSOL Java environment with the two arguments is also supported. The original benchmark's frequency indices correspond to 31.5, 40, 315 and 3550 Hz.
