from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
forbidden_dirs={'runs','outputs','build','dist'}
ignored_roots={'.venv'}
ignored_generated_dirs={'__pycache__','.pytest_cache'}
allowed_binary_files={
    'inputs/comsol_reference/magnetostatic_converged_55iter.vtu',
    'inputs/native_reference/blocked_magnetostatic_official_mesh/magnetostatic_solution.vtu',
    'inputs/native_reference/blocked_magnetostatic_native_hybrid_c48/magnetostatic_solution.vtu',
}
forbidden_ext={'.npz','.vtu','.vtk','.png','.jpg','.jpeg','.parquet','.h5','.hdf5','.pyc','.pyo'}
forbidden_reference_dirs={ROOT/'inputs/comsol_reference/stage29_nra',ROOT/'inputs/comsol_reference/stage32_figure8',ROOT/'inputs/comsol_reference/req5'}
def test_clean_source_package():
    bad=[]
    for p in ROOT.rglob('*'):
        rel=p.relative_to(ROOT)
        if rel.parts and rel.parts[0] in ignored_roots: continue
        if any(part in ignored_generated_dirs or part.endswith('.egg-info') for part in rel.parts): continue
        if p.is_dir() and p.name in forbidden_dirs:
            bad.append(str(rel))
        if p.is_file() and p.suffix.lower() in forbidden_ext and rel.as_posix() not in allowed_binary_files: bad.append(str(rel))
    for p in forbidden_reference_dirs:
        if p.exists(): bad.append(str(p.relative_to(ROOT)))
    assert not bad,bad
