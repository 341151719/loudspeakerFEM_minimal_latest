from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
forbidden_ext={'.npz','.vtu','.vtk','.png','.jpg','.jpeg','.parquet','.h5','.hdf5'}
allowed_files={
    'inputs/comsol_reference/magnetostatic_converged_55iter.vtu',
    'inputs/native_reference/blocked_magnetostatic_official_mesh/magnetostatic_solution.vtu',
    'inputs/native_reference/blocked_magnetostatic_native_hybrid_c48/magnetostatic_solution.vtu',
}
def test_no_results_packaged():
    bad=[]
    for p in ROOT.rglob('*'):
        if not p.is_file():continue
        rel=str(p.relative_to(ROOT))
        if rel.startswith('.venv/'):continue
        if p.suffix.lower() in forbidden_ext and rel not in allowed_files:bad.append(rel)
        if '/runs/' in '/'+rel or rel.startswith('runs/'):bad.append(rel)
    assert not bad, bad
