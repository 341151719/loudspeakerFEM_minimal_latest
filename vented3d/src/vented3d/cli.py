"""Portable source-checkout CLI; no reference responses enter the solver."""
from pathlib import Path
import argparse,json,os,sys

ROOT=Path(__file__).resolve().parents[2]
SOURCE_ROOT=Path(__file__).resolve().parents[1]


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['inspect','prepare','solve','test'])
    parser.add_argument('--input-dir',type=Path,default=ROOT/'inputs/vented_enclosure')
    parser.add_argument('--config',type=Path,default=ROOT/'configs/native_mixed.json')
    parser.add_argument('--out',type=Path,default=Path.cwd()/'runs/vented3d')
    parser.add_argument('--freqs',type=float,nargs='+',default=[40.])
    parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--backend',choices=['pardiso','direct'],default='pardiso')
    parser.add_argument('--mkl-rt',default=os.environ.get('VENTED_MKL_RT'))
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv)
    if args.threads<1:parser.error('--threads must be positive')
    os.environ.update(MKL_NUM_THREADS=str(args.threads),OMP_NUM_THREADS=str(args.threads),
        MKL_DYNAMIC='FALSE',MKL_THREADING_LAYER='INTEL',MKL_INTERFACE_LAYER='LP64',OPENBLAS_NUM_THREADS='1')
    if args.mkl_rt:
        runtime=Path(args.mkl_rt).resolve()
        if not runtime.is_file():parser.error('--mkl-rt must identify an installed oneMKL runtime')
        os.environ['VENTED_MKL_RT']=str(runtime)
        if os.environ.get('VENTED3D_RUNTIME_READY')!=str(runtime):
            os.environ['VENTED3D_RUNTIME_READY']=str(runtime)
            os.environ['LD_LIBRARY_PATH']=str(runtime.parent)+os.pathsep+os.environ.get('LD_LIBRARY_PATH','')
            os.execv(sys.executable,[sys.executable,*sys.orig_argv[1:]])
    if args.command=='test':
        import unittest
        suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
        result=unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1
    from .inputs import volume_inputs,prepare_kernels
    if args.command=='inspect':
        xyz,types,native=volume_inputs(args.input_dir)
        print(json.dumps({'input_dir':str(args.input_dir.resolve()),'native_pressure_dofs':len(native),
            'native_geometry_nodes':len(xyz),'native_acoustic_elements':{k:len(v['active']) for k,v in types.items()},
            'reference_fields_needed_for_solve':False,'backend':args.backend,'thread_limit':args.threads},indent=2))
        return 0
    if args.command=='prepare':
        print(json.dumps(prepare_kernels(args.input_dir,args.out/'acoustic_cache',print),indent=2));return 0
    import fcntl
    import numpy as np
    from vented1.core import atomic_json,INPUT_FILES,sha256,digest,frequency_key
    from vented1.elements import Motor
    from vented1.engine import solve_frequency
    from .model import build_native_coupled
    args.out.mkdir(parents=True,exist_ok=True)
    lock=(args.out/'campaign.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('This output directory already has an active solver')
    cfg=json.loads(args.config.read_text());cfg['solver']['core_backend']=args.backend
    if cfg.get('level',0)!=0:raise ValueError('Native element correspondence currently requires level=0')
    if any(not np.isfinite(f) or f<=0 for f in args.freqs):raise ValueError('Frequencies must be finite and positive')
    identity={'configuration':cfg,
        'sources':{str(p.relative_to(SOURCE_ROOT)):sha256(p)
            for package in ['vented1','vented3d']
            for p in sorted((SOURCE_ROOT/package).rglob('*.py'))},
        'physical_inputs':{name:sha256(args.input_dir/name) for name in INPUT_FILES},
        'native_geometry':{p.name:sha256(p) for p in sorted((args.input_dir/'native').iterdir()) if p.is_file()},
        'reference_fields_used_to_assemble_or_solve':False}
    signature=digest(identity);manifest=args.out/'run_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['compatibility_hash']!=signature:
        raise ValueError('Output has incompatible code/inputs/configuration; choose a new --out')
    atomic_json(manifest,{**identity,'compatibility_hash':signature,'input_location':str(args.input_dir.resolve())})
    state={'status':'running','stage':'assembly','pid':os.getpid(),'requested_frequencies_Hz':args.freqs,'steps':[],'benchmark_acceptance':False}
    def save():atomic_json(args.out/'STATUS.json',state)
    def progress(name):state['stage']=name;save();print(name,flush=True)
    try:
        save()
        done=[]
        for freq in args.freqs:
            p=args.out/(frequency_key(freq)+'.json')
            if args.resume and p.exists():
                previous=json.loads(p.read_text())
                if previous.get('solver_success') and previous.get('compatibility_hash')==signature and p.with_suffix('.npz').exists():done.append(freq)
        todo=[f for f in args.freqs if f not in done]
        if todo:
            model=build_native_coupled(args.input_dir,cfg,args.out,progress)
            motor=Motor.load(args.input_dir,cfg['strict_bundle'])
            for freq in todo:
                progress(f'full_coupled_solve_{freq:g}')
                result=solve_frequency(model,motor,freq,complex(*cfg['voltage_peak_V']),cfg,args.out,signature,resume=args.resume)
                state['steps'].append({'frequency_Hz':freq,'solver_success':result['solver_success'],'elapsed_s':result['elapsed_s']});save()
                if not result['solver_success']:
                    raise ArithmeticError(f'Numerical acceptance failed at {freq:g} Hz; inspect the saved frequency report')
        state.update(status='complete',stage='finished',reused_frequencies_Hz=done);save()
    except BaseException as exc:
        state.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',reason=str(exc));save();raise
    return 0
