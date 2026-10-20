"""Offline comparison of completed Python solves to explicit reference files.

This tool is never imported by the production solver. Saved COMSOL responses
are opened only here, after independent solve checkpoints have been produced.
"""
from pathlib import Path
import argparse,json,sys,os
os.environ['OPENBLAS_NUM_THREADS']='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from vented1.core import rows,frequency_key,atomic_json,sha256
from vented1.reference import expression_table,scalar_error
from vented1.elements import Motor


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--reference-dir',type=Path,required=True)
    parser.add_argument('--pressure-reference',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    refs=expression_table(args.reference_dir/'frequency_reference.csv')
    index={float(value.real):i for (name,i),value in refs.items() if name=='freq'}
    direct={(r['plane'],float(r['theta_deg']),int(r['solution_index'])):
        complex(float(r['pressure_real_Pa_peak']),float(r['pressure_imag_Pa_peak']))
        for r in rows(args.reference_dir/'directivity_reference.csv')}
    pressure=None
    if args.pressure_reference:
        with np.load(args.pressure_reference) as archive:pressure=archive['reference_pressure']
        pressure_index={31.5:0,40.:1,315.:2,3550.:3}
    result=[]
    for file in sorted(args.run.glob('f_*.json')):
        status=json.loads(file.read_text())
        if 'solver_success' not in status:continue
        f=float(status['frequency_Hz']);i=index[f]
        if file.name!=frequency_key(f)+'.json':continue
        archive=args.run/(frequency_key(f)+'.npz')
        if not archive.exists():raise ValueError('Completed frequency report has no field checkpoint')
        with np.load(archive) as data:
            reference=np.array([[direct[plane,float(angle),i] for angle in data['theta_deg']] for plane in ['XY','XZ']])
            row={'frequency_Hz':f,'solver_success':status['solver_success'],
                'coil_velocity':scalar_error(complex(data['coil_velocity']),refs['v0',i],1e-8),
                'current':scalar_error(complex(data['current']),refs['Fe',i]/Motor.BL,1e-10),
                'X_pressure':scalar_error(complex(data['directivity'][0,0]),reference[0,0],1e-5),
                'Z_pressure':scalar_error(complex(data['directivity'][1,18]),reference[1,18],1e-5),
                'radiation_74_relative_L2_error':float(np.linalg.norm(data['directivity']-reference)/np.linalg.norm(reference)),
                'elapsed_s':status['elapsed_s'],'peak_process_memory_bytes':status['peak_process_memory_bytes'],
                'status_sha256':sha256(file)}
            if pressure is not None:
                ref=pressure[pressure_index[f]]
                if ref.shape!=data['p_interior'].shape:raise ValueError('Native pressure topology size differs')
                row['native_pressure_dof_relative_L2_error']=float(np.linalg.norm(data['p_interior']-ref)/np.linalg.norm(ref))
            result.append(row)
    if not result:raise ValueError('No completed frequencies were found')
    result.sort(key=lambda row:row['frequency_Hz'])
    state=json.loads((args.run/'STATUS.json').read_text()) if (args.run/'STATUS.json').exists() else {}
    manifest=args.run/'run_manifest.json'
    report={'kind':'offline_COMSOL_comparison_after_independent_solve','reference_used_during_assembly_or_solve':False,
        'campaign_status':state.get('status'),'benchmark_acceptance':False,'records':result,
        'reference_files':{name:sha256(args.reference_dir/name) for name in ['frequency_reference.csv','directivity_reference.csv']},
        'run_manifest_sha256':sha256(manifest) if manifest.exists() else None}
    args.out.parent.mkdir(parents=True,exist_ok=True);atomic_json(args.out,report)
    for row in result:
        print(f"{row['frequency_Hz']:g} Hz: coil {100*row['coil_velocity']['complex_relative_error']:.4f}%; "
            f"X {100*row['X_pressure']['complex_relative_error']:.4f}%; Z {100*row['Z_pressure']['complex_relative_error']:.4f}%; "
            f"radiation {100*row['radiation_74_relative_L2_error']:.4f}%")


if __name__=='__main__':main()
