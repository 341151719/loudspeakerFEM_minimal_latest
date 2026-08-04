#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd


def main():
    ap=argparse.ArgumentParser();ap.add_argument('raw_dir',type=Path);ap.add_argument('--outdir',type=Path,default=Path('req8_analysis'));a=ap.parse_args();a.outdir.mkdir(parents=True,exist_ok=True)
    integ=pd.read_csv(a.raw_dir/'l4_magnetomechanics_force_integrals.csv')
    total=integ[integ.domain_id.astype(int)==0].copy()
    F=total.force_z_real_N.to_numpy()+1j*total.force_z_imag_N.to_numpy();I=total.I_real_A.to_numpy()+1j*total.I_imag_A.to_numpy();bl=F/I
    summary={
      'n_frequencies':int(len(total)),
      'force_field_nonzero':bool(np.nanmax(np.abs(F))>0),
      'median_abs_Fz_per_I_N_per_A':float(np.nanmedian(np.abs(bl))),
      'low_frequency_abs_Fz_per_I_N_per_A':float(np.nanmedian(np.abs(bl[total.freq_Hz<=100]))) if np.any(total.freq_Hz<=100) else None,
      'low_frequency_phase_Fz_per_I_deg':float(np.nanmedian(np.angle(bl[total.freq_Hz<=100],deg=True))) if np.any(total.freq_Hz<=100) else None,
      'BL_acceptance_8_to_13_N_per_A':bool(8<=np.nanmedian(np.abs(bl[total.freq_Hz<=100]))<=13) if np.any(total.freq_Hz<=100) else False,
    }
    total.assign(force_z_per_I_abs_N_per_A=np.abs(bl),force_z_per_I_phase_deg=np.angle(bl,deg=True)).to_csv(a.outdir/'req8_force_transfer.csv',index=False)
    (a.outdir/'req8_l4_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
