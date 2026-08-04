#!/usr/bin/env python3
from pathlib import Path
import argparse,json,pandas as pd,numpy as np
ap=argparse.ArgumentParser();ap.add_argument('raw_dir',type=Path);ap.add_argument('--out',type=Path,default=Path('req10_validation.json'));a=ap.parse_args()
p=a.raw_dir/'figure5_Jphi_domain_points.csv';l=a.raw_dir/'figure5_domain_joule_loss.csv';g=a.raw_dir/'figure5_blocked_global.csv'
d=pd.read_csv(p);loss=pd.read_csv(l);glob=pd.read_csv(g);required={50,900,2000,5000,8000};
summary={'point_rows':len(d),'frequencies':sorted(d.solved_freq_Hz.unique().tolist()),'domains':sorted(d.domain_id.unique().astype(int).tolist()),'all_required_frequencies':required.issubset(set(d.solved_freq_Hz)),'both_domains_each_frequency':all(set(x.domain_id.astype(int))=={6,23} for _,x in d.groupby('solved_freq_Hz')),'finite_Jiphi':bool(np.isfinite(d[['Jiphi_real_A_m2','Jiphi_imag_A_m2']]).all().all()),'loss_rows':len(loss),'loss_finite':bool(np.isfinite(loss[['Jiphi_loss_W','Jphi_loss_W']]).all().all()),'global_rows':len(glob)}
a.out.write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
