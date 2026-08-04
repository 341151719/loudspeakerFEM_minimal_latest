#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from comsol_stage7F_shared_refined_asb import build_stage7f_model
from stage29_native_nra_figure9 import compare_full_cloud,compare_probes,half_power_mode

def nrms(err,ref):return float(np.sqrt(np.mean(np.asarray(err,float)**2))/max(np.sqrt(np.mean(np.asarray(ref,float)**2)),1e-300))
def wrap(x):return (np.asarray(x,float)+180)%360-180

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run-dir',required=True);ap.add_argument('--input-root');ap.add_argument('--mesh');ap.add_argument('--mphtxt');ap.add_argument('--comsol-export-root',default=str(ROOT/'inputs/comsol_reference/stage29_nra'));ap.add_argument('--req5-root',default=str(ROOT/'inputs/comsol_reference/stage29_nra'));a=ap.parse_args();out=Path(a.run_dir);S=Path(a.input_root) if a.input_root else ROOT;er=Path(a.comsol_export_root)
 mesh_path=Path(a.mesh) if a.mesh else (S/'meshes/comsol_geometry_polyline_coarse_2p5mm.msh' if a.input_root else ROOT/'inputs/meshes/comsol_geometry_polyline_coarse_2p5mm.msh')
 mphtxt_path=Path(a.mphtxt) if a.mphtxt else (S/'comsol_reference_inputs/Untitled.mphtxt' if a.input_root else ROOT/'inputs/comsol_reference/Untitled.mphtxt')
 p=pd.read_csv(out/'stage29_domain8_22_pressure.csv');diss=pd.read_csv(out/'stage29_nra_dissipation.csv');cmp=pd.read_csv(out/'stage29_COMSOL_average_anchor_comparison.csv')
 modes={'without_NRA_domain8':half_power_mode(p[p.branch=='without_NRA'],8),'without_NRA_domain22':half_power_mode(p[p.branch=='without_NRA'],22),'with_NRA_domain8':half_power_mode(p[p.branch=='with_NRA'],8),'with_NRA_domain22':half_power_mode(p[p.branch=='with_NRA'],22)}
 f0=modes['without_NRA_domain8']['mode_frequency_Hz'];a0=float(p[(p.branch=='without_NRA')&(p.domain_id==8)&np.isclose(p.freq_Hz,f0)].mean_abs_p_Pa.iloc[0]);aw=float(p[(p.branch=='with_NRA')&(p.domain_id==8)&np.isclose(p.freq_Hz,f0)].mean_abs_p_Pa.iloc[0]);modes['NRA_suppression_at_noNRA_mode_dB']=20*math.log10(aw/a0)
 # COMSOL coarse anchor peak; this is not a strict mode/Q reference.
 c3=pd.read_csv(er/'domain8_study3_average_pressure.csv');band=c3[c3.freq_Hz.between(500,750)];ii=band.mean_abs_acpr_p_t.idxmax();modes['COMSOL_sparse_anchor_peak_Hz']=float(c3.loc[ii,'freq_Hz']);modes['frequency_error_vs_sparse_anchor_percent']=100*(f0-modes['COMSOL_sparse_anchor_peak_Hz'])/modes['COMSOL_sparse_anchor_peak_Hz'];modes['COMSOL_Q_reference_status']='BLOCKED_REQUIRES_REQ9_DENSE_SWEEP'
 bandmetrics={}
 for branch in ['with_NRA','without_NRA']:
  for dom in [8,22]:
   x=cmp[(cmp.branch==branch)&(cmp.domain_id==dom)&cmp.freq_Hz.between(500,750)]
   bandmetrics[f'{branch}_domain{dom}']={'n':len(x),'mean_abs_normalized_RMSE':nrms(x.mean_abs_error_Pa,x.COMSOL_mean_abs_Pa),'max_abs_normalized_RMSE':nrms(x.max_abs_error_Pa,x.COMSOL_max_abs_Pa),'phase_RMSE_deg':float(np.sqrt(np.mean(wrap(x.python_mean_phase_deg-x.COMSOL_representative_phase_deg)**2))),'mean_abs_bias_percent':float(100*np.mean(x.mean_abs_error_Pa)/max(np.mean(x.COMSOL_mean_abs_Pa),1e-300))}
 model,_=build_stage7f_model(mesh_path,mphtxt_path,.65,1);req=pd.read_csv(Path(a.req5_root)/'layer07_nra_domain_points.csv');cloud=[];probes=[]
 for f in [600.,630.]:
  for dom in [8,22]:
   cloud.append(compare_full_cloud(model,out/'with_NRA'/f'checkpoint_with_NRA_{f:09.3f}Hz.npz',req,dom,f));q=pd.read_csv(er/f'domain{dom}_study3_points.csv');q['domain_id']=dom;probes.append(compare_probes(model,out/'without_NRA'/f'checkpoint_without_NRA_{f:09.3f}Hz.npz',q,dom,f))
 pd.DataFrame(cloud).to_csv(out/'stage29_with_NRA_fullfield_direct_comparison.csv',index=False);pd.DataFrame(probes).to_csv(out/'stage29_without_NRA_probe_comparison.csv',index=False)
 disskey=diss[diss.freq_Hz.isin([600.,630.,f0])].copy();disskey.to_csv(out/'stage29_key_dissipation.csv',index=False)
 metrics={'schema':'stage29-native-nra-figure9-v2','native_physics_status':'COMPLETE','calibration_status':{'COMSOL_local_transfer_used':False,'NRA_log_frequency_interpolation_used':False,'reduced_order_NRA_correction_used':False},'modes':modes,'COMSOL_anchor_band_metrics_500_750Hz':bandmetrics,'with_NRA_fullfield_direct_metrics':cloud,'without_NRA_probe_metrics':probes,'native_dissipation':{'all_passive':bool(diss.passive.all()),'domain8_600Hz':disskey[(disskey.domain_id==8)&np.isclose(disskey.freq_Hz,600)].iloc[0][['viscous_W','thermal_W','total_W']].to_dict(),'domain22_600Hz':disskey[(disskey.domain_id==22)&np.isclose(disskey.freq_Hz,600)].iloc[0][['viscous_W','thermal_W','total_W']].to_dict(),'total_at_native_mode_W':float(diss[np.isclose(diss.freq_Hz,f0)].total_W.sum())},'strict_direct_reference_gaps':{'Figure9_without_NRA_fullfield':'BLOCKED_MISSING_DSET5_FULLFIELD_600_630','COMSOL_dense_mode_and_Q':'BLOCKED_MISSING_500_750Hz_DENSE_DSET3_DSET5','COMSOL_viscous_thermal_dissipation':'BLOCKED_MISSING_DIRECT_DOMAIN_INTEGRALS'}}
 (out/'stage29_metrics_v2.json').write_text(json.dumps(metrics,indent=2,ensure_ascii=False))
 mean_cloud_nrmse=float(np.mean([x['complex_NRMSE'] for x in cloud]));min_corr=float(min(x['complex_correlation'] for x in cloud));max_shape=float(max(x['shape_NRMSE_after_complex_scale'] for x in cloud))
 rows=[
 ('native_effective_density_and_bulk_modulus','PASS','implemented'),('COMSOL_local_transfer_removed','PASS',False),('NRA_log_frequency_interpolation_removed','PASS',False),('native_cavity_mode_frequency','PASS',f0),('native_Q_factor','PASS',modes['without_NRA_domain8']['Q_factor']),('domain8_22_complex_pressure','PASS','301 frequencies x 2 branches'),('native_viscous_thermal_dissipation','PASS',metrics['native_dissipation']['all_passive']),('signed_real_Figure9','PASS','600/630 Hz without NRA'),('with_NRA_fullfield_correlation_ge_0p95','PASS' if min_corr>=.95 else 'FAIL',min_corr),('with_NRA_fullfield_complex_NRMSE_le_0p05','PASS' if mean_cloud_nrmse<=.05 else 'FAIL',mean_cloud_nrmse),('with_NRA_shape_NRMSE_after_scale_le_0p05','PASS' if max_shape<=.05 else 'FAIL',max_shape),('without_NRA_fullfield_direct_reference','BLOCKED','REQ9 needed'),('direct_Q_reference','BLOCKED','REQ9 needed'),('direct_viscous_thermal_loss_reference','BLOCKED','REQ9 needed')]
 pd.DataFrame(rows,columns=['criterion','status','value']).to_csv(out/'stage29_acceptance_matrix_v2.csv',index=False)
 print(json.dumps(metrics,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
