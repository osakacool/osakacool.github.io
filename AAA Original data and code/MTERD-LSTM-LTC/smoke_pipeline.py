"""Artificial fixtures for software checks only; not MTERD empirical observations."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def run(*args):
    subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,check=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out).resolve()
    if out.exists():raise FileExistsError('Use a fresh smoke output path')
    out.mkdir(parents=True);(out/'ARTIFICIAL_FIXTURE_NOT_RESEARCH_DATA.txt').write_text(
        'Generated solely to check code behavior. Not real participants or MTERD data.\n')
    rng=np.random.default_rng(2026);records=[]
    ts=np.arange(0,12.1,.1);annotation_time=np.arange(13,dtype=float)
    for i in range(10):
        x=rng.normal(size=(len(ts),8)).astype(np.float32)
        x[:,0]=np.sin(ts)+i*.01
        features=out/f'fixture_{i}.npz';ann=out/f'fixture_{i}.csv'
        np.savez_compressed(features,X=x,time=ts,feature_names=np.array([f'fixture_feature_{j}' for j in range(8)]),
                            psycho_X=x[:,:3],psycho_feature_names=np.array(['loudness','sharpness','roughness']))
        # These named channels are artificial placeholders, not acoustic measurements.
        pd.DataFrame({'timestamp':annotation_time,'regulation_score':np.sin(annotation_time)+i*.01}).to_csv(ann,index=False)
        records.append(dict(participant_id=f'fixture_person_{i}',session_id=f'fixture_session_{i}',
                            features_path=features.name,annotations_path=ann.name))
    manifest=out/'fixture_manifest.csv';pd.DataFrame(records).to_csv(manifest,index=False)
    cfg=json.loads((ROOT/'configs/default.json').read_text())
    cfg.update(pca_components=4,pca_fit_max_frames=200,hidden_size=4,num_layers=2,sequence_length=12,
               max_epochs=1,batch_size=16,seeds=[17],tcn_dilations=[1,2],device='cpu',num_workers=0)
    config_path=out/'fixture_config.json';config_path.write_text(json.dumps(cfg,indent=2))
    run('scripts/02_validate_dataset.py','--manifest',manifest,'--config',config_path,'--report',out/'validation.json')
    run('scripts/04_preprocess_pca.py','--manifest',manifest,'--config',config_path,'--out',out/'pca')
    run('scripts/07_cross_validation.py','--manifest',manifest,'--config',config_path,'--out',out/'runs')
    run('scripts/08_evaluate_metrics.py','--runs',out/'runs','--config',config_path,'--out',out/'metrics')
    run('scripts/09_gmm_bic_analysis.py','--runs',out/'runs','--out',out/'metrics/gmm')
    run('scripts/10_generate_figures.py','--runs',out/'runs','--metrics',out/'metrics','--out',out/'figures')
    predictions=list((out/'runs').glob('*/predictions.csv'))
    assert len(predictions)==25, len(predictions)
    combined=pd.concat([pd.read_csv(p) for p in predictions])
    assert combined.model.nunique()==5
    assert combined.fold.nunique()==5
    assert combined.groupby('participant_id').fold.nunique().eq(1).all()
    assert (out/'metrics/paired_comparisons.csv').exists()
    assert (out/'figures/predictive_performance.pdf').exists()
    print('SMOKE PASS: five models, five folds, one seed; artificial fixtures only.')

if __name__=='__main__':main()
