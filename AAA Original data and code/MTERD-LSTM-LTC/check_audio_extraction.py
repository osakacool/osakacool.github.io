"""Real extractor check on a generated noise WAV; not a study recording."""
from pathlib import Path
import argparse
import subprocess
import sys
import numpy as np
import pandas as pd
import soundfile as sf

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(2026)
    sf.write(out/'ARTIFICIAL_NOISE.wav',rng.normal(0,.1,88200).astype('float32'),44100)
    pd.DataFrame({'timestamp':[0.,1.,2.],'regulation_score':[.1,.2,.3]}).to_csv(out/'labels.csv',index=False)
    pd.DataFrame([dict(participant_id='fixture_only',session_id='noise_fixture',
                      audio_path='ARTIFICIAL_NOISE.wav',features_path='',annotations_path='labels.csv')]).to_csv(out/'manifest.csv',index=False)
    subprocess.run([sys.executable,str(ROOT/'scripts/03_extract_audio_features.py'),
                    '--manifest',str(out/'manifest.csv'),'--output-manifest',str(out/'extracted.csv'),
                    '--out',str(out/'features'),'--window-seconds','1.0'],cwd=ROOT,check=True)
    files=list((out/'features').glob('*.npz'));assert len(files)==1
    with np.load(files[0],allow_pickle=False) as f:
        assert f['X'].shape==(11,6373),f['X'].shape
        assert np.isfinite(f['X']).all()
        np.testing.assert_allclose(np.diff(f['time']),.1)
    print('AUDIO EXTRACTOR PASS: 11 x 6373 functionals at 0.1-s hops from artificial noise.')

if __name__=='__main__':main()
