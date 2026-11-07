from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import argparse
import hashlib
import os
import numpy as np
from mterd.io import read_manifest, dump_json, sha256


def main():
    p=argparse.ArgumentParser(description='6373 ComParE functionals in trailing acoustic windows')
    p.add_argument('--manifest',required=True)
    p.add_argument('--output-manifest',default='docs/dataset_manifest_extracted.csv')
    p.add_argument('--out',default='data/processed/features')
    p.add_argument('--window-seconds',type=float,required=True,
                   help='Explicit analysis window, absent from manuscript; e.g. 1.0')
    p.add_argument('--hop-seconds',type=float,default=.1)
    a=p.parse_args()
    if a.window_seconds<=0 or a.hop_seconds<=0:raise ValueError('Window and hop must be positive')
    import opensmile
    import soundfile as sf
    smile=opensmile.Smile(feature_set=opensmile.FeatureSet.ComParE_2016,
                           feature_level=opensmile.FeatureLevel.Functionals)
    manifest=read_manifest(a.manifest);out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    mout=Path(a.output_manifest).resolve();mout.parent.mkdir(parents=True,exist_ok=True)
    for ix,row in enumerate(manifest.itertuples(index=False)):
        if not getattr(row,'audio_path',''):
            raise ValueError('Audio extraction requires audio_path for every session')
        signal,sr=sf.read(row.audio_path,dtype='float32',always_2d=True)
        if sr!=44100:raise ValueError(f'{row.session_id}: expected 44100 Hz, observed {sr}')
        signal=signal.mean(axis=1)
        width=int(round(a.window_seconds*sr));hop=int(round(a.hop_seconds*sr))
        if len(signal)<width:raise ValueError('Audio is shorter than one analysis window')
        features=[];times=[]
        for end in range(width,len(signal)+1,hop):
            frame=smile.process_signal(signal[end-width:end],sr)
            if frame.shape!=(1,6373):raise ValueError(f'Unexpected ComParE shape {frame.shape}')
            value=frame.to_numpy(dtype=np.float32)[0]
            if not np.isfinite(value).all():
                raise ValueError(f'Nonfinite functionals in {row.session_id} at {end/sr}; inspect audio/window')
            features.append(value);times.append(end/sr)
        name=hashlib.sha256(row.session_id.encode()).hexdigest()[:20]
        dest=out/f'{name}.npz'
        np.savez_compressed(dest,X=np.stack(features),time=np.array(times),
                             feature_names=np.array(smile.feature_names,dtype=str))
        dump_json(dest.with_suffix('.json'),dict(session_id=row.session_id,audio_sha256=sha256(row.audio_path),
               sample_rate=sr,window_seconds=a.window_seconds,hop_seconds=a.hop_seconds,
               features=6373,opensmile_version=opensmile.__version__,alignment='trailing_window_endpoint'))
        manifest.loc[ix,'features_path']=str(dest)
        print(row.session_id, len(times), 'frames',flush=True)
    for col in ['features_path','annotations_path','audio_path']:
        if col in manifest:
            manifest[col]=manifest[col].map(lambda s:os.path.relpath(s,mout.parent) if s else '')
    manifest.to_csv(mout,index=False)
    print(mout)

if __name__=='__main__':main()
