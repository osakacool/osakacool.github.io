from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import argparse
import joblib
from mterd.io import config, read_manifest, load_session, dump_json
from mterd.splits import participant_splits
from mterd.preprocessing import FeatureTransform


def main():
    p=argparse.ArgumentParser(description='Fit and export PCA separately within each training fold')
    p.add_argument('--manifest',required=True);p.add_argument('--config')
    p.add_argument('--out',default='data/processed/pca')
    a=p.parse_args();cfg=config(a.config);m=read_manifest(a.manifest)
    for split in participant_splits(m,cfg):
        out=Path(a.out)/f'fold{split["fold"]}';out.mkdir(parents=True,exist_ok=True)
        if (out/'preprocessor.joblib').exists():raise FileExistsError(out)
        transform=FeatureTransform(cfg['pca_components'],cfg['pca_fit_max_frames'],cfg['split_seed'])
        transform.fit(lambda: (load_session(m.iloc[i],cfg)['X'] for i in split['train']))
        joblib.dump(transform,out/'preprocessor.joblib')
        dump_json(out/'report.json',dict(split=split,explained_variance=transform.explained_variance,
                        reached_95_percent=transform.explained_variance>=.95,fit_frames=transform.fit_frames))
        for row in m.itertuples(index=False):
            s=load_session(row,cfg)
            # Session IDs are stored as metadata, never used as unchecked path components.
            import hashlib
            name=hashlib.sha256(row.session_id.encode()).hexdigest()[:20]
            import numpy as np
            np.savez_compressed(out/f'{name}.npz',X=transform.transform(s['X']),time=s['time'],
                                session_id=row.session_id)
        print(out)

if __name__=='__main__':main()
