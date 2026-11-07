from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def dump_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')

def config(path=None):
    return json.loads(Path(path or ROOT/'configs/default.json').read_text())

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def read_manifest(path):
    path = Path(path).resolve()
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {'participant_id','session_id','features_path','annotations_path'}
    if not required <= set(df):
        raise ValueError(f'Manifest requires {sorted(required)}')
    if df.empty:
        raise ValueError('Manifest is a header-only template. Add verified session records.')
    if df.session_id.duplicated().any():
        raise ValueError('session_id must be globally unique')
    for name in ['participant_id','session_id']:
        if (df[name].str.strip() == '').any():
            raise ValueError(f'{name} must not be empty')
    for field in ['features_path','annotations_path','audio_path']:
        if field in df:
            df[field] = df[field].map(lambda s: str((path.parent/s).resolve()) if s else '')
    return df

def load_session(row, cfg, psycho=False):
    with np.load(row.features_path, allow_pickle=False) as f:
        x = None if psycho else np.array(f['X'], dtype=np.float32)
        times = np.array(f['time'], dtype=float)
        names = np.array(f['feature_names']).astype(str).tolist()
        if psycho:
            # Genuine named psychoacoustic inputs must be supplied, not PCA proxies.
            pn = np.array(f['psycho_feature_names']).astype(str).tolist()
            if not set(cfg['psycho_features']) <= set(pn):
                raise ValueError(f'Missing psychoacoustic features in {row.session_id}')
            x = np.asarray(f['psycho_X'], dtype=np.float32)[:, [pn.index(n) for n in cfg['psycho_features']]]
            names = cfg['psycho_features']
    if x.ndim != 2 or x.shape != (len(times), len(names)) or len(times) < 2:
        raise ValueError(f'Invalid feature dimensions: {row.session_id}')
    if len(set(names)) != len(names):
        raise ValueError('Feature names must be unique')
    if not np.isfinite(x).all() or not np.isfinite(times).all():
        raise ValueError(f'Nonfinite feature/time values in {row.session_id}; resolve upstream')
    if not np.allclose(np.diff(times), cfg['frame_step_seconds'], atol=1e-5):
        raise ValueError(f'Expected regular {cfg["frame_step_seconds"]} s features')
    a = pd.read_csv(row.annotations_path)
    target = cfg['target']
    if not {'timestamp', target} <= set(a):
        raise ValueError(f'Annotation requires timestamp and observed {target}: {row.annotations_path}')
    at = a.timestamp.to_numpy(float); ay = a[target].to_numpy(float)
    if len(at)<2 or not np.isfinite(at).all() or not np.isfinite(ay).all() or np.any(np.diff(at)<=0):
        raise ValueError('Annotations must be finite, strictly increasing, with at least two rows')
    # Linear alignment within the observed interval, never extrapolation.
    y = np.interp(times, at, ay, left=np.nan, right=np.nan)
    return dict(X=x, time=times, y=y, feature_names=names, annotations=a,
                participant_id=row.participant_id, session_id=row.session_id)

def validate_manifest(path, cfg):
    m = read_manifest(path); records = []; errors = []
    names = None
    for row in m.itertuples(index=False):
        try:
            s = load_session(row, cfg)
            if names is not None and names != s['feature_names']:
                raise ValueError('Feature names or order differ between sessions')
            names = s['feature_names']
            n = cfg['sequence_length']
            if len(s['X']) < n or not np.isfinite(s['y'][n-1:]).any():
                raise ValueError('No full-length window with an observed target')
            records.append(dict(session_id=row.session_id, participant_id=row.participant_id,
                     frames=len(s['X']), features=s['X'].shape[1],
                     feature_sha256=sha256(row.features_path), annotation_sha256=sha256(row.annotations_path)))
        except Exception as exc:
            errors.append(dict(session_id=row.session_id, error=str(exc)))
    return dict(sessions=len(m), participants=int(m.participant_id.nunique()),
                passed=not errors, records=records, errors=errors)
