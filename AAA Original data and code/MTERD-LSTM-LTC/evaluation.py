from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t, ttest_rel
from .metrics import regression, ccc, ms_dtw, tec

METRICS = ['RMSE','Pearson_r','CCC','TCC','MS_DTW','TEC']

def summarize_predictions(frame, cfg):
    required = {'model','fold','seed','participant_id','session_id','timestamp','y_true','y_pred'}
    if not required <= set(frame):
        raise ValueError(f'Prediction CSV requires {sorted(required)}')
    keys = ['model','fold','seed','session_id','timestamp']
    if frame.duplicated(keys).any():
        raise ValueError('Duplicate test predictions')
    # Each participant must have exactly one outer test fold, across all models/seeds.
    if (frame.groupby('participant_id').fold.nunique() != 1).any():
        raise ValueError('Participant appears in multiple test folds')
    # Ensure methods/seeds use identical observations and target values in every fold.
    for fold, ff in frame.groupby('fold'):
        reference = None
        for _, g in ff.groupby(['model','seed']):
            selected = g.sort_values(['session_id','timestamp'])[
                ['participant_id','session_id','timestamp','y_true']].reset_index(drop=True)
            if reference is not None and not selected.equals(reference):
                raise ValueError(f'Methods/seeds do not share identical evaluation points in fold {fold}')
            reference = selected
    session_rows, fold_rows = [], []
    for (model,fold,seed), g in frame.groupby(['model','fold','seed']):
        per_session = []
        for session, s in g.groupby('session_id'):
            s = s.sort_values('timestamp')
            if not np.allclose(np.diff(s.timestamp),1.0):
                raise ValueError(f'Missing 1-Hz evaluation points in session {session}')
            y, p = s.y_true.to_numpy(), s.y_pred.to_numpy()
            row = dict(model=model,fold=int(fold),seed=int(seed),session_id=session,
                       participant_id=s.participant_id.iloc[0],n=len(s), **regression(y,p))
            row.update(TCC=ccc(y,p), MS_DTW=ms_dtw(y,p,cfg['ms_dtw_scales'],
                       cfg['ms_dtw_band_fraction'],cfg['ms_dtw_warp_penalty']),
                       TEC=tec(y,p,cfg['tec_reference'],cfg['tec_start_points']))
            session_rows.append(row); per_session.append(row)
        pooled = regression(g.y_true.to_numpy(),g.y_pred.to_numpy())
        pooled.update({metric:float(np.nanmean([r[metric] for r in per_session]))
                       for metric in ['TCC','MS_DTW','TEC']})
        fold_rows.append(dict(model=model,fold=int(fold),seed=int(seed),**pooled))
    return pd.DataFrame(session_rows), pd.DataFrame(fold_rows)

def fold_statistics(folds):
    # Seeds are repeated fits on the same participants, not independent folds.
    averaged = folds.groupby(['model','fold'])[METRICS].mean().reset_index()
    summary, comparisons = [], []
    for model, group in averaged.groupby('model'):
        for metric in METRICS:
            v = group[metric].dropna().to_numpy(); n = len(v)
            sd = float(v.std(ddof=1)) if n>1 else float('nan')
            half = float(t.ppf(.975,n-1)*sd/np.sqrt(n)) if n>1 else float('nan')
            summary.append(dict(model=model,metric=metric,n_folds=n,mean=float(v.mean()),
                        sd=sd,ci_low=float(v.mean()-half),ci_high=float(v.mean()+half)))
    ours = averaged[averaged.model=='lstm_ltc'].set_index('fold')
    for model in sorted(set(averaged.model)-{'lstm_ltc'}):
        baseline = averaged[averaged.model==model].set_index('fold')
        if set(ours.index) != set(baseline.index):
            raise ValueError(f'Unmatched fold IDs for {model}')
        if ours.empty:
            continue
        baseline = baseline.loc[ours.index]
        for metric in METRICS:
            a, b = ours[metric].to_numpy(), baseline[metric].to_numpy()
            good = np.isfinite(a) & np.isfinite(b); a,b=a[good],b[good]
            delta = a-b; n = len(delta)
            if n<2:
                continue
            dsd = delta.std(ddof=1)
            if dsd <= np.finfo(float).eps * max(1,float(np.max(np.abs(delta)))):
                dsd = 0.0
            stat, p = ttest_rel(a,b) if dsd>0 else (float('nan'),float('nan'))
            comparisons.append(dict(baseline=model,metric=metric,n_pairs=n,t=float(stat),p_raw=float(p),
                               mean_difference=float(delta.mean()),
                               cohen_dz=float(delta.mean()/dsd) if dsd>0 else float('nan')))
    c = pd.DataFrame(comparisons)
    if not c.empty:
        c['p_holm'] = np.nan
        good = c.p_raw.notna(); order = c.loc[good].p_raw.sort_values().index
        adjusted = np.maximum.accumulate([c.loc[idx,'p_raw']*(len(order)-rank)
                                          for rank,idx in enumerate(order)])
        c.loc[order,'p_holm'] = np.minimum(1,adjusted)
    return pd.DataFrame(summary), c, averaged

def evaluate_paths(paths, out, cfg):
    frames = [pd.read_csv(p,dtype={'participant_id':str,'session_id':str}) for p in paths]
    frame = pd.concat(frames,ignore_index=True)
    sessions, folds = summarize_predictions(frame,cfg)
    summary, comparisons, averaged = fold_statistics(folds)
    out = Path(out); out.mkdir(parents=True,exist_ok=True)
    for name,table in [('per_session',sessions),('per_fold_seed',folds),('per_fold',averaged),
                       ('summary',summary),('paired_comparisons',comparisons)]:
        table.to_csv(out/f'{name}.csv',index=False)
    return summary
