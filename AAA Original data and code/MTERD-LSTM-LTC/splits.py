import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

def participant_splits(manifest, cfg):
    g = manifest.participant_id.to_numpy()
    if len(np.unique(g)) < cfg['n_splits']:
        raise ValueError('Fewer participants than outer folds')
    folds = []
    for fold, (development, test) in enumerate(GroupKFold(cfg['n_splits']).split(manifest, groups=g)):
        inner = GroupShuffleSplit(n_splits=1, test_size=cfg['validation_fraction'],
                                   random_state=cfg['split_seed']+fold)
        tr, va = next(inner.split(development, groups=g[development]))
        train, val = development[tr], development[va]
        sets = [set(g[ix]) for ix in [train, val, test]]
        if any(sets[i] & sets[j] for i,j in [(0,1),(0,2),(1,2)]):
            raise RuntimeError('Participant leakage detected')
        folds.append(dict(fold=fold, train=train.tolist(), val=val.tolist(), test=test.tolist(),
                  train_participants=sorted(sets[0]), val_participants=sorted(sets[1]),
                  test_participants=sorted(sets[2])))
    return folds
