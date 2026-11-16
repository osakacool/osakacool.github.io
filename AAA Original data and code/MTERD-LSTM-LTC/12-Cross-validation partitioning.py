"""Participant-disjoint five-fold partitions for MTERD sessions."""
import argparse
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

def create_folds(data,participant_col='participant_id',stratify_col=None,n_splits=5):
    if participant_col not in data or data[participant_col].isna().any():raise ValueError('Participant identifiers are required')
    groups=data[participant_col]
    if groups.nunique()<n_splits:raise ValueError('Insufficient distinct participants')
    if stratify_col:
        if stratify_col not in data:raise ValueError('Missing stratification column')
        splitter=StratifiedGroupKFold(n_splits=n_splits,shuffle=True,random_state=42)
        iterator=splitter.split(data,data[stratify_col],groups)
    else:
        splitter=GroupKFold(n_splits=n_splits)
        iterator=splitter.split(data,groups=groups)
    result=data.copy();result['fold']=-1
    for fold,(train,test) in enumerate(iterator):
        assert set(groups.iloc[train]).isdisjoint(set(groups.iloc[test]))
        result.iloc[test,result.columns.get_loc('fold')]=fold
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--participant-col',default='participant_id');ap.add_argument('--stratify-col')
    args=ap.parse_args()
    data=pd.read_csv(args.input)
    create_folds(data,args.participant_col,args.stratify_col).to_csv(args.output,index=False)
    print('Participant-disjoint fold assignments saved:',args.output)
