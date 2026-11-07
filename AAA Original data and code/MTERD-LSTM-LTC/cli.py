import argparse
from pathlib import Path
from .io import config, read_manifest, validate_manifest, dump_json
from .splits import participant_splits

MODELS=['lstm_ltc','standard_lstm','attention_lstm','tcn','psychoacoustic']

def run_cli(default_models):
    p=argparse.ArgumentParser(description='Participant-disjoint training and held-out prediction')
    p.add_argument('--manifest',required=True)
    p.add_argument('--config',default=None)
    p.add_argument('--out',required=True)
    p.add_argument('--models',nargs='+',choices=MODELS,default=default_models)
    p.add_argument('--folds',nargs='+',type=int,default=None)
    p.add_argument('--seeds',nargs='+',type=int,default=None)
    a=p.parse_args();cfg=config(a.config)
    report=validate_manifest(a.manifest,cfg)
    if not report['passed']:
        raise ValueError(report['errors'])
    m=read_manifest(a.manifest);splits=participant_splits(m,cfg)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    split_path=out/'splits.json'
    if split_path.exists():
        import json
        if json.loads(split_path.read_text()) != splits:
            raise ValueError('Output contains a different split plan')
    else:
        dump_json(split_path,splits)
    dump_json(out/'validation_report.json',report)
    from .training import run_one
    for split in splits:
        if a.folds is not None and split['fold'] not in a.folds:
            continue
        for name in a.models:
            # Deterministic ridge uses the same explicit seed labels for matched comparisons.
            for seed in (a.seeds or cfg['seeds']):
                run_one(m,split,cfg,name,seed,out)
