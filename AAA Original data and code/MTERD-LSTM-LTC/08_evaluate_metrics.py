from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import argparse
from mterd.io import config
from mterd.evaluation import evaluate_paths


def main():
    p=argparse.ArgumentParser(description='Compute metrics from held-out 1-Hz predictions')
    p.add_argument('--runs',required=True);p.add_argument('--config')
    p.add_argument('--out',default='results/metrics')
    a=p.parse_args();paths=sorted(Path(a.runs).glob('*/predictions.csv'))
    if not paths:raise ValueError('No actual prediction files found')
    print(evaluate_paths(paths,a.out,config(a.config)).to_string(index=False))

if __name__=='__main__':main()
