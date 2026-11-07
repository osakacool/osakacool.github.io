from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import argparse
from mterd.io import config, validate_manifest, dump_json


def main():
    p=argparse.ArgumentParser(description='Validate actual session files and target availability')
    p.add_argument('--manifest',required=True);p.add_argument('--config')
    p.add_argument('--report',default='results/metrics/data_validation.json')
    p.add_argument('--strict-manuscript',action='store_true')
    a=p.parse_args();cfg=config(a.config);report=validate_manifest(a.manifest,cfg)
    if a.strict_manuscript:
        if report['participants']!=300 or report['sessions']!=1200:
            report['errors'].append({'error':'Actual counts differ from manuscript: 300 participants, 1200 sessions'})
        if any(r['features']!=6373 for r in report['records']):
            report['errors'].append({'error':'Raw feature dimension is not 6373'})
        report['passed']=not report['errors']
    dump_json(a.report,report);print(a.report)
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':main()
