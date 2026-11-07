from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import json
from mterd.io import sha256


def main():
    p=argparse.ArgumentParser(description='Download the manuscript-listed Kaggle resource')
    p.add_argument('--slug',default='colabsss/music-therapy-emotional-response-dataset')
    p.add_argument('--inventory',default='data/raw/download_inventory.csv')
    p.add_argument('--hash-files',action='store_true')
    a=p.parse_args()
    import kagglehub
    try:
        root=Path(kagglehub.dataset_download(a.slug)).resolve()
    except Exception as exc:
        raise SystemExit(f'Dataset download failed: {exc}. No replacement data were generated.')
    rows=[]
    for f in sorted(root.rglob('*')):
        if f.is_file():
            rows.append(dict(relative_path=str(f.relative_to(root)),bytes=f.stat().st_size,
                             sha256=sha256(f) if a.hash_files else ''))
    out=Path(a.inventory);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=['relative_path','bytes','sha256']);w.writeheader();w.writerows(rows)
    out.with_suffix('.json').write_text(json.dumps(dict(slug=a.slug,download_location=str(root),
          files=len(rows),status='downloaded_not_schema_verified'),indent=2))
    print(f'Downloaded {len(rows)} files to {root}; inspect actual files and build the session manifest.')

if __name__=='__main__':main()
