"""Use one bounded worker slot efficiently across unchanged frozen study groups."""
from pathlib import Path
import argparse,json,subprocess,sys,time


def count_unique(directory):
    if (directory/'writer.lock').exists():raise ValueError('Another writer owns '+str(directory))
    path=directory/'episodes.jsonl'
    if not path.exists():return 0
    ids=set()
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if not line.endswith('\n'):raise ValueError('Partial record: '+str(path))
            row=json.loads(line)
            if row['start_id'] in ids:raise ValueError('Duplicate record: '+str(path))
            ids.add(row['start_id'])
    return len(ids)


def run(args):
    cfg=json.loads(Path('dual-arm-mas/configs/learned_study.json').read_text())
    jobs=[]
    if args.phase in ['core','all']:
        jobs=[('core',f['fit'],cfg['core_count']) for f in cfg['fits'] if f['status']=='selected_validation_only']
    if args.phase in ['robustness','all']:
        jobs += [(c['name'],0,c['count']) for c in cfg['robustness'] if args.regime in c['regimes'] and c['sensing']==args.sensing]
    began=time.monotonic()
    for condition,fit,target in jobs:
        directory=Path('dual-arm-mas/runs/learned-study')/condition/f'fit-{fit}'/args.regime/args.sensing
        count=count_unique(directory)
        if count>target:raise ValueError('Unexpected extra episodes')
        if count==target:continue
        remaining=args.seconds-(time.monotonic()-began)
        if remaining<10:break
        subprocess.run([sys.executable,'dual-arm-mas/scripts/run_learned_study.py','--fit',str(fit),
            '--regime',args.regime,'--sensing',args.sensing,'--condition',condition,'--seconds',str(remaining)],check=True)
        if count_unique(directory)<target:break
    print('Bounded queue exited; no claim of study completion',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--regime',choices=['s1','s2','s3','s4'],required=True)
    p.add_argument('--sensing',choices=['gt','rgb'],required=True);p.add_argument('--phase',choices=['core','robustness','all'],default='core')
    p.add_argument('--seconds',type=float,default=240);run(p.parse_args())
