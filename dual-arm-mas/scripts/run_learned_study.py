"""One bounded manifest-driven learned evaluation slice, never retunes policies."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    root=Path(__file__).resolve().parents[2]
    config=json.loads((root/args.config).read_text(encoding='utf-8'))
    for path,sha in config['source_sha256'].items():
        if digest(root/path)!=sha:raise ValueError('Frozen learned source changed: '+path)
    for field in ['test_starts','perception']:
        if digest(root/config[field])!=config[field+'_sha256']:raise ValueError('Frozen asset changed: '+field)
    fit=next(f for f in config['fits'] if f['fit']==args.fit)
    if fit['status']!='selected_validation_only' or args.regime not in fit['policies']:
        raise ValueError('Training/validation selection incomplete for requested fit')
    policy=fit['policies'][args.regime]
    if digest(root/policy['checkpoint'])!=policy['sha256']:raise ValueError('Selected RL checkpoint changed')
    condition={} if args.condition=='core' else next(c for c in config['robustness'] if c['name']==args.condition)
    if condition and (args.regime not in condition['regimes'] or args.sensing!=condition['sensing']):
        raise ValueError('Regime/sensing not declared for this condition')
    count=condition.get('count',config['core_count'])
    output=f'dual-arm-mas/runs/learned-study/{args.condition}/fit-{args.fit}/{args.regime}/{args.sensing}'
    cmd=[sys.executable,str(root/'dual-arm-mas/scripts/evaluate_learned.py'),'--checkpoint',policy['checkpoint'],
        '--public',fit['public'],'--starts',config['test_starts'],'--count',str(count),'--sensing',args.sensing,
        '--decoder',config['decoder'],'--seed',str(config['evaluation_seed']),'--out',output,'--seconds',str(args.seconds)]
    if args.sensing=='rgb':
        cmd+=['--segmentation',config['perception']]
        if config['inspection_crop']:cmd+=['--crop']
    for key in ['drop','delay','b_limit_factor','occlusion','history_length']:
        if key in condition:cmd+=['--'+key.replace('_','-'),str(condition[key])]
    print('Bounded evaluation: '+' '.join(cmd),flush=True)
    subprocess.run(cmd,cwd=root,check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path('dual-arm-mas/configs/learned_study.json'))
    p.add_argument('--fit',type=int,required=True);p.add_argument('--regime',choices=['s1','s2','s3','s4'],required=True)
    p.add_argument('--sensing',choices=['gt','rgb'],required=True);p.add_argument('--condition',default='core')
    p.add_argument('--seconds',type=float,default=240);run(p.parse_args())
