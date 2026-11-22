"""Explicit renewed-window adapter; frozen evaluation mathematics stays untouched.

Only the evaluator's between-episode calendar gate is replaced. Source/data/runtime
checks in evaluate_learned.run still execute. Original JSONL prefixes and every
invocation of this adapter are independently pinned and audited.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,time,traceback
import evaluate_learned as evaluator

MANIFEST=Path('dual-arm-mas/configs/learned_continuation_20261120.json')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_prefix(out,manifest):
    key=str(out).replace('\\','/')
    original=manifest['original_prefixes'][key]
    path=out/'episodes.jsonl'
    with path.open('rb') if path.exists() else __import__('io').BytesIO() as stream:
        prefix=stream.read(original['bytes'])
    if len(prefix)!=original['bytes'] or hashlib.sha256(prefix).hexdigest()!=original['sha256']:
        raise ValueError('Original evaluation prefix was changed: '+key)


def read_ids(out):
    raw=out/'episodes.jsonl';ids=set()
    if not raw.exists():return ids
    with raw.open(encoding='utf-8') as stream:
        for line in stream:
            if not line.endswith('\n'):raise ValueError('Partial episode record')
            i=json.loads(line)['start_id']
            if i in ids:raise ValueError('Duplicate episode record')
            ids.add(i)
    return ids


def one_group(cfg,manifest,fit,regime,sensing,condition,seconds):
    name=condition.get('name','core');count=condition.get('count',cfg['core_count'])
    out=Path('dual-arm-mas/runs/learned-study')/name/f"fit-{fit['fit']}"/regime/sensing
    out.mkdir(parents=True,exist_ok=True);lock=out/'writer.lock'
    with lock.open('x') as stream:stream.write(str(os.getpid()))
    record={'started_utc':datetime.now(timezone.utc).isoformat(),'manifest_sha256':digest(MANIFEST),
            'adapter_sha256':digest(__file__),'out':str(out),'seconds':seconds}
    try:
        verify_prefix(out,manifest);before=read_ids(out);record['before_n']=len(before)
        policy=fit['policies'][regime]
        if digest(policy['checkpoint'])!=policy['sha256']:raise ValueError('Selected actor changed')
        args=argparse.Namespace(checkpoint=Path(policy['checkpoint']),public=Path(fit['public']),
            starts=Path(cfg['test_starts']),count=count,sensing=sensing,decoder=cfg['decoder'],
            seed=cfg['evaluation_seed'],out=out,seconds=seconds,images=False,
            segmentation=Path(cfg['perception']) if sensing=='rgb' else None,
            crop=bool(cfg['inspection_crop'] and sensing=='rgb'),
            drop=condition.get('drop',0.),delay=condition.get('delay',0),
            b_limit_factor=condition.get('b_limit_factor',1.),occlusion=condition.get('occlusion',0.),
            history_length=condition.get('history_length',4))
        evaluator.run(args)
        after=read_ids(out);verify_prefix(out,manifest)
        if not before<=after:raise ValueError('Evaluation lost prior episodes')
        record.update(after_n=len(after),new_start_ids=sorted(after-before),status='returned')
        return len(after)==count
    except BaseException:
        record.update(status='error',error=traceback.format_exc());raise
    finally:
        record['ended_utc']=datetime.now(timezone.utc).isoformat()
        with (out/'continuation_invocations.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(record)+'\n')
        lock.unlink()


def run(args):
    manifest=json.loads(MANIFEST.read_text(encoding='utf-8'))
    if digest(__file__)!=manifest['adapter_sha256']:raise ValueError('Continuation adapter changed')
    config_path=Path(manifest['study_config'])
    if digest(config_path)!=manifest['study_config_sha256']:raise ValueError('Frozen study changed')
    cfg=json.loads(config_path.read_text())
    for path,sha in cfg['source_sha256'].items():
        if digest(path)!=sha:raise ValueError('Frozen mathematical source changed: '+path)
    for key in ['test_starts','perception']:
        if digest(cfg[key])!=cfg[key+'_sha256']:raise ValueError('Frozen asset changed')
    end=datetime.fromisoformat(manifest['window_end_ist'])
    if datetime.now(timezone.utc)>=end:raise ValueError('Renewed evaluation window expired')
    # Explicitly recorded scheduling-only override, never a hidden pin bypass.
    evaluator.allowed_time=lambda:datetime.now(timezone.utc)<end
    jobs=[]
    if args.phase in ['core','all']:
        jobs += [(f,{}) for f in cfg['fits'] if f['status']=='selected_validation_only']
    if args.phase in ['robustness','all']:
        jobs += [(cfg['fits'][0],c) for c in cfg['robustness'] if args.regime in c['regimes'] and args.sensing==c['sensing']]
    began=time.monotonic()
    for fit,condition in jobs:
        out=Path('dual-arm-mas/runs/learned-study')/condition.get('name','core')/f"fit-{fit['fit']}"/args.regime/args.sensing
        if (out/'writer.lock').exists():raise ValueError('Another writer owns '+str(out))
        verify_prefix(out,manifest);count=condition.get('count',cfg['core_count'])
        n=len(read_ids(out))
        if n>count:raise ValueError('Unexpected extra episodes')
        if n==count:continue
        remaining=args.seconds-(time.monotonic()-began)
        if remaining<10 or not evaluator.allowed_time():break
        if not one_group(cfg,manifest,fit,args.regime,args.sensing,condition,remaining):break
    print('Renewed-window slice checkpointed; not a completion claim',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--regime',choices=['s1','s2','s3','s4'],required=True)
    p.add_argument('--sensing',choices=['gt','rgb'],required=True)
    p.add_argument('--phase',choices=['core','robustness','all'],default='all')
    p.add_argument('--seconds',type=float,default=240);run(p.parse_args())
