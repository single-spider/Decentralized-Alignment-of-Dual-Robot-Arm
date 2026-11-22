"""Bounded validation of the TWO predeclared trained candidates; no test access."""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess,sys,time
import torch


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    config_path=Path('dual-arm-mas/configs/learned_study.json')
    cfg=json.loads(config_path.read_text(encoding='utf-8'))
    fit=next(f for f in cfg['fits'] if f['fit']==args.fit)
    if fit['status']=='selected_validation_only':
        print('Already selected; refusing a second selection',flush=True);return
    for path,sha in cfg['source_sha256'].items():
        if digest(path)!=sha:raise ValueError('Frozen source changed: '+path)
    if digest(cfg['perception'])!=cfg['perception_sha256']:raise ValueError('Perception changed')
    began=time.monotonic();selected={};candidates={}
    validation=Path('dual-arm-mas/runs/validation/starts.json')
    ids={s['id'] for s in json.loads(validation.read_text())['starts'][:24]}
    for regime in ['s1','s2','s3','s4']:
        directory=Path(f"dual-arm-mas/runs/rl-{regime}-{fit['seed']}")
        if (directory/'writer.lock').exists():raise ValueError('Training still owns candidate output')
        latest=torch.load(directory/'latest.pt',map_location='cpu',weights_only=False)
        if latest['steps']<cfg['training_target_steps_per_regime_per_seed']:
            raise ValueError('Final training budget not yet complete: '+regime)
        late=directory/f"step-{latest['steps']}.pt"
        if not late.exists():shutil.copy2(directory/'latest.pt',late)
        early_files=sorted((p for p in directory.glob('step-*.pt') if 2500<=int(p.stem.split('-')[1])<4000),
                           key=lambda p:int(p.stem.split('-')[1]))
        if len(early_files)!=1:raise ValueError('Need exactly one predeclared early candidate: '+regime)
        records=[]
        for checkpoint in [early_files[0],late]:
            steps=int(checkpoint.stem.split('-')[1]);out=directory/f'validation-{steps}-neural-frozen'
            remaining=args.seconds-(time.monotonic()-began)
            if remaining<10:print('Bounded validation checkpoint; rerun same command',flush=True);return
            cmd=[sys.executable,'dual-arm-mas/scripts/evaluate_learned.py','--checkpoint',str(checkpoint),
                '--public',fit['public'],'--starts',str(validation),'--count','24','--sensing','rgb','--decoder',cfg['decoder'],
                '--seed','501','--segmentation',cfg['perception'],'--out',str(out),'--seconds',str(remaining)]
            if cfg['inspection_crop']:cmd+=['--crop']
            subprocess.run(cmd,check=True)
            rows=[json.loads(line) for line in (out/'episodes.jsonl').read_text().splitlines()]
            if len(rows)!=len(ids) or {r['start_id'] for r in rows}!=ids:
                print('Validation incomplete; rerun same command',flush=True);return
            summary_path=out/'summary.json';summary=json.loads(summary_path.read_text())
            meta=json.loads((out/'provenance.json').read_text())
            if (meta['checkpoint_sha256']!=digest(checkpoint) or meta['decoder']!=cfg['decoder']
                or meta['sensing']!='rgb' or meta['inspection_crop']!=cfg['inspection_crop']):
                raise ValueError('Validation provenance mismatch')
            records.append({'checkpoint':str(checkpoint),'sha256':digest(checkpoint),'selected_training_steps':steps,
                'validation_summary':str(summary_path),'validation_summary_sha256':digest(summary_path),
                'success_rate':summary['success_rate'],'mean_cost_rad2':summary['mean_cost_rad2'],'mean_steps':summary['mean_steps']})
        chosen=min(records,key=lambda r:(-r['success_rate'],r['mean_cost_rad2'],r['mean_steps']))
        selected[regime]={k:chosen[k] for k in ['checkpoint','sha256','selected_training_steps','validation_summary','validation_summary_sha256']}
        candidates[regime]=records
    # Preserve every common frozen field and other fit; only fill this pending fit.
    current=json.loads(config_path.read_text(encoding='utf-8'))
    if current!=cfg:raise ValueError('Study manifest changed during validation')
    fit['policies']=selected;fit['candidate_validation']=candidates;fit['status']='selected_validation_only'
    temporary=config_path.with_suffix('.tmp');temporary.write_text(json.dumps(cfg,indent=2)+'\n',encoding='utf-8')
    temporary.replace(config_path)
    print(json.dumps({'selected_fit':args.fit,'policies':selected}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fit',type=int,choices=[1,2],required=True)
    p.add_argument('--seconds',type=float,default=240);run(p.parse_args())
