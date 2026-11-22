"""Append-only dynamic assembly evaluation; frozen policies, no training/test tuning.

The original alignment study retains its 1000-case conditions. This separately
frozen contact-transfer study is exploratory, 128 paired starts per open condition
and 32 per stress condition, not a replacement for that original study.
"""
import argparse,hashlib,json,os,sys,time,shutil
from pathlib import Path
from dataclasses import asdict,replace
from datetime import datetime,timezone
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
import sfn
from dual_arm_mas.contact_world_v2 import ContactWorld,ContactConfig,insert
from dual_arm_mas.episode import AlignmentEpisode,SCALE
from dual_arm_mas.rl import load_policy,routed_features
from dual_arm_mas.vision import RGBPoseSensor
from evaluate_learned import VisionObservationLoop

class BudgetExpired(RuntimeError):pass

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def episode(start,group,study,model,public,sensor,allowed):
    def check():
        if not allowed():raise BudgetExpired('wall-clock slice ended; retry same start deterministically')
    cfg=ContactConfig();cfg=replace(cfg,motor_forces=tuple(x*group.get('motor_scale',1.) for x in cfg.motor_forces))
    with ContactWorld(cfg) as world:
        world.attach_parts(blocked=group.get('blocked',False));env=AlignmentEpisode(world);env.reset(start['q'],seed=study['evaluation_seed']+start['id'])
        check();settle=world.advance({n:world.q(n) for n in world.robots},240)
        sensor.reset();loop=VisionObservationLoop(env,sensor);trace=[];perceptions=[];failure=None
        if settle['stop_reason']:failure='settle_'+settle['stop_reason']
        previous={};rng=np.random.default_rng(study['evaluation_seed']+start['id']+1)
        for step in range(20):
            check()
            if failure:break
            obs=env.observations()
            if group['sensing']=='rgb':
                obs,rgb,perception=loop.replace(obs)
                diagnostic={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in perception.items() if k!='mask'}
                diagnostic['true_error_diagnostic_only']=world.error().tolist();perceptions.append(diagnostic)
                if obs is None:failure='perception';break
            error=obs['A']['error']
            if np.all(abs(error)<=np.array([.0008,.0008,np.deg2rad(.8)])):break
            if group['method']=='half':
                comm={'bytes':0,'messages':0,'rounds':0,'payload_hex':{}}
                fractions=np.full((2,3),.5);features=None
            else:
                features,comm,sent=routed_features(obs,group['regime'],public,env.history,
                    drop=group.get('drop',0.),rng=rng,delayed=previous if group.get('delay',0) else None)
                previous=sent
                with torch.no_grad():fractions=(model.distribution(torch.tensor(features)).probs@torch.linspace(0,1,5)).numpy()
            actions={n:fractions[i]*np.clip(obs[n]['error']/SCALE,-1,1)*SCALE*(-1 if n=='A' else 1) for i,n in enumerate(('A','B'))}
            actor_errors={n:o['error'].tolist() for n,o in obs.items()};env.done=False
            _,_,_,_,record=env.step(actions,comm)
            execution=record.pop('execution');record['dynamic_execution_contact']=None if execution is None else execution['contact']
            record['actor_error']=actor_errors;record['fractions']=fractions.tolist();record['local_features']=None if features is None else features.tolist();trace.append(record)
            if not record['feasible']:failure='alignment_execution:'+str(record['invalid_reason']);break
        check()
        # Fresh image gate before axial motion. Never use a stale or fixed-height
        # estimate after descent has begun; axial stage uses contact/encoder geometry.
        last_estimate=None
        if not failure:
            obs=env.observations()
            if group['sensing']=='rgb':
                obs,_,perception=loop.replace(obs)
                if obs is None:failure='perception_final_gate'
            if obs is not None:
                last_estimate=obs['A']['error'].tolist()
                if not np.all(abs(obs['A']['error'])<=np.array([.001,.001,np.deg2rad(1)])):failure='estimated_alignment_gate'
        alignment={'success':failure is None,'failure':failure,'trace':trace,'perceptions':perceptions,
            'final_estimate':last_estimate,'true_final_error_diagnostic_only':world.error().tolist()}
        initial_joints={n:world.q(n) for n in world.robots};previous_q={n:q.copy() for n,q in initial_joints.items()}
        insertion_cost=0.;insertion_l1=0.
        def measured_increment(record):
            nonlocal insertion_cost,insertion_l1,previous_q
            q={n:world.q(n) for n in world.robots};delta={n:q[n]-previous_q[n] for n in q}
            cost=float(sum(v@v for v in delta.values()));travel=float(sum(abs(v).sum() for v in delta.values()))
            record['measured_joints']={n:v.tolist() for n,v in q.items()};record['joint_displacement_cost_rad2']=cost;record['joint_travel_rad']=travel
            insertion_cost+=cost;insertion_l1+=travel;previous_q=q
        if failure:insertion={'success':False,'reason':'alignment_failed','trace':[],'final':world.insertion_state()}
        else:
            original=world.descend
            def bounded():check();return original()
            world.descend=bounded;insertion=insert(world,observer=measured_increment)
        # Hold/recovery endpoints are additionally measured, not omitted from cost.
        q={n:world.q(n) for n in world.robots};tail={n:q[n]-previous_q[n] for n in q}
        insertion_cost+=float(sum(v@v for v in tail.values()));insertion_l1+=float(sum(abs(v).sum() for v in tail.values()))
        return {'start_id':start['id'],'group':group['id'],'method':group['method'],'regime':group['regime'],'fit':group['fit'],
            'sensing':group['sensing'],'initial_joints':start['q'],'contact_config':asdict(cfg),'settle':settle,'alignment':alignment,'insertion':insertion,
            'success':insertion['success'],'reason':failure or insertion['reason'],'alignment_commands':len(trace),'insertion_commands':len(insertion['trace']),
            'alignment_cost_rad2':sum(x['motion_cost_rad2'] for x in trace),'insertion_endpoint_cost_rad2':insertion_cost,
            'alignment_joint_travel_rad':sum(x['joint_travel_rad'] for x in trace),'insertion_endpoint_joint_travel_rad':insertion_l1,
            'payload_bytes':sum(x['communication']['bytes'] for x in trace),'payload_messages':sum(x['communication']['messages'] for x in trace),
            'simulation_seconds':world.tick/240,'final_measured_joints':{n:v.tolist() for n,v in q.items()}}


def run(args):
    torch.set_num_threads(1);study=json.loads(args.config.read_text());group=next(x for x in study['groups'] if x['id']==args.group)
    if digest(study['starts'])!=study['starts_sha256']:raise ValueError('Frozen start manifest changed')
    end=datetime.fromisoformat(args.window_end)
    if end.tzinfo is None or datetime.now(timezone.utc)>=end:raise ValueError('Future timezone-aware window required')
    source=Path(__file__).resolve();base=source.parents[1];cfgpath=Path(study['alignment_config']);cfg=json.loads(cfgpath.read_text())
    if digest(cfgpath)!=study['alignment_config_sha256']:raise ValueError('Frozen alignment configuration changed')
    for path,sha in study['runtime_source_sha256'].items():
        if digest(path)!=sha:raise ValueError('Contact runtime source changed: '+path)
    fit=next(f for f in cfg['fits'] if f['fit']==group['fit']);policy=fit['policies'][group['regime']]
    assets=[args.config,cfgpath,Path(study['starts']),Path(fit['public']),Path(policy['checkpoint']),Path(cfg['perception'])]
    pins={str(p):digest(p) for p in assets}|study['runtime_source_sha256']
    if digest(policy['checkpoint'])!=policy['sha256'] or digest(cfg['perception'])!=cfg['perception_sha256']:raise ValueError('Weight digest changed')
    output=Path(study['output'])/group['id'];output.mkdir(parents=True,exist_ok=True);lock=output/'writer.lock'
    with lock.open('x') as stream:stream.write(str(os.getpid()))
    began=time.monotonic();before=[];aborted=None
    try:
        contract={'pins':pins,'group':group,'runtime':{'python':sys.version,'executable':sys.executable,'torch':str(torch.__version__),
            'numpy':np.__version__,'sfn':str(sfn.__file__)},'execution_mode':'dynamic_motors_fixed_grasps','seed':study['evaluation_seed']}
        provenance=output/'provenance.json';raw=output/'episodes.jsonl'
        if provenance.exists():
            if json.loads(provenance.read_text())!=contract:raise ValueError('Resume contract mismatch')
        else:
            if raw.exists():raise ValueError('Unpinned existing raw episodes')
            provenance.write_text(json.dumps(contract,indent=2));snapshot=output/'source_snapshot';snapshot.mkdir()
            for path in study['runtime_source_sha256']:shutil.copy2(path,snapshot/Path(path).name)
        seen=set()
        if raw.exists():
            for line in raw.read_text().splitlines(keepends=True):
                if not line.endswith('\n'):raise ValueError('Partial JSONL line')
                row=json.loads(line)
                if row['start_id'] in seen or row['group']!=group['id']:raise ValueError('Duplicate/mislabeled raw row')
                seen.add(row['start_id'])
        starts=json.loads(Path(study['starts']).read_text())['starts'][:group['count']]
        if not seen.issubset({s['id'] for s in starts}):raise ValueError('Unexpected saved ID')
        before=sorted(seen);model,saved=load_policy(policy['checkpoint'])
        if saved['regime']!=group['regime']:raise ValueError('Policy regime mismatch')
        public=json.loads(Path(fit['public']).read_text())
        sensor=RGBPoseSensor(segmentation_path=Path(cfg['perception']),crop=True)
        def allowed():return time.monotonic()-began<args.seconds and datetime.now(timezone.utc)<end
        for start in starts:
            if start['id'] in seen:continue
            if not allowed():break
            try:row=episode(start,group,study,model,public,sensor,allowed)
            except BudgetExpired:aborted=start['id'];break
            row['wall_seconds']=time.monotonic()-began
            with raw.open('a',encoding='utf-8') as stream:stream.write(json.dumps(row)+'\n');stream.flush()
            seen.add(start['id'])
            print(json.dumps({'group':group['id'],'n':len(seen),'id':start['id'],'success':row['success'],'reason':row['reason']}),flush=True)
        for path,sha in pins.items():
            if digest(path)!=sha:raise ValueError('Runtime changed during slice')
        summary={'group':group['id'],'n':len(seen),'target':group['count'],'added_ids':sorted(seen-set(before)),
            'aborted_budget_start_id':aborted,'window_end':args.window_end,'seconds':time.monotonic()-began}
        with (output/'invocations.jsonl').open('a') as f:f.write(json.dumps(summary)+'\n')
        (output/'progress.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
    finally:lock.unlink()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path('dual-arm-mas/configs/contact_study_v2.json'))
 p.add_argument('--group',required=True);p.add_argument('--seconds',type=float,default=240);p.add_argument('--window-end',required=True);run(p.parse_args())
