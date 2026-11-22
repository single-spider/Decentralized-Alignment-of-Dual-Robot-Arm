"""Matched GT/RGB learned-policy execution; no true-error sensing fallback."""
from pathlib import Path
import argparse,hashlib,json,sys,time,os,shutil
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
import torch
from PIL import Image
from dual_arm_mas.rl import load_policy,routed_features,actions_from_indices
from dual_arm_mas.vision import RGBPoseSensor
from dual_arm_mas.world import DualArmWorld,wrap
from dual_arm_mas.episode import AlignmentEpisode,EpisodeConfig,SCALE
from train_rl import allowed_time


class VisionObservationLoop:
    def __init__(self,env,sensor,occlusion=0.):self.env,self.sensor,self.occlusion=env,sensor,occlusion

    def replace(self,obs):
        if self.occlusion:
            rgb,yaw=self.sensor.render(self.env.world)
            # Actual RGB occlusion, not corruption of the geometric answer.
            size=rgb.shape[0];side=max(1,int(size*np.sqrt(self.occlusion)))
            x=(size-side)//2;y=int(size*.25)
            rgb[y:y+side,x:x+side]=0
            result=self.sensor.estimate(rgb,yaw);result['rgb_occlusion_fraction']=self.occlusion
        else:rgb,result=self.sensor.observe(self.env.world)
        if not result["valid"]:return None,rgb,result
        error=result["error"].copy()
        # The episode writes history during its physics step. Replace the newly
        # observed endpoint BEFORE exposing history to either S3 actor.
        self.env.sensed_error=error
        for n in ("A","B"):
            obs[n]["error"]=error.copy()
            if self.env.history[n]:self.env.history[n][-1]["error_after"]=error.tolist()
        return obs,rgb,result


def run(args):
    torch.set_num_threads(1)
    checkpoint_hash=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    model,saved=load_policy(args.checkpoint);public=json.loads(args.public.read_text())
    starts=json.loads(args.starts.read_text())["starts"][:args.count]
    args.out.mkdir(parents=True,exist_ok=True);rows=[];began=time.monotonic()
    sources=[Path(__file__),Path(__file__).with_name('train_rl.py')]+list((Path(__file__).parents[1]/'src/dual_arm_mas').glob('*.py'))
    assets=[args.starts,args.public]+([args.segmentation] if args.segmentation else [])
    pins={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources+assets}
    contract={"pins":pins,"checkpoint_sha256":checkpoint_hash,"regime":saved['regime'],"sensing":args.sensing,
              "decoder":args.decoder,"seed":args.seed,"count":args.count,
              "drop":args.drop,"delay":args.delay,"b_limit_factor":args.b_limit_factor,"occlusion":args.occlusion,
              "history_length":args.history_length}
    meta=args.out/'provenance.json';raw=args.out/'episodes.jsonl'
    if meta.exists():
        if json.loads(meta.read_text())!=contract:raise ValueError('Evaluation source/data/checkpoint/config changed')
    else:
        if raw.exists():raise ValueError('Existing unpinned episodes: choose a new output')
        meta.write_text(json.dumps(contract,indent=2),encoding='utf-8')
        for p in sources:
            dest=args.out/'source_snapshot'/p.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    seen=set()
    if raw.exists():
        with raw.open(encoding='utf-8') as f:
            for line in f:
                if not line.endswith('\n'):raise ValueError('Partial episode line')
                r=json.loads(line)
                if r['start_id'] in seen:raise ValueError('Duplicate learned episode')
                seen.add(r['start_id']);rows.append(r)
    with DualArmWorld() as world:
        env=AlignmentEpisode(world,EpisodeConfig(b_limit_factor=args.b_limit_factor));sensor=RGBPoseSensor(segmentation_path=args.segmentation);loop=VisionObservationLoop(env,sensor,args.occlusion)
        for start in starts:
            if start['id'] in seen:continue
            if not allowed_time() or time.monotonic()-began>args.seconds:break
            sensor.reset();obs=env.reset(start["q"],seed=args.seed+start["id"]);trace=[];sensing=[];failure=None
            previous_payload={};message_rng=np.random.default_rng(args.seed+start['id']+1)
            for step in range(env.config.horizon):
                if args.sensing=="rgb":
                    obs,rgb,perception=loop.replace(obs)
                    truth=world.error();diagnostic={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in perception.items() if k!="mask"}
                    diagnostic["true_error_diagnostic_only"]=truth.tolist()
                    if perception["valid"]:
                        d=perception["error"]-truth;d[2]=wrap(d[2]);diagnostic["sensing_error"]=d.tolist()
                    sensing.append(diagnostic)
                    if args.images and start["id"]<3:
                        Image.fromarray(rgb).save(args.out/f"rgb-{start['id']}-{step}.png")
                        colors=np.array([[0,0,0],[30,90,230],[255,205,20]],dtype=np.uint8)
                        Image.fromarray(colors[perception["mask"]]).save(args.out/f"mask-{start['id']}-{step}.png")
                    if obs is None:failure="perception_invalid";break
                histories={n:(h[-args.history_length:] if args.history_length else []) for n,h in env.history.items()}
                own_h={n:o['own_H'].tolist() for n,o in obs.items()}
                features,comm,sent=routed_features(obs,saved["regime"],public,histories,drop=args.drop,
                    rng=message_rng,delayed=previous_payload if args.delay else None)
                previous_payload=sent
                if args.decoder=='mode':
                    indices,_=model.choose(features,deterministic=True)
                    actions=actions_from_indices(indices,obs);fractions=indices*.25
                else:
                    # Deterministic mean of each learned categorical action.
                    # This is a declared deployment rule, not a cost allocator,
                    # safety fallback, or a view of the other actor's decision.
                    with torch.no_grad():
                        fractions=(model.distribution(torch.tensor(features)).probs @ torch.linspace(0,1,5)).numpy()
                    actions={n:fractions[i]*np.clip(obs[n]['error']/SCALE,-1,1)*SCALE*(-1 if n=='A' else 1)
                             for i,n in enumerate(('A','B'))}
                actor_errors={n:o["error"].tolist() for n,o in obs.items()}
                obs,_,term,trunc,record=env.step(actions,comm)
                record.pop("execution");record["actor_error"]=actor_errors;record["policy_fractions"]=fractions.tolist()
                record['own_H_diagnostic_only']=own_h;trace.append(record)
                if saved['regime']=='s3':
                    # Current executed teammate contribution is a scoring label
                    # only, never an actor input. The actor sees its own past.
                    prediction={}
                    for i,n in enumerate(('A','B')):
                        other='B' if n=='A' else 'A';c=np.clip(np.array(actor_errors[n])/SCALE,-1,1)*SCALE
                        valid=abs(c)>SCALE*.05;target=np.full(3,np.nan)
                        target[valid]=np.array(record['executed'][other])[valid]/c[valid]*(1 if n=='A' else -1)
                        prediction[n]={'predicted_fraction':features[i,26:29].tolist(),
                            'executed_fraction_label':[float(v) if np.isfinite(v) else None for v in target]}
                    record['prediction_diagnostic_only']=prediction
                if term or trunc:break
            success=bool(trace and trace[-1]["success"] and failure is None)
            rows.append({"start_id":start["id"],"regime":saved["regime"],"sensing":args.sensing,"success":success,
                "failure":failure or (trace[-1]["invalid_reason"] if trace and not trace[-1]["feasible"] else None) or (None if success else "timeout"),
                "final_error":world.error().tolist(),"steps":len(trace),"motion_cost_rad2":sum(t["motion_cost_rad2"] for t in trace),
                "joint_travel_rad":sum(t["joint_travel_rad"] for t in trace),"bytes":sum(t["communication"]["bytes"] for t in trace),
                "messages":sum(t["communication"]["messages"] for t in trace),"rounds":sum(t["communication"]["rounds"] for t in trace),
                "invalid_actions":sum(not t["feasible"] for t in trace),"trace":trace,"sensing_diagnostics":sensing})
            # Each episode is persisted; a bounded invocation cannot erase it.
            with (args.out/"episodes.jsonl").open("a",encoding="utf-8") as stream:stream.write(json.dumps(rows[-1])+"\n")
    summary={"episodes":len(rows),"success_rate":float(np.mean([r["success"] for r in rows])) if rows else None,
             "mean_cost_rad2":float(np.mean([r["motion_cost_rad2"] for r in rows])) if rows else None,
             "mean_steps":float(np.mean([r["steps"] for r in rows])) if rows else None,
             "failures":{k:sum(r["failure"]==k for r in rows) for k in {r["failure"] for r in rows} if k is not None},
             "checkpoint_sha256":checkpoint_hash,"sensing":args.sensing,"decoder":args.decoder,
             "regime":saved["regime"],"wall_seconds":time.monotonic()-began}
    if pins!={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources+assets}:
        raise RuntimeError('Source changed during learned evaluation')
    (args.out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8");print(json.dumps(summary),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--sensing",choices=["gt","rgb"],required=True)
    p.add_argument("--public",type=Path,default=Path("dual-arm-mas/runs/train-311/public.json"))
    p.add_argument("--starts",type=Path,default=Path("dual-arm-mas/runs/validation/starts.json"));p.add_argument("--count",type=int,default=24)
    p.add_argument("--out",type=Path,required=True);p.add_argument("--seed",type=int,default=501);p.add_argument("--seconds",type=float,default=240)
    p.add_argument("--decoder",choices=["mode","mean"],default="mean")
    p.add_argument('--drop',type=float,default=0.);p.add_argument('--delay',type=int,choices=[0,1],default=0)
    p.add_argument('--b-limit-factor',type=float,default=1.);p.add_argument('--occlusion',type=float,default=0.)
    p.add_argument('--history-length',type=int,choices=[0,1,4],default=4)
    p.add_argument('--segmentation',type=Path)
    p.add_argument("--images",action="store_true")
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True);lock=args.out/'writer.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:run(args)
    finally:lock.unlink()
