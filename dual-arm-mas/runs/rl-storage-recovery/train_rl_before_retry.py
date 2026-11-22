"""Bounded cooperative PPO training/evaluation on the measured coupled world."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
import torch
from dual_arm_mas.rl import CooperativePPO, load_policy, routed_features, central_features, actions_from_indices
from dual_arm_mas.episode import AlignmentEpisode
from dual_arm_mas.world import DualArmWorld
from dual_arm_mas.evidence import provenance
from sfn.training.rl_utils import compute_gae


def atomic_save(path, state):
    temporary=path.with_suffix(".tmp");torch.save(state,temporary);os.replace(temporary,path)


def allowed_time():
    return datetime.now(timezone(timedelta(hours=5,minutes=30))) < datetime.fromisoformat("2026-11-07T04:10:00+05:30")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(model, regime, public, starts, seed, output, stochastic=False):
    rows=[]
    with DualArmWorld() as world:
        env=AlignmentEpisode(world)
        for start in starts:
            if not allowed_time():
                break
            obs=env.reset(start["q"],seed=seed+start["id"]);trace=[]
            for _ in range(env.config.horizon):
                features,comm,_=routed_features(obs,regime,public,env.history)
                indices,_=model.choose(features,deterministic=not stochastic)
                obs,_,term,trunc,record=env.step(actions_from_indices(indices,obs),comm)
                record.pop("execution");record["policy_indices"]=indices.tolist();trace.append(record)
                if term or trunc:break
            rows.append({"start_id":start["id"],"success":trace[-1]["success"],"steps":len(trace),
                "final_error":world.error().tolist(),"cost_rad2":sum(r["motion_cost_rad2"] for r in trace),
                "joint_travel_rad":sum(r["joint_travel_rad"] for r in trace),
                "bytes":sum(r["communication"]["bytes"] for r in trace),
                "invalid_actions":sum(not r["feasible"] for r in trace),
                "reward":sum(r["reward"] for r in trace),"trace":trace})
    summary={"episodes":len(rows),"success_rate":np.mean([r["success"] for r in rows]).item() if rows else None,
             "mean_cost_rad2":np.mean([r["cost_rad2"] for r in rows]).item() if rows else None,
             "mean_steps":np.mean([r["steps"] for r in rows]).item() if rows else None,
             "invalid_actions":sum(r["invalid_actions"] for r in rows)}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({"summary":summary,"regime":regime,"stochastic":stochastic,"rows":rows},indent=2),encoding="utf-8")
    return summary


def run(args):
    torch.set_num_threads(1);torch.manual_seed(args.seed)
    public=json.loads(args.public.read_text());starts=json.loads(args.starts.read_text())["starts"]
    if args.eval_only:
        model,saved=load_policy(args.resume)
        summary=evaluate(model,saved["regime"],public,starts[:args.eval_count],args.seed,args.out/"evaluation.json")
        print(json.dumps({"fresh_process_reload":True,"checkpoint_sha256":digest(args.resume),**summary}),flush=True);return
    args.out.mkdir(parents=True,exist_ok=True)
    lock=args.out/"writer.lock"
    with lock.open("x") as f:f.write(str(os.getpid()))
    try:
        train(args,public,starts)
    finally:lock.unlink()


def train(args,public,starts):
    began=time.monotonic();rng=np.random.default_rng(args.seed)
    model=CooperativePPO();optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,eps=1e-5)
    total,update,episodes=0,0,0
    pins={str(p):digest(p) for p in [args.starts,args.public,Path(__file__),
        Path(__file__).parents[1]/"src/dual_arm_mas/rl.py",
        Path(__file__).parents[1]/"src/dual_arm_mas/world.py",
        Path(__file__).parents[1]/"src/dual_arm_mas/episode.py"]}
    if args.resume:
        saved=torch.load(args.resume,map_location="cpu",weights_only=False)
        if saved["pins"] != pins or saved["regime"] != args.regime:raise ValueError("Resume source/data/regime mismatch")
        model.load_state_dict(saved["model"]);optimizer.load_state_dict(saved["optimizer"])
        total,update,episodes=saved["steps"],saved["update"],saved["episodes"]
        rng.bit_generator.state=saved["numpy_rng"];torch.set_rng_state(saved["torch_rng"])
    else:
        (args.out/"provenance.json").write_text(json.dumps({"runtime":provenance(),"config":vars_json(args),
            "pins":pins,"algorithm":"shared categorical PPO actor, centralized scalar critic; GAE; no imitation",
            "reward_scale":.01},indent=2),encoding="utf-8")
        atomic_save(args.out/"initial.pt",{"model":model.state_dict(),"regime":args.regime,"steps":0})
    initial={k:v.detach().clone() for k,v in model.actor.state_dict().items()}
    with DualArmWorld() as world, (args.out/"training.jsonl").open("a",encoding="utf-8") as log:
        env=AlignmentEpisode(world)
        while total<args.steps and time.monotonic()-began<args.seconds and allowed_time():
            batch={k:[] for k in ("local","central","indices","logp","advantage","returns")}
            results=[];batch_steps=0
            while batch_steps<args.batch_steps and time.monotonic()-began<args.seconds and allowed_time():
                start=starts[int(rng.integers(len(starts)))];obs=env.reset(start["q"],seed=int(rng.integers(2**31)))
                trajectory=[];reward_total=0.;cost=0.
                for _ in range(env.config.horizon):
                    features,comm,_=routed_features(obs,args.regime,public,env.history)
                    central=central_features(env.central_state())
                    indices,logp=model.choose(features)
                    with torch.no_grad(): value=model.value(torch.tensor(central))
                    obs,reward,term,trunc,record=env.step(actions_from_indices(indices,obs),comm)
                    with torch.no_grad(): next_value=model.value(torch.tensor(central_features(env.central_state())))
                    trajectory.append((features,central,indices,logp,reward*.01,value,next_value,term,trunc))
                    reward_total+=reward;cost+=record["motion_cost_rad2"]
                    if term or trunc:break
                advantages=compute_gae([t[4] for t in trajectory],[t[5] for t in trajectory],
                    [t[6] for t in trajectory],[t[7] for t in trajectory],[t[8] for t in trajectory],.99,.95)
                for t,adv in zip(trajectory,advantages):
                    for name,val in zip(("local","central","indices","logp","advantage","returns"),
                        (t[0],t[1],t[2],t[3],float(adv),float(adv+t[5]))):batch[name].append(val)
                results.append({"success":record["success"],"steps":len(trajectory),"reward":reward_total,
                                "cost":cost,"invalid":not record["feasible"]})
                batch_steps+=len(trajectory);episodes+=1
            if not batch_steps:break
            tensors={k:torch.tensor(np.asarray(v),dtype=torch.long if k=="indices" else torch.float32) for k,v in batch.items()}
            advantages=tensors["advantage"];advantages=(advantages-advantages.mean())/advantages.std(unbiased=False).clamp_min(1e-6)
            losses=[];maxgrad=0.;kl=0.
            for epoch in range(args.epochs):
                for ix in torch.randperm(batch_steps).split(128):
                    dist=model.distribution(tensors["local"][ix]);newlogp=dist.log_prob(tensors["indices"][ix]).sum(-1)
                    ratio=(newlogp-tensors["logp"][ix]).exp();adv=advantages[ix,None]
                    actor=-torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv).mean()
                    value=model.value(tensors["central"][ix]);value_loss=(value-tensors["returns"][ix]).square().mean()
                    entropy=dist.entropy().sum(-1).mean();loss=actor+.5*value_loss-.01*entropy
                    if not torch.isfinite(loss):raise RuntimeError("Nonfinite PPO loss")
                    optimizer.zero_grad();loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),.5)
                    if not torch.isfinite(grad):raise RuntimeError("Nonfinite gradient")
                    optimizer.step();maxgrad=max(maxgrad,float(grad));losses.append(float(loss.detach()))
                    kl=float((tensors["logp"][ix]-newlogp).mean().detach())
                if kl>.03:break
            total+=batch_steps;update+=1
            delta=sum(float((v-initial[k]).square().sum()) for k,v in model.actor.state_dict().items())**.5
            row={"update":update,"steps":total,"episodes":episodes,"batch_episodes":len(results),
                "success_rate":float(np.mean([r["success"] for r in results])),
                "mean_reward":float(np.mean([r["reward"] for r in results])),"mean_cost_rad2":float(np.mean([r["cost"] for r in results])),
                "mean_steps":float(np.mean([r["steps"] for r in results])),"invalid_episodes":sum(r["invalid"] for r in results),
                "loss":float(np.mean(losses)),"max_preclip_gradient_norm":maxgrad,"last_approx_kl":kl,
                "actor_parameter_change_l2_this_invocation":delta,"wall_seconds":time.monotonic()-began}
            log.write(json.dumps(row)+"\n");log.flush()
            atomic_save(args.out/"latest.pt",{"model":model.state_dict(),"optimizer":optimizer.state_dict(),
                "regime":args.regime,"steps":total,"update":update,"episodes":episodes,"seed":args.seed,
                "numpy_rng":rng.bit_generator.state,"torch_rng":torch.get_rng_state(),"pins":pins,"config":vars_json(args)})
            print(json.dumps(row),flush=True)
    print(f"Checkpointed {total} steps; regime {args.regime}; {args.out}",flush=True)


def vars_json(args):return {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--regime",choices=["s1","s2","s3","s4"],default="s1")
    p.add_argument("--starts",type=Path,default=Path("dual-arm-mas/runs/train-311/starts.json"))
    p.add_argument("--public",type=Path,default=Path("dual-arm-mas/runs/train-311/public.json"))
    p.add_argument("--out",type=Path,required=True);p.add_argument("--seed",type=int,default=401)
    p.add_argument("--steps",type=int,default=10000);p.add_argument("--seconds",type=float,default=240)
    p.add_argument("--batch-steps",type=int,default=256);p.add_argument("--epochs",type=int,default=5)
    p.add_argument("--lr",type=float,default=.0003);p.add_argument("--resume",type=Path)
    p.add_argument("--eval-only",action="store_true");p.add_argument("--eval-count",type=int,default=24)
    run(p.parse_args())
