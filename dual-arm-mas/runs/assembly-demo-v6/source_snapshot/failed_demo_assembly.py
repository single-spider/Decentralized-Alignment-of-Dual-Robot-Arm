"""Actual saved-RL/neural-RGB alignment -> dynamic guarded insertion pilot.

New contact evidence only; does not inherit alignment-study performance claims.
Vision template assumes pre-insertion height, so no template estimates are used
in axial descent: insertion uses measured contact/depth safeguards instead.
"""
import argparse,hashlib,json,sys,time,shutil
from pathlib import Path
from datetime import datetime,timezone
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from dual_arm_mas.contact_world_v2 import ContactWorld,insert
from dual_arm_mas.episode import AlignmentEpisode,SCALE
from dual_arm_mas.rl import load_policy,routed_features
from dual_arm_mas.vision import RGBPoseSensor
from evaluate_learned import VisionObservationLoop


def run(args):
    end=datetime.fromisoformat(args.window_end)
    if end.tzinfo is None or datetime.now(timezone.utc)>=end:raise ValueError('Future timezone-aware window required')
    args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);began=time.monotonic()
    cfg=json.loads(Path('dual-arm-mas/configs/learned_study.json').read_text());policy=cfg['fits'][0]['policies']['s1']
    digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    for path,sha in cfg['source_sha256'].items():
        if digest(path)!=sha:raise ValueError('Frozen alignment source mismatch: '+path)
    assert digest(policy['checkpoint'])==policy['sha256'] and digest(cfg['perception'])==cfg['perception_sha256']
    model,saved=load_policy(policy['checkpoint']);public=json.loads(Path(cfg['fits'][0]['public']).read_text())
    manifest=Path('dual-arm-mas/runs/validation/starts.json');starts=json.loads(manifest.read_text())['starts']
    sensor=RGBPoseSensor(segmentation_path=Path(cfg['perception']),crop=True);episodes=[];frames=[]
    font=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',16)
    def allowed():return time.monotonic()-began<args.seconds and datetime.now(timezone.utc)<end
    for start_id,blocked,method in [(0,False,'half'),(0,False,'learned'),(4,False,'learned'),(0,True,'learned')]:
        if not allowed():break
        start=next(s for s in starts if s['id']==start_id)
        with ContactWorld() as world:
            world.attach_parts(blocked=blocked);env=AlignmentEpisode(world);env.reset(start['q'],seed=501+start_id)
            settle=world.advance({n:world.q(n) for n in world.robots},240)
            loop=VisionObservationLoop(env,sensor);sensor.reset();trace=[];last_rgb=None;last_mask=None;last_est=None;stage='ALIGNMENT';status='running'
            initial_h={n:o['own_H'].tolist() for n,o in env.observations().items()}
            def part_vector(n):
                pos,quat=world.part_pose(n);return np.array([*pos,world.p.getEulerFromQuaternion(quat)[2]])
            previous_parts={n:part_vector(n) for n in ('A','B')}
            requested={n:np.zeros(4) for n in ('A','B')};measured={n:np.zeros(4) for n in ('A','B')}
            def update_contributions(commands):
                nonlocal previous_parts,requested,measured
                current={n:part_vector(n) for n in ('A','B')};requested=commands
                measured={n:current[n]-previous_parts[n] for n in current}
                for value in measured.values():value[3]=(value[3]+np.pi)%(2*np.pi)-np.pi
                previous_parts=current
            def fmt(v):return np.array2string(np.round(np.asarray(v)*[1000,1000,1000,180/np.pi],3),precision=3)
            palette=np.array([[12,16,23],[30,90,230],[255,205,20]],dtype=np.uint8)
            def capture(record=None):
                p=world.p;cid=world.client_id
                view=p.computeViewMatrixFromYawPitchRoll([0,.05,.38],1.6,45,-30,0,2)
                proj=p.computeProjectionMatrixFOV(50,1.5,.02,5)
                scene=p.getCameraImage(900,600,view,proj,renderer=p.ER_TINY_RENDERER,physicsClientId=cid)
                canvas=Image.new('RGB',(1200,900),'#101923');canvas.paste(Image.fromarray(np.asarray(scene[2],dtype=np.uint8).reshape(600,900,4)[:,:,:3]),(0,60))
                # Zoom is a real second rendered view, not scaled physical motion.
                hole,_=world.part_pose('B');zoom=p.computeViewMatrix((hole+np.array([.10,-.13,.01])).tolist(),hole.tolist(),[0,0,1])
                near=p.getCameraImage(300,300,zoom,p.computeProjectionMatrixFOV(40,1,.001,1),renderer=p.ER_TINY_RENDERER,physicsClientId=cid)
                canvas.paste(Image.fromarray(np.asarray(near[2],dtype=np.uint8).reshape(300,300,4)[:,:,:3]),(900,60))
                if last_rgb is not None:canvas.paste(Image.fromarray(last_rgb).resize((150,150)),(900,385))
                if last_mask is not None:canvas.paste(Image.fromarray(palette[last_mask]).resize((150,150)),(1050,385))
                state=world.insertion_state();draw=ImageDraw.Draw(canvas)
                estimate='unavailable' if last_est is None else str(np.round(last_est*[1000,1000,180/np.pi],3))
                lines=[f'Contact assembly | validation {start_id} | {method} | blocked={blocked} | {stage} | {status}',
                    'Dynamic joint MOTORS | gravity -9.81 | collision parts | fixed grasps | state-preserving IK',
                    f"Measured tip depth {state['depth_m']*1000:+.2f} mm | normal force {state['normal_force_n']:.2f} N | tick {world.tick}",
                    f"CURRENT true XY/yaw DIAGNOSTIC: {np.round(np.array(state['error_m_m_rad'])*[1000,1000,180/np.pi],3)}",
                    'LAST pre-action VISION estimate (mm,mm,deg): '+estimate,
                    'A requested '+fmt(requested['A'])+' | measured '+fmt(measured['A'])+' [mm,mm,mm,deg]',
                    'B requested '+fmt(requested['B'])+' | measured '+fmt(measured['B'])+' [mm,mm,mm,deg]',
                    ('A/B: frozen shared S1 PPO, local actors' if method=='learned' else 'A/B: public fixed-half RGB baseline')+f" | 0B teammate payload | initial Hxx*1000 A/B {initial_h['A'][0][0]*1000:.3f}/{initial_h['B'][0][0]*1000:.3f}",
                    'Axial stage: 2 N normal-force admittance, 8 N hard trigger + retract; NOT axial RL.',
                    'RGB/NN insets are LAST alignment image during descent; fixed-height vision not reused.',
                    'Endpoint replay: alignment approx 1x / insertion approx 2x; 400ms/frame, no motion scaling.']
                draw.text((905,365),'LAST RGB       NN regions',font=font,fill='white')
                draw.text((12,15),lines[0],font=font,fill='white')
                for i,line in enumerate(lines[1:]):draw.text((12,670+i*20),line,font=font,fill='white')
                frames.append(canvas)
            capture()
            failure=None
            if settle['stop_reason']:failure='settle_'+settle['stop_reason']
            for step in range(20):
                if failure or not allowed():break
                obs,rgb,perception=loop.replace(env.observations());last_rgb=rgb;last_mask=perception.get('mask')
                if obs is None:
                    failure='perception: '+str(perception.get('reason'));Image.fromarray(rgb).save(args.out/f'failed-rgb-{start_id}-{int(blocked)}.png');break
                last_est=np.asarray(perception['error'])
                if np.all(abs(last_est)<=np.array([.0008,.0008,np.deg2rad(.8)])):break
                if method=='half':fractions=np.full((2,3),.5)
                else:
                    features,comm,_=routed_features(obs,'s1',public,env.history)
                    with torch.no_grad():fractions=(model.distribution(torch.tensor(features)).probs@torch.linspace(0,1,5)).numpy()
                actions={n:fractions[i]*np.clip(obs[n]['error']/SCALE,-1,1)*SCALE*(-1 if n=='A' else 1) for i,n in enumerate(('A','B'))}
                # Do not terminate this pilot solely on the old loose true-error
                # alignment terminal; final transition is gated below explicitly.
                env.done=False
                _,_,term,trunc,record=env.step(actions,comm);record.pop('execution');record['actor_error']=last_est.tolist();record['fractions']=fractions.tolist();trace.append(record)
                if not record['feasible']:failure=record['invalid_reason'];break
                update_contributions({n:np.array([v[0],v[1],0,v[2]]) for n,v in actions.items()})
                capture()
            if not allowed():failure='time_budget'
            # Refresh final image before phase transition (no stale pose gate).
            if not failure:
                obs,rgb,perception=loop.replace(env.observations());last_rgb=rgb;last_mask=perception.get('mask')
                if obs is None:failure='perception'
                elif not np.all(abs(perception['error'])<=np.array([.001,.001,np.deg2rad(1)])):failure='vision_alignment_gate'
                else:last_est=np.asarray(perception['error'])
            alignment={'success':failure is None,'failure':failure,'trace':trace,'true_final_error_diagnostic':world.error().tolist(),
                       'last_estimate':None if last_est is None else last_est.tolist()}
            if failure:insertion={'success':False,'reason':'alignment_failed','trace':[],'final':world.insertion_state()}
            else:
                stage='INSERTION'
                def observer(record):
                    update_contributions({'A':np.array([0,0,-world.contact_config.insertion_increment_m,0]),'B':np.zeros(4)})
                    if record['attempt']%2==0 or record['stop_reason']:capture(record)
                # Clock is checked per descent via wrapper, not just per episode.
                original=world.descend
                def bounded_descent():
                    if not allowed():raise ValueError('time budget exhausted')
                    return original()
                world.descend=bounded_descent
                insertion=insert(world,observer=observer)
            status=insertion['reason'];stage='TERMINAL'
            recovery=insertion.get('recovery',{})
            if recovery.get('attempted'):status+=' / '+('RELEASED' if recovery.get('released') else 'RECOVERY_STOPPED')
            update_contributions({'A':np.array([0,0,world.contact_config.recovery_distance_m if recovery.get('attempted') else 0,0]),'B':np.zeros(4)})
            capture()
            row={'start_id':start_id,'blocked':blocked,'method':method,'initial_own_H':initial_h,'alignment':alignment,'settle':settle,'insertion':insertion}
            episodes.append(row);(args.out/f'episode-{start_id}-{int(blocked)}-{method}.json').write_text(json.dumps(row,indent=2))
            print(json.dumps({'id':start_id,'blocked':blocked,'method':method,'alignment':alignment['success'],'insertion':insertion['success'],
                'reason':insertion['reason'],'depth_mm':insertion['final']['depth_m']*1000}),flush=True)
    if frames:frames[0].save(args.out/'contact-assembly.gif',save_all=True,append_images=frames[1:],duration=400,loop=0)
    paths=[Path(__file__),Path('dual-arm-mas/src/dual_arm_mas/contact_world.py'),Path('dual-arm-mas/src/dual_arm_mas/contact_world_v2.py'),manifest,Path(policy['checkpoint']),Path(cfg['perception']),
        Path('dual-arm-mas/configs/learned_study.json'),Path(cfg['fits'][0]['public'])]
    for source in [Path(__file__),Path('dual-arm-mas/src/dual_arm_mas/contact_world.py'),Path('dual-arm-mas/src/dual_arm_mas/contact_world_v2.py')]:
        dest=args.out/'source_snapshot'/source.name;dest.parent.mkdir(exist_ok=True);shutil.copy2(source,dest)
    result={'scope':'Four validation demonstration episodes only; not held-out contact study. Frozen alignment policy transferred, axial insertion deterministic.',
        'completed_episodes':len(episodes),'expected_episodes':4,'frames':len(frames),'seconds':time.monotonic()-began,
        'source_asset_sha256':{str(p):digest(p) for p in paths},'frozen_alignment_source_sha256':cfg['source_sha256'],
        'episodes':[{k:v for k,v in e.items() if k not in ['alignment','insertion']}|{'alignment_success':e['alignment']['success'],
        'insertion_success':e['insertion']['success'],'reason':e['insertion']['reason']} for e in episodes],
        'window_end':args.window_end,'limitations':['Fixed constraints are assumed grasps, not frictional grasp validation.','Force threshold stops next simulation tick; peak can exceed threshold.','No neural pose used after axial descent starts; template fixed-height assumption invalid then.']}
    (args.out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({'episodes':len(episodes),'frames':len(frames),'seconds':time.monotonic()-began}))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--seconds',type=float,default=240);p.add_argument('--window-end',required=True);run(p.parse_args())
