"""Fresh-process RGB/learned simulator replay with honest diagnostic overlays."""
from pathlib import Path
import argparse,json,sys,hashlib,shutil
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
import torch
import sfn
from PIL import Image,ImageDraw,ImageFont
from dual_arm_mas.world import DualArmWorld,wrap
from dual_arm_mas.episode import AlignmentEpisode,SCALE
from dual_arm_mas.rl import load_policy,routed_features
from dual_arm_mas.vision import RGBPoseSensor
from dual_arm_mas.policies import Protocol
from evaluate_learned import VisionObservationLoop
from train_rl import allowed_time


def replay(args):
    torch.set_num_threads(1);model,saved=load_policy(args.checkpoint)
    public=json.loads(args.public.read_text());starts=json.loads(args.starts.read_text())["starts"]
    args.out.mkdir(parents=True,exist_ok=True);frames=[];episodes=[]
    sources=[Path(__file__),Path(__file__).with_name('evaluate_learned.py'),Path(__file__).with_name('train_rl.py')]+list((Path(__file__).parents[1]/'src/dual_arm_mas').glob('*.py'))
    dependency=Path(sfn.__file__).parent
    sources += [dependency/r for r in ['panda/kinematics.py','training/common.py','models/segmentation.py','models/unet.py',
                                     'models/vsn.py','models/position.py','models/orientation.py']]
    h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assets=[args.checkpoint,args.starts,args.public]+([args.segmentation] if args.segmentation else [])
    pins={str(p):h(p) for p in sources+assets}
    for p in sources:
        dest=args.out/'source_snapshot'/p.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    try:font=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',18)
    except OSError:font=ImageFont.load_default()
    colors=np.array([[12,16,23],[30,90,230],[255,205,20]],dtype=np.uint8)
    with DualArmWorld() as world:
        env=AlignmentEpisode(world);sensor=RGBPoseSensor(segmentation_path=args.segmentation);loop=VisionObservationLoop(env,sensor)
        for start_id in [int(v) for v in args.ids.split(',')]:
            start=next(s for s in starts if s['id']==start_id)
            for method in (('half','learned','learned-mode') if args.include_mode_failure and start_id==int(args.ids.split(',')[0]) else ('half','learned')):
                if not allowed_time():break
                world.sync_parts=original_sync if 'original_sync' in locals() else world.sync_parts
                obs=env.reset(start['q'],seed=501+start_id);sensor.reset();protocol=Protocol(public)
                trace=[];active=False;counter=0;requested={n:np.zeros(3) for n in ('A','B')}
                comm={'bytes':0,'messages':0,'rounds':0}
                before=world.task_poses();last_estimate=np.zeros(3);status='initial';step=0
                initial_h={n:obs[n]['own_H'].tolist() for n in ('A','B')}
                def capture():
                    p=world.p;cid=world.client_id
                    view=p.computeViewMatrixFromYawPitchRoll([0,.05,.38],1.65,45,-30,0,2)
                    proj=p.computeProjectionMatrixFOV(50,1.5,.02,5)
                    raw=p.getCameraImage(960,640,view,proj,renderer=p.ER_TINY_RENDERER,physicsClientId=cid)
                    scene=Image.fromarray(np.asarray(raw[2],dtype=np.uint8).reshape(640,960,4)[:,:,:3]).resize((768,512))
                    rgb,_=sensor.render(world);mask=sensor.segment(rgb)
                    canvas=Image.new('RGB',(1280,800),'#101923');canvas.paste(scene,(0,65))
                    canvas.paste(Image.fromarray(rgb).resize((256,256)),(768,65))
                    canvas.paste(Image.fromarray(colors[mask]).resize((256,256)),(1024,65))
                    draw=ImageDraw.Draw(canvas)
                    draw.text((18,10),f'Dual-arm alignment | case {start_id} | {method} | {saved["regime"] if method.startswith("learned") else "fixed 50/50"}',fill='white',font=font)
                    draw.text((780,330),'Actual camera RGB      '+('SFN neural regions' if args.segmentation else 'RGB color regions'),fill='white',font=font)
                    draw.text((780,355),'Camera mounted to B encoder/FK frame',fill='white',font=font)
                    draw.text((780,383),'No body-ID/depth/true-error sensing',fill='white',font=font)
                    draw.text((780,411),f'Initial Hxx x1000: A {initial_h["A"][0][0]*1000:.3f} B {initial_h["B"][0][0]*1000:.3f}',fill='white',font=font)
                    draw.text((780,439),'A holds blue peg; B holds green fixture',fill='white',font=font)
                    draw.text((780,467),'Mean learned categorical fractions' if method=='learned' else ('MODAL decoder failure condition' if method=='learned-mode' else 'Matched RGB baseline controller'),fill='white',font=font)
                    draw.text((780,495),f'Payload: {comm["bytes"]} bytes / {comm["messages"]} messages / {comm["rounds"]} rounds',fill='white',font=font)
                    draw.text((780,523),'A/B share PPO weights; local role inputs' if method.startswith('learned') else 'A/B each apply a public fixed half share',fill='white',font=font)
                    truth=world.error();poses=world.task_poses()
                    draw.text((18,590),f'Step {step:02d} | {status} | kinematic / zero gravity / no contact',fill='white',font=font)
                    draw.text((18,620),'LAST pre-action VISION estimate (mm,mm,deg): '+fmt(last_estimate),fill='#ffd04a',font=font)
                    draw.text((18,650),'CURRENT TRUE error (DIAGNOSTIC ONLY):       '+fmt(truth),fill='#a8bac9',font=font)
                    for i,n in enumerate(('A','B')):
                        measured=poses[n]-before[n];measured[2]=wrap(measured[2])
                        draw.text((18,680+30*i),f'{n}: requested '+fmt(requested[n])+' | measured '+fmt(measured),fill='white',font=font)
                    draw.text((18,750),'Playback slowed: one frame per 6/240 s simulated; 200 ms display. No motion scaling.',fill='#a8bac9',font=font)
                    frames.append(canvas)
                original_sync=world.sync_parts
                def hook():
                    nonlocal counter
                    original_sync()
                    if active:
                        counter+=1
                        if counter%6==0:capture()
                world.sync_parts=hook
                for step in range(env.config.horizon):
                    obs,_,perception=loop.replace(obs)
                    if obs is None:status='FAILURE: invalid vision';capture();break
                    last_estimate=perception['error'].copy();before=world.task_poses()
                    if method.startswith('learned'):
                        features,comm,_=routed_features(obs,saved['regime'],public,env.history)
                        with torch.no_grad():
                            dist=model.distribution(torch.tensor(features))
                            fraction=(dist.probs@torch.linspace(0,1,5)).numpy() if method=='learned' else dist.logits.argmax(-1).numpy()*.25
                        requested={n:fraction[i]*np.clip(obs[n]['error']/SCALE,-1,1)*SCALE*(-1 if n=='A' else 1)
                                   for i,n in enumerate(('A','B'))}
                    else:requested,comm=protocol.actions('half',obs,env.history)
                    if step==0:capture()
                    active=True;status='executing simultaneous commands'
                    obs,_,term,trunc,record=env.step(requested,comm)
                    active=False;record.pop('execution');record['vision_error_used']=last_estimate.tolist();trace.append(record)
                    if term or trunc:
                        status='SUCCESS: measured tolerance reached' if record['success'] else 'FAILURE: '+str(record['invalid_reason'] or 'timeout')
                        capture();frames.extend([frames[-1]]*5);break
                episodes.append({'start_id':start_id,'method':method,'regime':saved['regime'],'status':status,
                    'initial_own_H':initial_h,'trace':trace,'true_final_error':world.error().tolist()})
                world.sync_parts=original_sync
    if frames:frames[0].save(args.out/'learned-vision.gif',save_all=True,append_images=frames[1:],duration=200,loop=0)
    if pins!={str(p):h(p) for p in sources+assets}:raise RuntimeError('Demo source/data/checkpoint changed during capture')
    (args.out/'replay.json').write_text(json.dumps({'pins':pins,'config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'checkpoint_sha256':h(args.checkpoint),'gif_sha256':h(args.out/'learned-vision.gif'),
        'selection':'Explicit validation IDs; paired baseline/learned starts. No trajectories authored or motion enlarged.',
        'episodes':episodes,'frames':len(frames)},indent=2),encoding='utf-8')
    print(json.dumps({'frames':len(frames),'episodes':[{'case':e['start_id'],'method':e['method'],'status':e['status']} for e in episodes]}),flush=True)


def fmt(error):return f'[{error[0]*1000:+7.3f}, {error[1]*1000:+7.3f}, {np.rad2deg(error[2]):+7.3f}]'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--starts',type=Path,default=Path('dual-arm-mas/runs/validation/starts.json'))
    p.add_argument('--public',type=Path,default=Path('dual-arm-mas/runs/train-311/public.json'))
    p.add_argument('--segmentation',type=Path)
    p.add_argument('--ids',default='0,4');p.add_argument('--include-mode-failure',action='store_true');replay(p.parse_args())
