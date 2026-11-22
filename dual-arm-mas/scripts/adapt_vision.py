"""Small simulator-data adaptation of the EXISTING SFN RGB segmenter.

Body IDs are supervised TRAINING labels only. Deployment receives RGB only.
The position/yaw geometry is the calibrated dual-arm template adapter, since
the old fixed-receiver pose checkpoint was measured incompatible.
"""
from pathlib import Path
import argparse,json,sys,time,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
import torch
from dual_arm_mas.world import DualArmWorld
from dual_arm_mas.vision import RGBPoseSensor
from sfn.models.vsn import VirtualSensorNetwork
from train_rl import atomic_save,allowed_time


def collect(path,manifest_paths,count,seed):
    rng=np.random.default_rng(seed);images=[];masks=[];records=[]
    starts=[]
    for file in manifest_paths:
        starts.extend([(str(file),s) for s in json.loads(file.read_text())["starts"]])
    with DualArmWorld() as world:
        world.attach_parts();sensor=RGBPoseSensor(size=384)
        for index in range(count):
            if not allowed_time():break
            file,start=starts[index%len(starts)];world.restore(start['q'])
            # Beyond the saved initial distribution, observe both arms moving.
            if index>=len(starts):
                actions={n:rng.uniform(-1,1,3)*[.0015,.0015,.025] for n in ('A','B')}
                try:world.execute(actions)
                except ValueError:continue  # retain count/rejection in manifest below
            center,quat=world.expected_part_pose('B');yaw=world.p.getEulerFromQuaternion(quat)[2]
            c,s=np.cos(yaw),np.sin(yaw);R=np.array([[c,-s,0],[s,c,0],[0,0,1]])
            view=world.p.computeViewMatrix(np.asarray(center)+R@sensor.offset,center,[0,0,1])
            raw=world.p.getCameraImage(384,384,view,sensor.proj.T.flatten().tolist(),
                renderer=world.p.ER_TINY_RENDERER,physicsClientId=world.client_id)
            rgb=np.asarray(raw[2],dtype=np.uint8).reshape(384,384,4)[:,:,:3]
            ids=np.asarray(raw[4],dtype=np.int64).reshape(384,384)&((1<<24)-1)
            mask=np.zeros((384,384),dtype=np.uint8);mask[ids==world.parts['A']]=1;mask[ids==world.seam]=2
            images.append(rgb);masks.append(mask)
            records.append({'source_manifest':file,'start_id':start['id'],'actual_q':world.snapshot(),
                            'true_error_training_label_only':world.error().tolist()})
    np.savez_compressed(path,rgb=np.array(images),mask=np.array(masks))
    path.with_suffix('.json').write_text(json.dumps({'requested':count,'accepted':len(images),'rejected':count-len(images),
        'seed':seed,'label_source':'simulator body ID training labels, NEVER final execution observations','records':records},indent=2),encoding='utf-8')


def run(args):
    torch.set_num_threads(1);torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed)
    began=time.monotonic();args.out.mkdir(parents=True,exist_ok=True);device='cuda' if torch.cuda.is_available() else 'cpu'
    train_path=args.out/'train.npz';val_path=args.out/'validation.npz'
    if not train_path.exists():collect(train_path,[Path(f'dual-arm-mas/runs/train-{s}/starts.json') for s in [311,312,313]],args.count,args.seed)
    if not val_path.exists():collect(val_path,[Path('dual-arm-mas/runs/validation/starts.json')],24,args.seed+1)
    train=np.load(train_path);validation=np.load(val_path)
    original=Path('peg-in-hole-sfn/models/segmentation_mesh_v2_ce.pt')
    model=VirtualSensorNetwork.from_checkpoints(segmentation_path=original).segmentation
    model.requires_grad_(True);model.to(device);optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    epoch=0;updates=0;best=-1.
    if args.resume:
        saved=torch.load(args.resume,map_location='cpu',weights_only=False);model.load_state_dict(saved['model_state_dict']);optimizer.load_state_dict(saved['optimizer_state_dict'])
        epoch=saved['epoch'];updates=saved['global_step'];best=saved['best_validation_iou']
        rng.bit_generator.state=saved['numpy_rng'];torch.set_rng_state(saved['torch_rng'])
    counts=np.bincount(train['mask'].reshape(-1),minlength=3)
    weights=np.sqrt(counts.max()/np.maximum(counts,1));weights=np.minimum(weights,20)
    criterion=torch.nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=device))
    (args.out/'provenance.json').write_text(json.dumps({'seed':args.seed,'device':device,'torch_version':torch.__version__,
        'original_checkpoint':str(original),'original_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),
        'train_sha256':hashlib.sha256(train_path.read_bytes()).hexdigest(),'validation_sha256':hashlib.sha256(val_path.read_bytes()).hexdigest(),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'class_counts':counts.tolist(),'class_weights':weights.tolist()},indent=2))
    with (args.out/'training.jsonl').open('a',encoding='utf-8') as log:
        while epoch<args.epochs and time.monotonic()-began<args.seconds and allowed_time():
            model.train();losses=[]
            for indices in np.array_split(rng.permutation(len(train['rgb'])),max(1,len(train['rgb'])//4)):
                rgb=torch.tensor(train['rgb'][indices].copy(),device=device,dtype=torch.float32).permute(0,3,1,2)/255
                mask=torch.tensor(train['mask'][indices].copy(),device=device,dtype=torch.long)
                brightness=float(rng.uniform(.8,1.2));rgb=(rgb*brightness).clamp(0,1)
                loss=criterion(model(rgb),mask);optimizer.zero_grad();loss.backward()
                grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                if not torch.isfinite(loss) or not torch.isfinite(grad):raise RuntimeError('Nonfinite segmentation fine-tune')
                optimizer.step();updates+=1;losses.append(float(loss.detach()))
            model.eval();intersection=np.zeros(3);union=np.zeros(3)
            with torch.no_grad():
                for i in range(len(validation['rgb'])):
                    rgb=torch.tensor(validation['rgb'][i:i+1].copy(),device=device,dtype=torch.float32).permute(0,3,1,2)/255
                    predicted=model(rgb).argmax(1).cpu().numpy()[0];label=validation['mask'][i]
                    for k in range(3):intersection[k]+=np.sum((predicted==k)&(label==k));union[k]+=np.sum((predicted==k)|(label==k))
            iou=intersection/np.maximum(union,1);score=float(iou[1:].mean());epoch+=1
            row={'epoch':epoch,'updates':updates,'loss':float(np.mean(losses)),'validation_iou':iou.tolist(),'wall_seconds':time.monotonic()-began}
            log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
            state={'schema_version':1,'model_name':'dual_arm_rgb_segmentation_adaptation','model_config':{'in_channels':3,'classes':3,'base':16},
                'model_state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'optimizer_state_dict':optimizer.state_dict(),
                'epoch':epoch,'global_step':updates,'best_validation_iou':max(best,score),'numpy_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),
                'metrics':row,'data_split':{'training':'311/312/313 only','validation':29,'test':'untouched'},
                'label_source':'body IDs used only for supervised training/validation; deployed segmentation is RGB model forward'}
            atomic_save(args.out/'latest.pt',state)
            if score>best:best=score;atomic_save(args.out/'best.pt',state)
    print(f'RGB segmentation adaptation saved: {args.out}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--seed',type=int,default=702)
    p.add_argument('--count',type=int,default=192);p.add_argument('--epochs',type=int,default=20);p.add_argument('--seconds',type=float,default=240)
    p.add_argument('--resume',type=Path);run(p.parse_args())
