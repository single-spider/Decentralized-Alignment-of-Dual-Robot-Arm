"""Image-only sensing diagnostics on saved realizable poses and moving B."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from PIL import Image
from dual_arm_mas.world import DualArmWorld,wrap
from dual_arm_mas.vision import RGBPoseSensor

def run(args):
    args.out.mkdir(parents=True,exist_ok=True);rows=[]
    starts=json.loads(args.starts.read_text())["starts"][:args.count]
    with DualArmWorld() as world:
        world.attach_parts();sensor=RGBPoseSensor(segmentation_path=args.segmentation,crop=args.crop)
        for start in starts:
            world.restore(start["q"]);sensor.reset()
            for phase in range(3 if args.move_b else 1):
                if phase:
                    # Both arms have a measured contribution to the same error.
                    world.execute({"A":[-.0005,.0003,.008],"B":[.0005,-.0003,-.008]})
                rgb,result=sensor.observe(world);mask=result.pop("mask");truth=world.error()
                row={"start_id":start["id"],"phase":phase,**result,"true_error":truth.tolist()}
                if result["error"] is not None:
                    estimated=result["error"];difference=estimated-truth;difference[2]=wrap(difference[2])
                    row["error"]=estimated.tolist();row["sensing_error"]=difference.tolist()
                rows.append(row)
                if start["id"]<3 and phase==0:
                    Image.fromarray(rgb).save(args.out/f"rgb-{start['id']}.png")
                    Image.fromarray(np.array([[0,0,0],[30,90,230],[255,205,20]],dtype=np.uint8)[mask]).save(args.out/f"mask-{start['id']}.png")
    valid=[r for r in rows if r["valid"]];errors=np.array([r["sensing_error"] for r in valid])
    summary={"poses":len(rows),"valid":len(valid),"per_axis_mae_m_m_rad":np.mean(abs(errors),axis=0).tolist() if valid else None,
        "per_axis_p95_m_m_rad":np.quantile(abs(errors),.95,axis=0).tolist() if valid else None,
        "per_axis_max_m_m_rad":np.max(abs(errors),axis=0).tolist() if valid else None,
        "sensing_source":"actual RGB, "+("adapted SFN neural segmentation" if args.segmentation else "color segmentation")+", calibrated primitive-template pose; no body/depth/true-error input"}
    (args.out/"result.json").write_text(json.dumps({"summary":summary,"rows":rows},indent=2),encoding="utf-8")
    print(json.dumps(summary),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--starts",type=Path,required=True);p.add_argument("--count",type=int,default=24)
    p.add_argument("--segmentation",type=Path)
    p.add_argument('--crop',action='store_true')
    p.add_argument("--move-b",action="store_true");p.add_argument("--out",type=Path,required=True);run(p.parse_args())
