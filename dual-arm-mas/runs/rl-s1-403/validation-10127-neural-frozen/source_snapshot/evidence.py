"""Compact measured evidence with editable-runtime provenance."""
import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image


def provenance():
    import sfn
    root = Path(__file__).resolve().parents[3]
    project = Path(__file__).resolve().parents[2]
    files = (list(Path(__file__).parent.glob("*.py")) + list((project / "scripts").glob("*.py"))
             + list((project / "configs").glob("*.json")) + list((Path(sfn.__file__).parent / "panda").glob("*.py")))
    return {
        "time_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
        "executable": sys.executable, "sfn_path": sfn.__file__,
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "dirty": subprocess.check_output(["git", "status", "--short"], cwd=root, text=True),
        "versions": {name: importlib.metadata.version(name) for name in ["numpy", "pybullet", "Pillow"]},
        "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    }


def save_result(out, result):
    result.setdefault("provenance", provenance())
    (out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return {"passed": result["passed"], "result": str((out / "result.json").resolve())}


def scene_image(p, cid, path, target=(0, 0, 0.4), distance=1.65):
    view = p.computeViewMatrixFromYawPitchRoll(target, distance, 45, -30, 0, 2)
    projection = p.computeProjectionMatrixFOV(50, 1.5, 0.02, 5)
    pixels = p.getCameraImage(960, 640, view, projection, renderer=p.ER_TINY_RENDERER, physicsClientId=cid)[2]
    frame = Image.fromarray(np.asarray(pixels, dtype=np.uint8).reshape(640, 960, 4)[:, :, :3])
    frame.save(path)
    return frame
