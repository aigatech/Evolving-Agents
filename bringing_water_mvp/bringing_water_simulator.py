import os
import sys
from pathlib import Path

# --- Resolve paths assuming BEHAVIOR-1K is a sibling of Evolving-Agents ---
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]                     # .../Evolving-Agents
SIBLING_BEHAVIOR_1K = REPO_ROOT.parent / "BEHAVIOR-1K"

# 1) Make sure we can import OmniGibson from the sibling repo if not installed
OG_SRC = SIBLING_BEHAVIOR_1K / "OmniGibson"
if OG_SRC.exists():
    sys.path.append(str(OG_SRC))

# 2) Point OG to the BDDL root that contains activity folders (e.g., bringing_water/)
#    Your file is: BEHAVIOR-1K/bddl/bddl/activity_definitions/bringing_water/problem0.bddl
BDDL_ROOT = SIBLING_BEHAVIOR_1K / "bddl" / "bddl" / "activity_definitions"
os.environ.setdefault("OG_BDDL_ROOT", str(BDDL_ROOT))  # BehaviorTask will look here

import yaml
import numpy as np
import torch
from PIL import Image

import omnigibson as og
from omnigibson.macros import gm

# Headless by default
gm.HEADLESS = True

CFG_PATH = THIS_DIR / "simulator.yaml"
OUT_DIR = THIS_DIR / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.array(x)

def _find_first_rgb(obs):
    """
    Find an RGB image in OG's nested observation dict.
    Prefer the external sensor, then fall back to robot cameras.
    """
    # Prefer external sensor if present
    try:
        rgb = obs["external"]["external_sensor0"]["rgb"]
        arr = _to_numpy(rgb)
        if arr.ndim == 3:
            return arr, "external / external_sensor0 / rgb"
    except Exception:
        pass

    # Try common robot camera keys
    for ent in obs.keys():
        if ent == "external":
            continue
        for cam_hint in ("Camera:0", "eyes:Camera:0", "zed_link:Camera:0", "head:Camera:0"):
            path = (ent, f"{ent}:{cam_hint}", "rgb")
            cur = obs
            ok = True
            for node in path:
                if not isinstance(cur, dict) or node not in cur:
                    ok = False
                    break
                cur = cur[node]
            if ok:
                arr = _to_numpy(cur)
                if isinstance(arr, np.ndarray) and arr.ndim == 3:
                    return arr, " / ".join(path)

    # DFS fallback
    def dfs(d, path):
        if isinstance(d, dict):
            for k, v in d.items():
                if k == "rgb":
                    arr = _to_numpy(v)
                    if isinstance(arr, np.ndarray) and arr.ndim == 3:
                        return arr, " / ".join(path + [k])
                found = dfs(v, path + [k])
                if found is not None:
                    return found
        return None

    found = dfs(obs, [])
    return found if found is not None else (None, None)

def main():
    cfg = yaml.safe_load(open(CFG_PATH, "r"))

    # Print where we're loading BDDL from (useful sanity check)
    print(f"OG_BDDL_ROOT = {os.environ.get('OG_BDDL_ROOT')}")

    env = og.Environment(cfg)

    # Build the world & get initial observation
    obs = env.reset()

    print(obs)

    rgb, src = _find_first_rgb(obs[0])
    if rgb is None:
        print("No RGB frame found in initial observation.")
    else:
        # Normalize to uint8
        if rgb.dtype != np.uint8:
            if rgb.max() <= 1.0:
                rgb = (rgb * 255.0).clip(0, 255).astype(np.uint8)
            else:
                rgb = rgb.clip(0, 255).astype(np.uint8)
        out_path = OUT_DIR / "initial.png"
        Image.fromarray(rgb).save(out_path)
        print(f"Saved initial image to {out_path} (source: {src})")

    og.shutdown()

if __name__ == "__main__":
    main()
