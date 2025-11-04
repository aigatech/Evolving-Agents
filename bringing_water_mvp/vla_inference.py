#!/usr/bin/env python3
"""
OpenVLA-style inference that outputs a 7-D action vector per image.

Edit the CONFIG block below, then run:
  python vla_inference.py

- Supports a single image path (e.g., "../images/frame.jpg")
  or a glob (e.g., "../images/*.jpg").
- Prints each result and also saves to actions.csv in this folder.

Expected 7-vector (order):
[Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]
"""

import os
import glob
import csv
import sys
from typing import List

import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

# =========================
# ====== CONFIG HERE ======
# =========================
# Path(s) to image(s). Use a single file path or a glob. The images folder is a sibling to this script.
# Examples:
#   IMAGE_INPUT = "../images/frame.jpg"
#   IMAGE_INPUT = "../images/*.png"
IMAGE_INPUT = "images/initial.png"

# Natural language instruction for the robot's action.
INSTRUCTION = "What action should the robot take to bring all the bottles of water from the refrigerator in the kitchen to the coffee table in the living room and close the refrigerator door?"

# Hugging Face model id. (Swap to a MiniVLA HF id when that’s available in the same API.)
MODEL_ID = "openvla/openvla-7b"

# Device/dtype. If you hit issues with bfloat16 on your GPU, try "float16" or "float32".
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = "bfloat16"  # "bfloat16" | "float16" | "float32"

# Optional un-normalization key (e.g., "bridge_orig") for de-normalized actions.
UNNORM_KEY = "bridge_orig"  # e.g., "bridge_orig" or keep None

# Deterministic (False => greedy). Leave False for deterministic outputs.
DO_SAMPLE = False

# Save results here
OUTPUT_CSV = "actions.csv"
# =========================
# ==== END OF CONFIG ======
# =========================


def _dtype_from_string(s: str):
    s = s.lower()
    if s == "bfloat16":
        return torch.bfloat16
    if s == "float16":
        return torch.float16
    if s == "float32":
        return torch.float32
    raise ValueError(f"Unsupported DTYPE: {s}")


def load_model_and_processor(model_id: str, device: str, dtype: torch.dtype):
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    return model, processor


def predict_action_for_image(
    model,
    processor,
    image_path: str,
    instruction: str,
    device: str,
    dtype: torch.dtype,
    unnorm_key: str = None,
    do_sample: bool = False,
) -> List[float]:
    img = Image.open(image_path).convert("RGB")
    prompt = f"In: What action should the robot take to {instruction}?\nOut:"
    inputs = processor(prompt, img).to(device, dtype=dtype)

    # predict_action is implemented in the model via trust_remote_code
    with torch.no_grad():
        action = model.predict_action(
            **inputs,
            unnorm_key=unnorm_key,
            do_sample=do_sample
        )

    # Some builds return a dict like {"action": np.ndarray}
    if isinstance(action, dict) and "action" in action:
        action = action["action"]

    # Normalize to a 1D numpy array
    if torch.is_tensor(action):
        arr = action.detach().cpu().numpy()
    elif isinstance(action, np.ndarray):
        arr = action
    elif isinstance(action, (list, tuple)):
        arr = np.asarray(action)
    else:
        raise TypeError(f"Unsupported action type: {type(action)}")

    arr = arr.reshape(-1)  # flatten
    vec = [float(x) for x in arr]
    return vec



def main():
    # Resolve dtype
    dtype = _dtype_from_string(DTYPE)

    # Resolve images list
    if any(ch in IMAGE_INPUT for ch in ["*", "?", "["]):
        image_paths = sorted(glob.glob(IMAGE_INPUT))
    else:
        image_paths = [IMAGE_INPUT]

    if not image_paths:
        print(f"No images found for pattern/path: {IMAGE_INPUT}", file=sys.stderr)
        sys.exit(1)

    # Load model + processor
    print(f"Loading model '{MODEL_ID}' on {DEVICE} with dtype={DTYPE} ...")
    model, processor = load_model_and_processor(MODEL_ID, DEVICE, dtype)

    # Inference loop
    rows = []
    print(f"Running inference for {len(image_paths)} image(s) ...")
    for path in image_paths:
        try:
            vec = predict_action_for_image(
                model,
                processor,
                path,
                INSTRUCTION,
                DEVICE,
                dtype,
                unnorm_key=UNNORM_KEY,
                do_sample=DO_SAMPLE,
            )
            print(f"{os.path.basename(path)} => {', '.join(f'{x:.6f}' for x in vec)}")
            rows.append([path] + vec)
        except Exception as e:
            print(f"[ERROR] {path}: {e}", file=sys.stderr)

    # Save CSV
    out_path = os.path.join(os.path.dirname(__file__), OUTPUT_CSV)
    header = ["image_path", "dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} result(s) to {out_path}")


if __name__ == "__main__":
    main()
