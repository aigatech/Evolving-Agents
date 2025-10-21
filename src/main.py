from dotenv import load_dotenv
load_dotenv()

import os
from pathlib import Path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PIL import Image
from src.utils.robot_utils import get_model
from config.vla_config import GenerateConfig
import torch

cfg = GenerateConfig(
    model_family="prismatic",
    pretrained_checkpoint="/home/shivang/Desktop/AIGT/pretrained/minivla-libero90-prismatic/checkpoints/step-122500-epoch-55-loss=0.0743.pt",
    hf_token="HF_TOKEN",
    trt_vision_engine=None,
    require_trt_vision=False,
    trtllm_engine_dir=None,
    trtllm_max_new_tokens=64,
    load_in_8bit=False,
    load_in_4bit=False,
    center_crop=True,
    obs_history=1,
    use_wrist_image=False,
    task_suite_name="libero_spatial",
    num_steps_wait=10,
    num_trials_per_task=50,
    run_id_note=None,
    local_log_dir="./logs",
    prefix='',
    use_wandb=False,
    wandb_project="prismatic",
    wandb_entity=None,
    seed=7,
)
model = get_model(cfg)
print(f"Model loaded: {type(model)}")

image = Image.open("test.jpg")
instruction = "pick up a fruit and place it in a bowl"
# The model contains normalization statistics under `model.norm_stats`.
# `predict_action(..., unnorm_key=...)` expects one of those keys. Use a valid
# key if available, otherwise call without `unnorm_key` so the model will pick
# the only available stats or operate without unnormalization.
unnorm_key = None
try:
    norm_stats = getattr(model, "norm_stats", None)
    if isinstance(norm_stats, dict) and len(norm_stats) > 0:
        available = list(norm_stats.keys())
        # Prefer the first available key (models with a single dataset will
        # have exactly one key). Print available options for debugging.
        print(f"Available norm_stats keys: {available}")
        unnorm_key = available[0]
        print(f"Using unnorm_key='{unnorm_key}' for predict_action")
    else:
        print("No norm_stats present on model; calling predict_action without unnorm_key")

    if unnorm_key is not None:
        action = model.predict_action(image, instruction, unnorm_key=unnorm_key)
    else:
        action = model.predict_action(image, instruction)
    print(action)
except AssertionError as e:
    # Catch the specific assertion from _check_unnorm_key and retry without
    # providing the key (useful if stats changed unexpectedly).
    print(f"AssertionError while selecting unnorm_key: {e}")
    print("Retrying predict_action without unnorm_key...")
    action = model.predict_action(image, instruction)
    print(action)
