import os
from pathlib import Path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from src.utils.robot_utils import get_model
from config.vla_config import GenerateConfig

if __name__ == "__main__":
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
