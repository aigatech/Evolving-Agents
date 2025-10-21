import sys
sys.path.append("/home/shivang/Desktop/AIGT/BEHAVIOR-1K/OmniGibson")


import yaml
from PIL import Image

import omnigibson as og
from omnigibson.macros import gm, macros

gm.HEADLESS = True

# cfg = dict()

# # Define scene
# cfg["scene"] = {
#     "type": "Scene",
#     "floor_plane_visible": True,
# }

# # Define objects
# cfg["objects"] = [
#     {
#         "type": "DatasetObject",
#         "name": "delicious_apple",
#         "category": "apple",
#         "model": "agveuv",
#         "position": [0, 0, 1.0],
#     },
#     {
#         "type": "PrimitiveObject",
#         "name": "incredible_box",
#         "primitive_type": "Cube",
#         "rgba": [0, 1.0, 1.0, 1.0],
#         "scale": [0.5, 0.5, 0.1],
#         "fixed_base": True,
#         "position": [-1.0, 0, 1.0],
#         "orientation": [0, 0, 0.707, 0.707],
#     },
#     {
#         "type": "LightObject",
#         "name": "brilliant_light",
#         "light_type": "Sphere",
#         "intensity": 50000,
#         "radius": 0.1,
#         "position": [3.0, 3.0, 4.0],
#     },
# ]

# # Define robots
# cfg["robots"] = [
#     {
#         "type": "Fetch",
#         "name": "baby_robot",
#         "obs_modalities": ["rgb", "depth"],
#     },
# ]

# # Define task
# cfg["task"] = {
#     "type": "DummyTask",
#     "termination_config": dict(),
#     "reward_config": dict(),
# }

cfg = yaml.safe_load(open("config/simulator.yaml", "r"))

# Create the environment
env = og.Environment(cfg)

og.sim.viewer_camera.set_position_orientation(
    position=[1.6, 6.15, 1.5], orientation=[-0.2322, 0.5895, 0.7199, -0.2835]
)

# Allow camera teleoperation
og.sim.enable_viewer_camera_teleoperation()

# Step!
for i in range(100):
    action = env.robots[0].action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    robot_name = list(state.keys())[0]
    # img = Image.fromarray(state['external']['external_sensor0']['rgb'].detach().cpu().numpy())
    img = Image.fromarray(state[robot_name][f'{robot_name}:zed_link:Camera:0']['rgb'].detach().cpu().numpy())
    print(state)
    img.save(f"images/fetch_rgb_{i}.png")
    if terminated or truncated:
        og.log.info("Episode finished after {} timesteps".format(i + 1))
        break
# for idx in range(10000):
#     obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
#     print(obs)
#     robot_name = list(obs.keys())[0]
#     # img = Image.fromarray(obs["baby_robot"]["baby_robot:eyes:Camera:0"]['rgb'].detach().cpu().numpy())
#     # img = Image.fromarray(obs[robot_name][f'{robot_name}:eef_link:Camera:0']['rgb'].detach().cpu().numpy())
    # img = Image.fromarray(obs['external']['external_sensor0']['rgb'].detach().cpu().numpy())
    # img.save(f"images/fetch_rgb_{idx}.png")


og.shutdown()