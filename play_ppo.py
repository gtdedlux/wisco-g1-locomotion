import time
import argparse

import mujoco.viewer
from stable_baselines3 import PPO

from g1_env import G1Env

import numpy as np

parser = argparse.ArgumentParser(description="View a standing policy checkpoint.")
parser.add_argument("--checkpoint", default="runs/g1_stand_v2_20261025_205207/g1_ppo_stand_final.zip")
parser.add_argument("--duration", type=float, default=60.0)
args = parser.parse_args()
env = G1Env(episode_seconds=args.duration)

model = PPO.load(
    args.checkpoint,
    device="cpu",
)
obs, info = env.reset()

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    left_foot = env.unwrapped.model.body("left_ankle_roll_link").id
    right_foot = env.unwrapped.model.body("right_ankle_roll_link").id
    playback_steps = 0
    while viewer.is_running():

        step_start = time.time()

        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)
        playback_steps += 1

        if playback_steps % 100 == 0:  # Every 2 simulated seconds
            data = env.unwrapped.data
            # Update positions for the integrated qpos without recomputing forces.
            mujoco.mj_kinematics(env.model, data)

            separation = np.linalg.norm(
                data.xpos[left_foot, :2] - data.xpos[right_foot, :2]
            )

            print(
                f"time={data.time:.1f}s "
                f"foot_separation={separation:.3f}m "
                f"height={data.qpos[2]:.3f}m"
            )

        viewer.sync()
        if terminated or truncated:
            print(
                f"Episode ended at {env.data.time:.1f}s: "
                f"terminated={terminated}, truncated={truncated}"
            )
            break

        control_dt = env.model.opt.timestep * env.decimation

        remaining = control_dt - (time.time() - step_start)

        if remaining > 0:
            time.sleep(remaining)
