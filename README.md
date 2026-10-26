# WiscoHumanoids — G1 Locomotion

A reinforcement learning project using MuJoCo, Gymnasium, and Stable-Baselines3 PPO to train a Unitree G1 robot. The current focus is standing balance before progressing to walking.

## Files

- `g1_env.py` — Robot environment, PD controller, observations, and rewards.
- `train_ppo.py` — Continues training from a checkpoint and saves evaluations and models.
- `play_ppo.py` — Displays a trained policy and logs foot separation and height.

## Setup

Install dependencies:

```bash
pip install numpy gymnasium mujoco "stable-baselines3[extra]"
```

Obtain the robot assets from [Unitree RL Gym](https://github.com/unitreerobotics/unitree_rl_gym). The environment expects:

```text
~/unitree_rl_gym/resources/robots/g1_description/scene.xml
```

A compatible trained checkpoint is required for both commands below. Model assets and checkpoints are not included in this repository.

## Usage

View a policy on macOS:

```bash
mjpython play_ppo.py --checkpoint /path/to/model.zip --duration 60
```

Continue training:

```bash
python train_ppo.py --checkpoint /path/to/model.zip --timesteps 300000
```

## Progress and Next Steps

The policy has learned to stand, but gradual sideways leg spreading remains an issue. Hip-posture and foot-width penalties have been implemented; the width penalty still needs evaluation.

Next steps are to quantify stance drift, improve sustained balance, test robustness, and progress to slow walking. Stable walking has not yet been achieved.
