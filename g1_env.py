import os

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


class G1Env(gym.Env):

    def __init__(self, width_penalty_weight=0.0, width_tolerance=0.1,
                 episode_seconds=10.0):

        if width_penalty_weight < 0 or width_tolerance < 0 or episode_seconds <= 0:
            raise ValueError("Invalid width penalty or episode duration")

        model_path = os.path.expanduser(
            "~/unitree_rl_gym/resources/robots/g1_description/scene.xml"
        )

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.foot_ids = [self.model.body(f"{side}_ankle_roll_link").id
                         for side in ("left", "right")]
        reference = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, reference)
        self.reference_width = float(np.linalg.norm(
            reference.xpos[self.foot_ids[0], :2] - reference.xpos[self.foot_ids[1], :2]))
        self.width_limit = self.reference_width * (1.0 + width_tolerance)
        self.width_penalty_weight = width_penalty_weight
        self.width_data = mujoco.MjData(self.model)

        # ----- low-level controller -----

        self.kp = np.array([
            100, 100, 100, 150, 40, 40,
            100, 100, 100, 150, 40, 40
        ], dtype=np.float64)

        self.kd = np.array([
            2, 2, 2, 4, 2, 2,
            2, 2, 2, 4, 2, 2
        ], dtype=np.float64)
        # Temporary diagnostic: change only the ankle-pitch gains.
        self.kp[[4, 10]] = 80.0   # previously 40
        self.kd[[4, 10]] = 4.0    # previously 2

        self.default_q = self.model.qpos0[7:19].copy()

        self.action_scale = 0.25

        # 500 Hz physics → 50 Hz policy
        self.decimation = 10

        # ----- RL spaces -----

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(12,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(48,),
            dtype=np.float32
        )

        self.previous_action = np.zeros(12)

        # Desired [vx, vy, yaw rate]
        self.command = np.array([0.0, 0.0, 0.0])

        self.max_episode_steps = int(np.ceil(episode_seconds / (self.model.opt.timestep * self.decimation)))
        self.step_count = 0
    
    
    def _get_observation(self):

        base_ang_vel = self.data.qvel[3:6].copy()

        q = self.data.qpos[7:19].copy()
        joint_pos_error = q - self.default_q

        joint_vel = self.data.qvel[6:18].copy()

        quat = self.data.qpos[3:7].copy()

        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, quat)
        rot = rot.reshape(3, 3)

        world_linear_velocity = self.data.qvel[0:3].copy()
        body_linear_velocity = rot.T @ world_linear_velocity

        projected_gravity = (
            rot.T @ np.array([0.0, 0.0, -1.0])
        )

        obs = np.concatenate([
            base_ang_vel,
            body_linear_velocity,
            projected_gravity,
            self.command,
            joint_pos_error,
            joint_vel,
            self.previous_action
        ])

        return obs.astype(np.float32)
    
    def step(self, action):
        

        action = np.clip(action, -1.0, 1.0)

        target_q = (
            self.default_q
            + self.action_scale * action
        )

        torque_squared_sum = 0.0

        for _ in range(self.decimation):

            q = self.data.qpos[7:19]
            dq = self.data.qvel[6:18]

            tau = (
                self.kp * (target_q - q)
                - self.kd * dq
            )

            self.data.ctrl[:] = tau

            torque_squared_sum += np.sum(tau ** 2)

            mujoco.mj_step(
                self.model,
                self.data
            )

        action_rate_penalty = np.sum((action - self.previous_action) ** 2)
        self.previous_action = action.copy()

        obs = self._get_observation()

        projected_gravity = obs[6:9]
        upright_reward = max(0.0, -projected_gravity[2])

        self.step_count += 1
        
        # Track commanded velocity in the body frame
        quat = self.data.qpos[3:7].copy()
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, quat)
        rot = rot.reshape(3, 3)

        world_linear_velocity = self.data.qvel[0:3].copy()
        body_linear_velocity = rot.T @ world_linear_velocity
        vx_body = body_linear_velocity[0]
        vy_body = body_linear_velocity[1]
        yaw_rate = self.data.qvel[5]

        target_vx = self.command[0]
        target_vy = self.command[1]
        target_yaw_rate = self.command[2]

        tracking_error = (
            (vx_body - target_vx) ** 2
            + (vy_body - target_vy) ** 2
            + (yaw_rate - target_yaw_rate) ** 2
        )

        velocity_reward = np.exp(
            -tracking_error / 0.25
        )

        # Small reward for remaining alive
        survival_reward = 0.2

        # Penalize excessive torque
        mean_tau_sq = (
            torque_squared_sum
            / self.decimation
        )

        torque_penalty = (
            0.00001 * mean_tau_sq
        )

        # Penalize hip roll/yaw deviations that can accompany leg spreading.
        joint_error = self.data.qpos[7:19] - self.default_q
        stance_indices = [1, 2, 7, 8]

        stance_penalty = 2.0 * np.sum(
            joint_error[stance_indices] ** 2
        )

        width_penalty = 0.0
        if self.width_penalty_weight:
            # A separate data object avoids changing the simulation's solver state.
            mujoco.mj_copyData(self.width_data, self.model, self.data)
            mujoco.mj_kinematics(self.model, self.width_data)
            width = np.linalg.norm(self.width_data.xpos[self.foot_ids[0], :2]
                                   - self.width_data.xpos[self.foot_ids[1], :2])
            excess = max(0.0, float(width) - self.width_limit) / self.reference_width
            width_penalty = self.width_penalty_weight * excess ** 2

        reward = (
            1.0 * velocity_reward
            + 0.5 * upright_reward
            + 0.2 * survival_reward
            - torque_penalty
            - 0.03 * action_rate_penalty
            - stance_penalty
            - width_penalty
        )
        base_height = self.data.qpos[2]

        terminated = bool(
            base_height < 0.65
            or upright_reward < np.cos(np.deg2rad(30))
        )

        truncated = bool(
            self.step_count >= self.max_episode_steps
        )

        info = {
            "velocity_reward": velocity_reward,
            "upright_reward": upright_reward,
            "survival_reward": survival_reward,
            "torque_penalty": torque_penalty,
            "action_rate_penalty": action_rate_penalty,
            "base_height": base_height,
            "vx_body": vx_body,
            "vy_body": vy_body,
            "yaw_rate": yaw_rate,
            "forward_velocity": world_linear_velocity[0],
            "stance_penalty": float(stance_penalty),
            "width_penalty": float(width_penalty)
        }

        return (
            obs,
            float(reward),
            terminated,
            truncated,
            info
        )
        
    # def reset(self, seed=None, options=None):

    #     super().reset(seed=seed)

    #     mujoco.mj_resetData(
    #         self.model,
    #         self.data
    #     )

    #     # Start approximately in nominal standing posture
    #     self.data.qpos[7:19] = self.default_q

    #     self.data.qvel[:] = 0.0

    #     mujoco.mj_forward(
    #         self.model,
    #         self.data
    #     )

    #     self.previous_action[:] = 0.0
    #     self.step_count = 0

    #     obs = self._get_observation()

    #     return obs, {}
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Restore the model's original positions and velocities.
        mujoco.mj_resetData(self.model, self.data)

        # Make the PD targets match the model's original leg pose.
        self.default_q = self.model.qpos0[7:19].copy()
        self.data.qpos[7:19] = self.default_q

        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.previous_action[:] = 0.0
        self.step_count = 0

        return self._get_observation(), {}
