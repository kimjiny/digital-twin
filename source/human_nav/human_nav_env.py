# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform


@configclass
class HumanNavEnvCfg(DirectRLEnvCfg):
    # --- env ---
    decimation = 4
    episode_length_s = 12.0
    action_space = 2          # planar velocity command (vx, vy)
    # obs: to_goal(2) + vel(2) + rel_static_obs(2*3) + rel_robot(2) + robot_vel(2) = 14
    observation_space = 14
    state_space = 0

    # --- simulation ---
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 120.0, render_interval=decimation)

    # --- scene ---
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024, env_spacing=12.0, replicate_physics=True, clone_in_fabric=True
    )

    # --- agent (capsule "human") ---
    agent_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Agent",
        spawn=sim_utils.CapsuleCfg(
            radius=0.3,
            height=1.0,
            axis="Z",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=70.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.4, 0.85)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.9)),
    )

    # --- moving robot obstacle ---
    robot_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.CylinderCfg(
            radius=0.35,
            height=1.2,
            axis="Z",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.25, 0.15)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, 0.0, 0.9)),
    )

    # --- optional room backdrop ---
    use_room: bool = False
    room_collision: bool = False
    room_usd_path: str = "/home/dclcom57/jy/isaac_project/environments/simple_room.usd"

    # --- task params ---
    max_speed = 1.5            # m/s
    spawn_height = 0.9         # m
    goal_radius = 0.5          # m  -> success
    collision_radius = 0.5     # m  -> failure vs static obstacle (center-to-center)
    robot_collision_radius = 0.7  # m  -> failure vs moving robot

    # moving robot motion (patrols along y at x = robot_x)
    robot_x = 1.0
    robot_amp = 3.0            # m  amplitude in y
    robot_freq = 0.2           # Hz

    # --- reward scales ---
    rew_progress = 1.0
    rew_goal = 50.0
    rew_collision = -25.0
    rew_robot_collision = -30.0
    rew_step = -0.01


@configclass
class HumanNavRoomEnvCfg(HumanNavEnvCfg):
    use_room = True
    room_collision = False
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=256, env_spacing=12.0, replicate_physics=True, clone_in_fabric=True
    )


class HumanNavEnv(DirectRLEnv):
    cfg: HumanNavEnvCfg

    def __init__(self, cfg: HumanNavEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # static obstacle layout (local offsets): a vertical wall of 3 obstacles
        self.obstacle_offsets = torch.tensor(
            [[0.0, -2.0], [0.0, 0.0], [0.0, 2.0]], device=self.device
        )
        self.num_obstacles = self.obstacle_offsets.shape[0]

        origins_xy = self.scene.env_origins[:, :2]  # (N, 2)
        self.obstacles_w = origins_xy.unsqueeze(1) + self.obstacle_offsets.unsqueeze(0)  # (N, K, 2)

        # buffers
        self.goal_w = torch.zeros(self.num_envs, 2, device=self.device)
        self.vel_command = torch.zeros(self.num_envs, 2, device=self.device)
        self.prev_dist = torch.zeros(self.num_envs, device=self.device)
        self._cur_dist = torch.zeros(self.num_envs, device=self.device)
        self._reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._collided = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # moving robot state
        self.robot_phase = torch.zeros(self.num_envs, device=self.device)
        self.robot_w = torch.zeros(self.num_envs, 2, device=self.device)
        self.robot_vel_w = torch.zeros(self.num_envs, 2, device=self.device)
        self.sim_time = torch.zeros(self.num_envs, device=self.device)
        self.control_dt = self.cfg.sim.dt * self.cfg.decimation

    # ------------------------------------------------------------------ #
    def _setup_scene(self):
        self.agent = RigidObject(self.cfg.agent_cfg)
        self.robot = RigidObject(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.rigid_objects["agent"] = self.agent
        self.scene.rigid_objects["robot"] = self.robot

        # optional room backdrop (spawned to all envs after cloning)
        if self.cfg.use_room:
            room_cfg = sim_utils.UsdFileCfg(usd_path=self.cfg.room_usd_path)
            if self.cfg.room_collision:
                room_cfg.collision_props = sim_utils.CollisionPropertiesCfg()
            room_cfg.func("/World/envs/env_.*/Room", room_cfg, translation=(0.0, 0.0, 0.0))

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------ #
    def _update_robot(self):
        # patrol along y at x = robot_x (env-local), sinusoidal
        phase = 2.0 * math.pi * self.cfg.robot_freq * self.sim_time + self.robot_phase
        y = self.cfg.robot_amp * torch.sin(phase)
        vy = self.cfg.robot_amp * 2.0 * math.pi * self.cfg.robot_freq * torch.cos(phase)
        origins_xy = self.scene.env_origins[:, :2]
        self.robot_w[:, 0] = origins_xy[:, 0] + self.cfg.robot_x
        self.robot_w[:, 1] = origins_xy[:, 1] + y
        self.robot_vel_w[:, 0] = 0.0
        self.robot_vel_w[:, 1] = vy
        # write kinematic pose to sim
        pose = torch.zeros(self.num_envs, 7, device=self.device)
        pose[:, 0:2] = self.robot_w
        pose[:, 2] = self.cfg.spawn_height
        pose[:, 3] = 1.0  # quat w = 1 (identity)
        self.robot.write_root_pose_to_sim(pose)

    # ------------------------------------------------------------------ #
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.vel_command = torch.clamp(actions, -1.0, 1.0) * self.cfg.max_speed
        self.sim_time += self.control_dt
        self._update_robot()

    def _apply_action(self) -> None:
        vel = torch.zeros(self.num_envs, 6, device=self.device)
        vel[:, 0:2] = self.vel_command
        self.agent.write_root_velocity_to_sim(vel)

    # ------------------------------------------------------------------ #
    def _get_observations(self) -> dict:
        agent_xy = self.agent.data.root_pos_w[:, :2]
        agent_vel = self.agent.data.root_lin_vel_w[:, :2]
        to_goal = self.goal_w - agent_xy
        rel_obs = (self.obstacles_w - agent_xy.unsqueeze(1)).reshape(self.num_envs, -1)
        rel_robot = self.robot_w - agent_xy
        obs = torch.cat([to_goal, agent_vel, rel_obs, rel_robot, self.robot_vel_w], dim=-1)
        return {"policy": obs}

    # ------------------------------------------------------------------ #
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        agent_xy = self.agent.data.root_pos_w[:, :2]
        self._cur_dist = torch.norm(self.goal_w - agent_xy, dim=-1)
        d_obs = torch.norm(self.obstacles_w - agent_xy.unsqueeze(1), dim=-1)
        min_obs = torch.min(d_obs, dim=-1).values
        d_robot = torch.norm(self.robot_w - agent_xy, dim=-1)

        self._reached = self._cur_dist < self.cfg.goal_radius
        self._hit_obs = min_obs < self.cfg.collision_radius
        self._hit_robot = d_robot < self.cfg.robot_collision_radius
        self._collided = self._hit_obs | self._hit_robot

        terminated = self._reached | self._collided
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _get_rewards(self) -> torch.Tensor:
        progress = self.prev_dist - self._cur_dist
        self.prev_dist = self._cur_dist.clone()
        reward = (
            self.cfg.rew_progress * progress
            + self.cfg.rew_goal * self._reached.float()
            + self.cfg.rew_collision * self._hit_obs.float()
            + self.cfg.rew_robot_collision * self._hit_robot.float()
            + self.cfg.rew_step
        )
        return reward

    # ------------------------------------------------------------------ #
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.agent._ALL_INDICES
        super()._reset_idx(env_ids)

        n = len(env_ids)
        origins_xy = self.scene.env_origins[env_ids, :2]

        start_local = torch.zeros(n, 2, device=self.device)
        start_local[:, 0] = sample_uniform(-4.0, -3.0, (n,), self.device)
        start_local[:, 1] = sample_uniform(-2.5, 2.5, (n,), self.device)

        goal_local = torch.zeros(n, 2, device=self.device)
        goal_local[:, 0] = sample_uniform(3.0, 4.0, (n,), self.device)
        goal_local[:, 1] = sample_uniform(-2.5, 2.5, (n,), self.device)

        self.goal_w[env_ids] = origins_xy + goal_local

        root_state = self.agent.data.default_root_state[env_ids].clone()
        root_state[:, 0:2] = origins_xy + start_local
        root_state[:, 2] = self.cfg.spawn_height
        root_state[:, 7:] = 0.0
        self.agent.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.agent.write_root_velocity_to_sim(root_state[:, 7:], env_ids)

        # reset robot motion (random phase, reset time)
        self.robot_phase[env_ids] = sample_uniform(0.0, 2.0 * math.pi, (n,), self.device)
        self.sim_time[env_ids] = 0.0

        self.prev_dist[env_ids] = torch.norm(goal_local - start_local, dim=-1)
        self._cur_dist[env_ids] = self.prev_dist[env_ids]
