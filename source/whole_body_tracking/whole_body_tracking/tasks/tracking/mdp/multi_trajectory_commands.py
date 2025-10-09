from __future__ import annotations

import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

from whole_body_tracking.tasks.tracking.mdp.commands import MotionLoader
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MultiTrajectoryMotionCommand(CommandTerm):
    """Motion Command supporting multi-trajectory random sampling"""
    
    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], 
            dtype=torch.long, device=self.device
        )
        
    
        self.motions=[]
        for motions in self.cfg.motion_files:
            self.motions.append(MotionLoader(motions, self.body_indexes.tolist(), device=self.device))
    
        self.current_motion_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.max_timesteps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0
        # Problem 1: Concatenate all motion bins instead of using max
        motion_bin_counts = [
            int(motion.time_step_total // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1 
            for motion in self.motions
        ]
        self.bin_count = sum(motion_bin_counts)
        self.motion_bin_offsets = torch.cumsum(torch.tensor([0] + motion_bin_counts[:-1]), dim=0).to(self.device)
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], 
            device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()
        self.motion_joint_pos = torch.cat([m.joint_pos for m in self.motions], dim=0)
        self.motion_joint_vel = torch.cat([m.joint_vel for m in self.motions], dim=0)
        self.motion_body_pos_w = torch.cat([m.body_pos_w for m in self.motions], dim=0)
        self.motion_body_quat_w = torch.cat([m.body_quat_w for m in self.motions], dim=0)
        self.motion_body_lin_vel_w = torch.cat([m.body_lin_vel_w for m in self.motions], dim=0)
        self.motion_body_ang_vel_w = torch.cat([m.body_ang_vel_w for m in self.motions], dim=0)
        self.motion_anchor_body_pos_w = torch.cat([m.body_pos_w[:, self.motion_anchor_body_index] for m in self.motions], dim=0)
        self.motion_anchor_body_quat_w = torch.cat([m.body_quat_w[:, self.motion_anchor_body_index] for m in self.motions], dim=0)
        self.motion_anchor_body_lin_vel_w = torch.cat([m.body_lin_vel_w[:, self.motion_anchor_body_index] for m in self.motions], dim=0)
        self.motion_anchor_body_ang_vel_w = torch.cat([m.body_ang_vel_w[:, self.motion_anchor_body_index] for m in self.motions], dim=0)
        motion_lengths = [m.joint_pos.shape[0] for m in self.motions]
        self.motion_offsets = torch.cumsum(torch.tensor([0] + motion_lengths[:-1]), dim=0).to(self.device)


        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_motion"] = torch.zeros(self.num_envs, device=self.device)
        print("[MultiTrajectoryMotionCommand] Performing initial sampling...")
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        self._resample_command(all_env_ids)
        print("[MultiTrajectoryMotionCommand] Initialization complete")
    def _sample_motion_indices(self, env_ids: torch.Tensor):
        """Sample new motion indices for specified environments"""

        # Randomly sample motion and starting time steps
        sampled_indices = torch.randint(
                0, len(self.motions), 
                (len(env_ids),), 
                device=self.device
            )
        return sampled_indices


    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)
    
    @property
    def joint_pos(self) -> torch.Tensor:
        global_indices = self.motion_offsets[self.current_motion_indices] + self.time_steps
        return self.motion_joint_pos[global_indices]
    
    @property
    def joint_vel(self) -> torch.Tensor:
        global_indices = self.motion_offsets[self.current_motion_indices] + self.time_steps
        return self.motion_joint_vel[global_indices]
    
    @property
    def body_pos_w(self) -> torch.Tensor:
        global_indices = self.motion_offsets[self.current_motion_indices] + self.time_steps
        return self.motion_body_pos_w[global_indices] + self._env.scene.env_origins[:, None, :]
    
    @property
    def body_quat_w(self) -> torch.Tensor:
        global_indices = self.motion_offsets[self.current_motion_indices] + self.time_steps
        return self.motion_body_quat_w[global_indices]
    
    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        global_indices = self.motion_offsets[self.current_motion_indices] + self.time_steps
        return self.motion_body_lin_vel_w[global_indices]
    
    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        global_indices = self.motion_offsets[self.current_motion_indices] + self.time_steps
        return self.motion_body_ang_vel_w[global_indices]
    
    @property
    def anchor_pos_w(self) -> torch.Tensor:
        global_indices = self.motion_offsets[self.current_motion_indices] + self.time_steps
        return self.motion_anchor_body_pos_w[global_indices] + self._env.scene.env_origins
    
    @property
    def anchor_quat_w(self) -> torch.Tensor:
        global_indices = self.motion_offsets[self.current_motion_indices] + self.time_steps
        return self.motion_anchor_body_quat_w[global_indices]
    
    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        global_indices = self.motion_offsets[self.current_motion_indices] + self.time_steps
        return self.motion_anchor_body_lin_vel_w[global_indices]
    
    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        global_indices = self.motion_offsets[self.current_motion_indices] + self.time_steps
        return self.motion_anchor_body_ang_vel_w[global_indices]
    
    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]
    
    def _update_metrics(self):
        """Update metrics"""
        # Compute various errors
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)
        
        self.metrics["error_body_pos"] = torch.norm(self.body_pos_w - self.robot_body_pos_w, dim=-1).mean(dim=-1)
        self.metrics["error_body_rot"] = quat_error_magnitude(
            self.body_quat_w.view(-1, 4), self.robot_body_quat_w.view(-1, 4)
        ).view(self.num_envs, -1).mean(dim=-1)
        
        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(dim=-1)
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(dim=-1)
        
        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)
    def _adaptive_sampling(self, env_ids: Sequence[int]):
        """Adaptive sampling with per-environment handling and timeout awareness"""
        if hasattr(self._env, 'termination_manager') and self._env.termination_manager is not None:
            episode_failed = self._env.termination_manager.terminated[env_ids]
            
            if torch.any(episode_failed):
                # Problem 2: Use motion-specific bin offsets for failure recording
                current_bin_indices = []
                for i, env_id in enumerate(env_ids):
                    if episode_failed[i]:
                        motion_idx = self.current_motion_indices[env_id].item()
                        motion = self.motions[motion_idx]
                        # Calculate motion-specific bin count for this motion
                        motion_bin_count = int(motion.time_step_total // (1 / (self._env.cfg.decimation * self._env.cfg.sim.dt))) + 1
                        local_bin_idx = torch.clamp(
                            (self.time_steps[env_id] * motion_bin_count) // max(motion.time_step_total, 1),
                            0, motion_bin_count - 1
                        )
                        # Map to global bin index using motion bin offsets
                        global_bin_idx = self.motion_bin_offsets[motion_idx] + local_bin_idx
                        current_bin_indices.append(global_bin_idx)
                
                if len(current_bin_indices) > 0:
                    fail_bins = torch.stack(current_bin_indices)
                    self._current_bin_failed[:] = torch.bincount(
                        fail_bins, minlength=self.bin_count
                    )

        # 计算采样概率
        sampling_probabilities = self.bin_failed_count + \
                                self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(
            sampling_probabilities, self.kernel.view(1, 1, -1)
        ).view(-1)
        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        sampled_bins = torch.multinomial(
            sampling_probabilities, len(env_ids), replacement=True
        )
        
        # Problem 3: Apply timeout-aware sampling for each environment
        episode_timeout_steps = int(self._env.cfg.episode_length_s / (self._env.cfg.decimation * self._env.cfg.sim.dt))
        
        for i, env_id in enumerate(env_ids):
            global_bin = sampled_bins[i].item()
            
            # Find which motion this bin belongs to
            motion_idx = 0
            for j in range(len(self.motion_bin_offsets)):
                if j == len(self.motion_bin_offsets) - 1 or global_bin < self.motion_bin_offsets[j + 1]:
                    motion_idx = j
                    break
            
            # Calculate local bin within the motion
            local_bin = global_bin - self.motion_bin_offsets[motion_idx]
            motion = self.motions[motion_idx]
            motion_bin_count = int(motion.time_step_total // (1 / (self._env.cfg.decimation * self._env.cfg.sim.dt))) + 1
            
            # Check timeout constraint
            max_safe_time_step = max(0, motion.time_step_total - episode_timeout_steps)
            max_safe_local_bin = int((max_safe_time_step * motion_bin_count) // motion.time_step_total)
            
            # Clamp to safe range
            safe_local_bin = min(local_bin, max_safe_local_bin)
            
            # Set motion and time step
            self.current_motion_indices[env_id] = motion_idx
            self.time_steps[env_id] = int(
                (safe_local_bin / motion_bin_count * (motion.time_step_total - 1))
            )

        # 更新metrics
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count
        
        # Calculate which motion the top bin belongs to
        top_bin = imax.item()
        top_motion_idx = 0
        for j in range(len(self.motion_bin_offsets)):
            if j == len(self.motion_bin_offsets) - 1 or top_bin < self.motion_bin_offsets[j + 1]:
                top_motion_idx = j
                break
        self.metrics["sampling_top1_motion"][:] = top_motion_idx
    
    def _resample_command(self, env_ids: Sequence[int]):
        """Resample command: select motion + select starting point"""
        if len(env_ids) == 0:
            return
        
        # 1. Sample motion indices
        sampled_motion_indices = self._sample_motion_indices(env_ids)
        self.current_motion_indices[env_ids] = sampled_motion_indices
        
        # 2. Set max_timesteps for each environment
        for env_id in env_ids:
            motion_idx = self.current_motion_indices[env_id].item()
            self.max_timesteps[env_id] = self.motions[motion_idx].time_step_total - 1
        
        # 3. Adaptive sampling starting point
        self._adaptive_sampling(env_ids)

        # 4. Get initial state and add randomization
        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        # Pose randomization
        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) 
                     for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
        )
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(
            rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
        )
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        
        # Velocity randomization
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) 
                     for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
        )
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        # Joint position randomization
        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()
        joint_pos += sample_uniform(
            *self.cfg.joint_position_range, joint_pos.shape, joint_pos.device
        )
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], 
            soft_joint_pos_limits[:, :, 0], 
            soft_joint_pos_limits[:, :, 1]
        )
        
        # 4. Set robot state
        self.robot.write_joint_state_to_sim(
            joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids
        )
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], 
                      root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

    def _update_command(self):
        """Update command for each time step"""
        # 1. Advance time step
        self.time_steps += 1
        
        # 2. Check if resampling is needed (motion end)
        needs_resample = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for i, motion in enumerate(self.motions):
            mask = self.current_motion_indices == i
            needs_resample[mask] = self.time_steps[mask] >= motion.time_step_total
        
        env_ids = torch.where(needs_resample)[0]
        self._resample_command(env_ids)

        # 3. Update relative target (same logic as MotionCommand)
        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(
            delta_ori_w, self.body_pos_w - anchor_pos_w_repeat
        )

        # 4. Update adaptive sampling statistics

        self.bin_failed_count = (
                self.cfg.adaptive_alpha * self._current_bin_failed + 
                (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
            )
        self._current_bin_failed.zero_()
        
        # Update metrics for logging
        self._update_metrics()


@configclass
class MultiTrajectoryMotionCommandCfg(CommandTermCfg):
    """Configuration for multi-trajectory motion command."""
    
    class_type: type = MultiTrajectoryMotionCommand
    
    asset_name: str = MISSING
    """Name of the robot asset in the scene."""
    
    motion_files: list[str] = MISSING
    """Path to motion file, directory containing multiple motion files, or wandb registry pattern."""
    
    anchor_body_name: str = MISSING
    """Name of the anchor body."""
    
    body_names: list[str] = MISSING
    """Names of bodies to track."""
    
    # Command resampling (MotionCommandCfg)
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    """Range for resampling time (not used in multi-trajectory mode but required for compatibility)."""
    
    debug_vis: bool = False
    """Enable debug visualization."""
    
    pose_range: dict[str, tuple[float, float]] = {}
    """Range for pose perturbations. Keys: x, y, z, roll, pitch, yaw."""
    
    velocity_range: dict[str, tuple[float, float]] = {}
    """Range for velocity perturbations. Keys: x, y, z, roll, pitch, yaw."""
    
    joint_position_range: tuple[float, float] = (-0.52, 0.52)
    """Range for joint position perturbations."""
    
    # Visualization
    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
    
    adaptive_kernel_size: int = 3
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

