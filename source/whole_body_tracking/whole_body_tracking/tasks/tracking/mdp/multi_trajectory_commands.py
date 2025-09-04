"""
Enhanced MotionCommand that supports multi-trajectory random sampling
"""

import math
import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

from whole_body_tracking.utils.multi_motion_loader import MultiMotionLoader

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MultiTrajectoryMotionCommand(CommandTerm):
    """支持多轨迹随机采样的 Motion Command"""
    
    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], 
            dtype=torch.long, device=self.device
        )
        
        # 加载多个 motion 文件
        self.motion_loader = MultiMotionLoader(self.cfg.motion_file, self.body_indexes.tolist(), device=self.device)
        
        # 为每个环境存储当前状态
        self.current_motion_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.max_timesteps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        # 缓存当前帧的数据
        self._joint_pos = torch.zeros(self.num_envs, self.motion_loader.motions[0]['joint_pos'].shape[1], device=self.device)
        self._joint_vel = torch.zeros(self.num_envs, self.motion_loader.motions[0]['joint_vel'].shape[1], device=self.device)
        self._body_pos_w = torch.zeros(self.num_envs, len(self.cfg.body_names), 3, device=self.device)
        self._body_quat_w = torch.zeros(self.num_envs, len(self.cfg.body_names), 4, device=self.device)
        self._body_lin_vel_w = torch.zeros(self.num_envs, len(self.cfg.body_names), 3, device=self.device)
        self._body_ang_vel_w = torch.zeros(self.num_envs, len(self.cfg.body_names), 3, device=self.device)
        
        # 相对位置和方向
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0
        
        # 初始化指标
        self._initialize_metrics()
        
        # 初始化采样
        self._sample_new_trajectories(torch.arange(self.num_envs, device=self.device))
        
    def _initialize_metrics(self):
        """初始化指标"""
        self.metrics = {}
        metric_names = [
            "error_anchor_pos", "error_anchor_rot", "error_anchor_lin_vel", "error_anchor_ang_vel",
            "error_body_pos", "error_body_rot", "error_body_lin_vel", "error_body_ang_vel",
            "error_joint_pos", "error_joint_vel"
        ]
        for name in metric_names:
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)
    
    def _sample_new_trajectories(self, env_ids: torch.Tensor):
        """为指定环境采样新的轨迹"""
        if len(env_ids) == 0:
            return
            
        # 随机采样 motion 和起始时间步
        motion_indices, timesteps = self.motion_loader.sample_motion_and_timestep(len(env_ids))
        
        self.current_motion_indices[env_ids] = motion_indices
        self.time_steps[env_ids] = timesteps
        
        # 设置每个环境的最大时间步
        for i, env_id in enumerate(env_ids):
            motion_idx = motion_indices[i].item()
            self.max_timesteps[env_id] = self.motion_loader.motions[motion_idx]['time_step_total'] - 1
        
        # 更新当前帧数据
        self._update_current_frame_data(env_ids)
    
    def _update_current_frame_data(self, env_ids: torch.Tensor):
        """更新指定环境的当前帧数据"""
        for env_id in env_ids:
            motion_idx = self.current_motion_indices[env_id].item()
            timestep = self.time_steps[env_id].item()
            
            motion_data = self.motion_loader.get_motion_data(motion_idx, timestep)
            
            self._joint_pos[env_id] = motion_data['joint_pos']
            self._joint_vel[env_id] = motion_data['joint_vel']
            self._body_pos_w[env_id] = motion_data['body_pos_w']
            self._body_quat_w[env_id] = motion_data['body_quat_w']
            self._body_lin_vel_w[env_id] = motion_data['body_lin_vel_w']
            self._body_ang_vel_w[env_id] = motion_data['body_ang_vel_w']
    
    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)
    
    @property
    def joint_pos(self) -> torch.Tensor:
        return self._joint_pos
    
    @property
    def joint_vel(self) -> torch.Tensor:
        return self._joint_vel
    
    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w + self._env.scene.env_origins[:, None, :]
    
    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w
    
    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w
    
    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w
    
    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self.motion_anchor_body_index] + self._env.scene.env_origins
    
    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self.motion_anchor_body_index]
    
    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self.motion_anchor_body_index]
    
    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self.motion_anchor_body_index]
    
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
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos
    
    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel
    
    def _update_command(self):
        """更新命令：前进时间步，检查是否需要重新采样轨迹"""
        # 前进时间步
        self.time_steps += 1
        
        # 找到需要重新采样的环境（到达轨迹末尾）
        env_ids_to_resample = torch.where(self.time_steps > self.max_timesteps)[0]
        
        if len(env_ids_to_resample) > 0:
            # 为这些环境采样新轨迹
            self._sample_new_trajectories(env_ids_to_resample)
        
        # 为所有环境更新当前帧数据
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        self._update_current_frame_data(all_env_ids)
        
        # 更新相对位置和方向
        self._update_relative_transforms()
        
        # 计算指标
        self._update_metrics()
    
    def _update_relative_transforms(self):
        """更新相对变换"""
        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        
        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))
        
        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)
    
    def _update_metrics(self):
        """更新指标"""
        # 计算各种误差
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
    
    def _resample_command(self, env_ids: Sequence[int]):
        """重新采样命令（用于与现有接口兼容）"""
        if len(env_ids) == 0:
            return
        
        env_ids_tensor = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        self._sample_new_trajectories(env_ids_tensor)
        
        # 应用姿态和速度扰动
        self._apply_pose_and_velocity_perturbations(env_ids_tensor)
    
    def _apply_pose_and_velocity_perturbations(self, env_ids: torch.Tensor):
        """应用姿态和速度扰动"""
        if len(env_ids) == 0:
            return
            
        # 获取当前状态
        root_pos = self._body_pos_w[env_ids, 0].clone()
        root_ori = self._body_quat_w[env_ids, 0].clone()
        root_lin_vel = self._body_lin_vel_w[env_ids, 0].clone()
        root_ang_vel = self._body_ang_vel_w[env_ids, 0].clone()
        
        # 姿态扰动
        if hasattr(self.cfg, 'pose_range') and self.cfg.pose_range:
            range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
            ranges = torch.tensor(range_list, device=self.device)
            rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
            root_pos += rand_samples[:, 0:3]
            orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
            root_ori = quat_mul(orientations_delta, root_ori)
        
        # 速度扰动
        if hasattr(self.cfg, 'velocity_range') and self.cfg.velocity_range:
            range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
            ranges = torch.tensor(range_list, device=self.device)
            rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
            root_lin_vel += rand_samples[:, :3]
            root_ang_vel += rand_samples[:, 3:]
        
        # 关节位置扰动
        joint_pos = self.joint_pos[env_ids].clone()
        joint_vel = self.joint_vel[env_ids].clone()
        
        if hasattr(self.cfg, 'joint_position_range'):
            joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
            soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
            joint_pos = torch.clip(joint_pos, soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1])
        
        # 写入机器人状态
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1), env_ids=env_ids
        )
