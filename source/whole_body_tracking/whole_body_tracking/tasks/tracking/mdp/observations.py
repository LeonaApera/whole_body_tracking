from __future__ import annotations

import torch
from typing import TYPE_CHECKING
import pdb
from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.robot_anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, :3].view(env.num_envs, -1)


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, 3:6].view(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)

def latent_space(env: ManagerBasedEnv, 
                 command_name: str,
                 vqvae_model_path: str = "/home/thomas/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/final_model.pt",
                 window_size: int = 60) -> torch.Tensor:
    """
    Generate latent space representation using pre-trained VQ-VAE model.
    
    Args:
        env: The environment instance
        command_name: Name of the motion command to use
        vqvae_model_path: Path to the pre-trained VQ-VAE model
        window_size: Size of the sliding window (default: 60 frames for 2 seconds at 30fps)
    
    Returns:
        Concatenated latent features from lower and upper body encoders
        Shape: (num_envs, latent_dim)
    """
    import os
    from whole_body_tracking.tasks.tracking.mdp.vqvae_sliding_window_conv import SlidingWindowVQVAEConv
    
    # Get the motion command
    command: MotionCommand = env.command_manager.get_term(command_name)
    
    # Initialize VQ-VAE model if not already loaded
    if not hasattr(env, '_vqvae_model'):
        # Load the pre-trained model
        model_path = vqvae_model_path
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"VQ-VAE model not found at {model_path}")
        
        # Create model instance with correct parameters matching the pre-trained model
        env._vqvae_model = SlidingWindowVQVAEConv(
            window_size=window_size,
            n_joints=29,
            joint_dim=2,
            code_num=256,  # Match the pre-trained model
            code_dim=128,
            commitment_cost=0.25,
            width=128,
            depth=3,
            down_t=2,
            stride_t=2,
            dilation_growth_rate=2
        )
        
        # Load the pre-trained weights
        checkpoint = torch.load(model_path, map_location=env.device)
        if 'model_state_dict' in checkpoint:
            env._vqvae_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            env._vqvae_model.load_state_dict(checkpoint)
        
        env._vqvae_model.to(env.device)
        env._vqvae_model.eval()
        
        print(f"[INFO] Loaded VQ-VAE model from {model_path}")
    
    # Prepare motion data window
    # We need to construct the sliding window from current motion data
    batch_size = env.num_envs
    
    # For this implementation, we'll use the current timestep and create a window
    # by sampling from the motion data around the current time
    motion_window = torch.zeros(batch_size, window_size, 29, 2, device=env.device)
    
    for env_idx in range(batch_size):
        motion_idx = int(command.current_motion_indices[env_idx].item())
        current_timestep = int(command.time_steps[env_idx].item())
        # Get the motion data for this environment
        motion_data = command.motion_loader.motions[motion_idx]
        motion_joint_pos = motion_data['joint_pos']  # (T, n_joints)
        motion_joint_vel = motion_data['joint_vel']  # (T, n_joints)
        
        # Create a window around the current timestep
        start_idx = max(0, current_timestep - window_size // 2)
        end_idx = min(motion_joint_pos.shape[0], start_idx + window_size)
        actual_length = int(end_idx - start_idx)
        
        # Fill the window with motion data
        if actual_length > 0:
            # Take joint positions and velocities
            window_pos = motion_joint_pos[start_idx:end_idx]  # (actual_length, n_joints)
            window_vel = motion_joint_vel[start_idx:end_idx]  # (actual_length, n_joints)
            
            # Combine pos and vel in the last dimension
            window_data = torch.stack([window_pos, window_vel], dim=-1)  # (actual_length, n_joints, 2)
            
            # Place in the motion window, pad if necessary
            if actual_length >= window_size:
                motion_window[env_idx] = window_data[:window_size]
            else:
                # Pad by repeating the last frame
                motion_window[env_idx, :actual_length] = window_data
                if actual_length > 0:
                    pad_length = int(window_size - actual_length)
                    motion_window[env_idx, actual_length:] = window_data[-1].unsqueeze(0).repeat(pad_length, 1, 1)
    
    # Encode using VQ-VAE (get latent representations before quantization)
    with torch.no_grad():
        codes, latents = env._vqvae_model.encode(motion_window)
    
    # Concatenate lower and upper body latents
    lower_latent = latents['lower']  # (batch_size, code_dim, T_reduced)
    upper_latent = latents['upper']  # (batch_size, code_dim, T_reduced)
    
    # Global average pooling to get fixed-size representation
    lower_pooled = torch.mean(lower_latent, dim=2)  # (batch_size, code_dim)
    upper_pooled = torch.mean(upper_latent, dim=2)  # (batch_size, code_dim)
    
    # Concatenate lower and upper body features
    latent_features = torch.cat([lower_pooled, upper_pooled], dim=1)  # (batch_size, 2 * code_dim)
    
    return latent_features.view(env.num_envs, -1)