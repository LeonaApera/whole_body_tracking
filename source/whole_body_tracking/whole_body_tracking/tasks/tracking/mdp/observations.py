from __future__ import annotations

import torch
from typing import TYPE_CHECKING
import pdb
from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

def command_future_frames(env: ManagerBasedEnv, command_name: str, n_future_frames: int = 10) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    batch_size = env.num_envs

    # Try to determine number of joints from available sources
    if hasattr(command.cfg, 'n_joints'):
        n_joints = command.cfg.n_joints
            # MultiTrajectoryMotionCommand uses motion_loader.motions list
    if hasattr(command, 'motion_loader') and len(command.motion_loader.motions) > 0:
        n_joints = command.motion_loader.motions[0]['joint_pos'].shape[1]
    else:
        # single MotionCommand: command.motion.joint_pos exists
        n_joints = command.motion.joint_pos.shape[-1]

    future_frames = torch.zeros((batch_size, n_future_frames, n_joints, 2), device=env.device)

    for env_idx in range(batch_size):
        # determine motion index and current timestep robustly
        # Multi-trajectory command provides current_motion_indices per-env
        if hasattr(command, 'current_motion_indices'):
            motion_idx = int(command.current_motion_indices[env_idx].item())
        else:
            # for single-motion command use a single motion index (0)
            motion_idx = 0
        # time step
        if hasattr(command, 'time_steps'):
            current_timestep = int(command.time_steps[env_idx].item())
        elif hasattr(command, 'time_step'):
            current_timestep = int(command.time_step)


        # fetch motion data safely depending on loader type
        motion_joint_pos = None
        motion_joint_vel = None
        try:
            if hasattr(command, 'motion_loader') and hasattr(command.motion_loader, 'motions'):
                motion_data = command.motion_loader.motions[motion_idx]
                motion_joint_pos = motion_data['joint_pos']
                motion_joint_vel = motion_data['joint_vel']
            elif hasattr(command, 'motion'):
                # single MotionCommand uses motion.* arrays indexed by time_steps
                motion_joint_pos = command.motion.joint_pos
                motion_joint_vel = command.motion.joint_vel
            else:
                # last resort: attempt to read command.joint_pos/joint_vel as full tensors
                if hasattr(command, 'joint_pos'):
                    motion_joint_pos = command.joint_pos
                if hasattr(command, 'joint_vel'):
                    motion_joint_vel = command.joint_vel
        except Exception:
            motion_joint_pos = None
            motion_joint_vel = None

        if motion_joint_pos is None or motion_joint_vel is None:
            # nothing we can do for this env, leave zeros
            continue

        # motion_joint_* may be torch tensors or numpy arrays; ensure indexing works
        # For multi-motion loader entries, they are numpy arrays or torch tensors with shape (T, n_joints)
        T = motion_joint_pos.shape[0]
        start_idx = current_timestep + 1
        end_idx = min(T, start_idx + n_future_frames)
        actual_length = int(max(0, end_idx - start_idx))

        if actual_length > 0:
            window_pos = motion_joint_pos[start_idx:end_idx]
            window_vel = motion_joint_vel[start_idx:end_idx]

            # ensure tensors
            if not isinstance(window_pos, torch.Tensor):
                window_pos = torch.tensor(window_pos, dtype=torch.float32, device=env.device)
            if not isinstance(window_vel, torch.Tensor):
                window_vel = torch.tensor(window_vel, dtype=torch.float32, device=env.device)

            window_data = torch.stack([window_pos, window_vel], dim=-1)  # (actual_length, n_joints, 2)

            if actual_length >= n_future_frames:
                future_frames[env_idx] = window_data[:n_future_frames]
            else:
                future_frames[env_idx, :actual_length] = window_data
                if actual_length > 0:
                    pad_length = int(n_future_frames - actual_length)
                    future_frames[env_idx, actual_length:] = window_data[-1].unsqueeze(0).repeat(pad_length, 1, 1)

    return future_frames.view(env.num_envs, -1)
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
                 n_future_frames: int = 60) -> torch.Tensor:
    """
    Generate latent space representation using pre-trained VQ-VAE model.
    
    Args:
        env: The environment instance
        command_name: Name of the motion command to use
        vqvae_model_path: Path to the pre-trained VQ-VAE model
        n_future_frames: Size of the sliding window (default: 60 frames for 2 seconds at 30fps)

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
            window_size=n_future_frames,
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
    
    # Prepare motion data window using future n frames relative to current timestep.
    batch_size = env.num_envs

    # Determine n_joints robustly from command config or motion data
    n_joints = None
    # prefer command.cfg
    if hasattr(command.cfg, 'n_joints'):
        n_joints = int(command.cfg.n_joints)

    # Inspect motion data to infer joint count if needed
    def _infer_from_motion_data(motion_source):
        try:
            # If it's an ndarray or torch tensor, it will have a 'shape' attribute
            if hasattr(motion_source, 'shape') and len(motion_source.shape) >= 2:
                return int(motion_source.shape[1])
            # If it's a dict-like motion entry (from motion_loader), inspect 'joint_pos'
            if isinstance(motion_source, dict) and 'joint_pos' in motion_source:
                jp = motion_source['joint_pos']
                if hasattr(jp, 'shape') and len(jp.shape) >= 2:
                    return int(jp.shape[1])
        except Exception:
            return None
        return None

    # Try to get a sample motion to infer n_joints
    sample_motion = None
    if hasattr(command, 'motion_loader') and getattr(command.motion_loader, 'motions', None):
        sample_motion = command.motion_loader.motions[0]
    elif hasattr(command, 'motion'):
        sample_motion = getattr(command, 'motion')

    if n_joints is None and sample_motion is not None:
        n_joints = _infer_from_motion_data(sample_motion)

    # Fallback to a conservative default if still unknown
    if n_joints is None:
        n_joints = 29

    # We'll build a future window of length n_future_frames (time major)
    motion_window = torch.zeros(batch_size, n_future_frames, n_joints, 2, device=env.device)

    for env_idx in range(batch_size):
        # Determine motion index and current timestep robustly
        if hasattr(command, 'current_motion_indices'):
            motion_idx = int(command.current_motion_indices[env_idx].item())
        else:
            motion_idx = 0

        if hasattr(command, 'time_steps'):
            current_timestep = int(command.time_steps[env_idx].item())
        elif hasattr(command, 'time_step'):
            current_timestep = int(command.time_step)
        else:
            # If no timestep info, skip
            continue

        # Fetch motion arrays depending on command type
        motion_joint_pos = None
        motion_joint_vel = None
        try:
            if hasattr(command, 'motion_loader') and hasattr(command.motion_loader, 'motions'):
                motion_data = command.motion_loader.motions[motion_idx]
                motion_joint_pos = motion_data.get('joint_pos', None)
                motion_joint_vel = motion_data.get('joint_vel', None)
            elif hasattr(command, 'motion'):
                # single-motion command; assume motion.joint_pos shape (T, n_joints)
                motion_joint_pos = getattr(command.motion, 'joint_pos', None)
                motion_joint_vel = getattr(command.motion, 'joint_vel', None)
            else:
                if hasattr(command, 'joint_pos'):
                    motion_joint_pos = command.joint_pos
                if hasattr(command, 'joint_vel'):
                    motion_joint_vel = command.joint_vel
        except Exception:
            motion_joint_pos = None
            motion_joint_vel = None

        if motion_joint_pos is None or motion_joint_vel is None:
            # nothing we can do for this env, leave zeros
            continue

        # Ensure numpy -> tensor and on correct device
        if not isinstance(motion_joint_pos, torch.Tensor):
            motion_joint_pos = torch.tensor(motion_joint_pos, dtype=torch.float32, device=env.device)
        if not isinstance(motion_joint_vel, torch.Tensor):
            motion_joint_vel = torch.tensor(motion_joint_vel, dtype=torch.float32, device=env.device)

        T = motion_joint_pos.shape[0]
        # future window starts at next frame
        start_idx = current_timestep + 1
        end_idx = min(T, start_idx + n_future_frames)
        actual_length = int(max(0, end_idx - start_idx))

        if actual_length > 0:
            window_pos = motion_joint_pos[start_idx:end_idx]
            window_vel = motion_joint_vel[start_idx:end_idx]
            window_data = torch.stack([window_pos, window_vel], dim=-1)  # (actual_length, n_joints, 2)

            if actual_length >= n_future_frames:
                motion_window[env_idx] = window_data[:n_future_frames]
            else:
                motion_window[env_idx, :actual_length] = window_data
                if actual_length > 0:
                    pad_length = int(n_future_frames - actual_length)
                    motion_window[env_idx, actual_length:] = window_data[-1].unsqueeze(0).repeat(pad_length, 1, 1)
    
    # Encode using VQ-VAE (get latent representations before quantization)
    with torch.no_grad():
        codes, latents = env._vqvae_model.encode(motion_window)
    
    # Concatenate lower and upper body latents
    lower_latent = latents['lower']  # (batch_size, code_dim, T_reduced)
    upper_latent = latents['upper']  # (batch_size, code_dim, T_reduced)
    
    # # Global average pooling to get fixed-size representation
    # lower_pooled = torch.mean(lower_latent, dim=2)  # (batch_size, code_dim)
    # upper_pooled = torch.mean(upper_latent, dim=2)  # (batch_size, code_dim)
    
    # Concatenate lower and upper body features
    latent_features = torch.cat([lower_latent, upper_latent], dim=1)  # (batch_size, 2 * code_dim)

    return latent_features.view(env.num_envs, -1)