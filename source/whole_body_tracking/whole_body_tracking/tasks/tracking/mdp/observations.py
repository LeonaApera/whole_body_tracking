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

def latent_space_67(env: ManagerBasedEnv, 
                 command_name: str,
                 vqvae_model_path: str = "/home/yuxin/Projects/VQVAE/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/vqvae/best_model_32.pt",
                 n_future_frames: int = 100, 
                 dim: int = 32) -> torch.Tensor:
    """
    Generate latent space representation using pre-trained VQ-VAE model with 67-dimensional features.
    
    Args:
        env: The environment instance
        command_name: Name of the motion command to use
        vqvae_model_path: Path to the pre-trained VQ-VAE model checkpoint
        n_future_frames: Size of the sliding window (default: 100 frames)

    Returns:
        Latent features from VQ-VAE encoder
        Shape: (num_envs, latent_dim * reduced_time_steps)
    """
    import os
    import sys
    sys.path.append('/home/yuxin/Projects/VQVAE/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/vqvae')
    from vqvae_dim_experiment import VQVae
    from lafan1_sliding_window_dataset import LAFAN1MotionData
    
    # Get the motion command
    command: MotionCommand = env.command_manager.get_term(command_name)
    
    # Initialize VQ-VAE model if not already loaded
    if not hasattr(env, '_vqvae_67_model'):
        # Load the pre-trained model
        model_path = vqvae_model_path
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"VQ-VAE model not found at {model_path}")
        
        # Create model instance with correct parameters for 67-dimensional features
        env._vqvae_67_model = VQVae(
            nfeats=67,  # 29 joint_pos + 29 joint_vel + 3 anchor_pos + 6 anchor_rot_6d
            quantizer='ema_reset',
            code_num=512,
            code_dim=dim,
            output_emb_width=dim,
            down_t=2,
            stride_t=2,
            width=512,
            depth=3,
            dilation_growth_rate=3,
            norm=None,
            activation='relu'
        )
        
        # Load the pre-trained weights
        checkpoint = torch.load(model_path, map_location=env.device)
        if 'model_state_dict' in checkpoint:
            env._vqvae_67_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            env._vqvae_67_model.load_state_dict(checkpoint)
        
        env._vqvae_67_model.to(env.device)
        env._vqvae_67_model.eval()
        
        print(f"[INFO] Loaded VQ-VAE 67-dim model from {model_path}")
    
    # Prepare motion data window using future n frames relative to current timestep
    batch_size = env.num_envs

    # Determine n_joints robustly from command config or motion data
    n_joints = 29  # Default for LAFAN1 format
    if hasattr(command.cfg, 'n_joints'):
        n_joints = int(command.cfg.n_joints)

    # We'll build a future window of length n_future_frames with 67-dimensional features
    motion_window = torch.zeros(batch_size, n_future_frames, 67, device=env.device)

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
        motion_anchor_pos = None
        motion_anchor_quat = None
        
        try:
            if hasattr(command, 'motion_loader') and hasattr(command.motion_loader, 'motions'):
                motion_data = command.motion_loader.motions[motion_idx]
                motion_joint_pos = motion_data.get('joint_pos', None)
                motion_joint_vel = motion_data.get('joint_vel', None)
                # For anchor data, we need to access body data
                if 'body_pos_w' in motion_data and 'body_quat_w' in motion_data:
                    motion_anchor_pos = motion_data['body_pos_w'][:, 0, :]  # (T, 3) - first body as anchor
                    motion_anchor_quat = motion_data['body_quat_w'][:, 0, :]  # (T, 4)
            elif hasattr(command, 'motion'):
                # single-motion command; assume motion.joint_pos shape (T, n_joints)
                motion_joint_pos = getattr(command.motion, 'joint_pos', None)
                motion_joint_vel = getattr(command.motion, 'joint_vel', None)
                # Try to get anchor data from motion
                if hasattr(command.motion, 'body_pos_w') and hasattr(command.motion, 'body_quat_w'):
                    motion_anchor_pos = command.motion.body_pos_w[:, 0, :]
                    motion_anchor_quat = command.motion.body_quat_w[:, 0, :]
        except Exception:
            motion_joint_pos = None
            motion_joint_vel = None
            motion_anchor_pos = None
            motion_anchor_quat = None

        if motion_joint_pos is None or motion_joint_vel is None:
            # nothing we can do for this env, leave zeros
            continue

        # Ensure numpy -> tensor and on correct device
        if not isinstance(motion_joint_pos, torch.Tensor):
            motion_joint_pos = torch.tensor(motion_joint_pos, dtype=torch.float32, device=env.device)
        if not isinstance(motion_joint_vel, torch.Tensor):
            motion_joint_vel = torch.tensor(motion_joint_vel, dtype=torch.float32, device=env.device)
        
        # Convert anchor quaternion to 6D rotation representation if available
        anchor_rot_6d = None
        if motion_anchor_pos is not None and motion_anchor_quat is not None:
            if not isinstance(motion_anchor_pos, torch.Tensor):
                motion_anchor_pos = torch.tensor(motion_anchor_pos, dtype=torch.float32, device=env.device)
            if not isinstance(motion_anchor_quat, torch.Tensor):
                motion_anchor_quat = torch.tensor(motion_anchor_quat, dtype=torch.float32, device=env.device)
            
            # Convert quaternion to rotation matrix and extract first 2 columns (6D representation)
            # Normalize quaternion
            motion_anchor_quat = motion_anchor_quat / torch.norm(motion_anchor_quat, dim=-1, keepdim=True)
            
            w, x, y, z = motion_anchor_quat[:, 0], motion_anchor_quat[:, 1], motion_anchor_quat[:, 2], motion_anchor_quat[:, 3]
            
            # Compute rotation matrix elements
            xx, yy, zz = x*x, y*y, z*z
            xy, xz, yz = x*y, x*z, y*z
            wx, wy, wz = w*x, w*y, w*z
            
            # Build rotation matrix
            T_anchor = motion_anchor_pos.shape[0]
            R = torch.zeros(T_anchor, 3, 3, device=env.device, dtype=torch.float32)
            
            R[:, 0, 0] = 1 - 2*(yy + zz)
            R[:, 0, 1] = 2*(xy - wz)
            R[:, 0, 2] = 2*(xz + wy)
            
            R[:, 1, 0] = 2*(xy + wz)
            R[:, 1, 1] = 1 - 2*(xx + zz)
            R[:, 1, 2] = 2*(yz - wx)
            
            R[:, 2, 0] = 2*(xz - wy)
            R[:, 2, 1] = 2*(yz + wx)
            R[:, 2, 2] = 1 - 2*(xx + yy)
            
            # Extract first 2 columns and flatten to 6D
            anchor_rot_6d = R[:, :, :2].reshape(T_anchor, 6)  # (T, 6)

        T = motion_joint_pos.shape[0]
        # future window starts at next frame
        start_idx = current_timestep
        end_idx = min(T, start_idx + n_future_frames)
        actual_length = int(max(0, end_idx - start_idx))

        if actual_length > 0:
            window_joint_pos = motion_joint_pos[start_idx:end_idx]  # (actual_length, n_joints)
            window_joint_vel = motion_joint_vel[start_idx:end_idx]  # (actual_length, n_joints)
            
            # 替换逐帧循环
            window_data = torch.cat([
                window_joint_pos,  # (actual_length, 29)
                window_joint_vel,  # (actual_length, 29)
            motion_anchor_pos[start_idx:end_idx] if motion_anchor_pos is not None else torch.zeros(actual_length, 3, device=env.device),
                anchor_rot_6d[start_idx:end_idx] if anchor_rot_6d is not None else torch.zeros(actual_length, 6, device=env.device)
                ], dim=1)  # (actual_length, 67)

            if actual_length >= n_future_frames:
                motion_window[env_idx] = window_data[:n_future_frames]
            else:
                motion_window[env_idx, :actual_length] = window_data
                if actual_length > 0:
                    # Pad with the last frame
                    pad_length = int(n_future_frames - actual_length)
                    motion_window[env_idx, actual_length:] = window_data[-1].unsqueeze(0).repeat(pad_length, 1)
    
    # Encode using VQ-VAE (get latent representations from encoder)
    with torch.no_grad():
        # Forward through encoder to get latent representation
        x_in = motion_window.permute(0, 2, 1)  # (batch_size, 67, n_future_frames)
        x_encoder = env._vqvae_67_model.encoder(x_in)  # (batch_size, code_dim, reduced_time)
        x_quantized, commit_loss, perplexity = env._vqvae_67_model.quantizer(x_encoder)
        # Flatten temporal dimension to get fixed-size representation
        latent_features = x_quantized[:,:,0]

    return latent_features
