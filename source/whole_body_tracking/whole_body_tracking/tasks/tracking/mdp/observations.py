from __future__ import annotations

import torch
from typing import TYPE_CHECKING
import pdb
from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms, quat_rotate_inverse

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
    lower_latent = lower_latent[:,:,0]  # (batch_size, code_dim)
    upper_latent = upper_latent[:,:,0]  # (batch_size, code_dim)
    # Concatenate lower and upper body features
    latent_features = torch.cat([lower_latent, upper_latent], dim=1)  # (batch_size, 2 * code_dim)

    return latent_features.view(env.num_envs, -1)

def latent_space_67_preprocessed(env: ManagerBasedEnv, 
                                command_name: str,
                                preprocessed_file_path: str) -> torch.Tensor:
    """
    Read preprocessed VQ-VAE latent features from file (OPTIMIZED VERSION).
    
    This function loads pre-computed latent features to avoid real-time VQ-VAE inference,
    significantly improving training and inference speed. Uses vectorized operations
    for maximum performance.
    
    Args:
        env: The environment instance
        command_name: Name of the motion command to use
        preprocessed_file_path: Path to the preprocessed latent features file (.pkl)

    Returns:
        Latent features for current timestep
        Shape: (num_envs, latent_dim)
    """
    import pickle
    import os
    
    # Initialize preprocessed features loader if not already loaded
    # OPTIMIZATION: Load all features to GPU once, avoiding repeated conversions
    if not hasattr(env, '_preprocessed_latents_gpu'):
        if not os.path.exists(preprocessed_file_path):
            raise FileNotFoundError(f"Preprocessed features file not found: {preprocessed_file_path}")
        
        with open(preprocessed_file_path, 'rb') as f:
            preprocessed_data = pickle.load(f)
        
        # Convert all latent features to GPU tensor at once
        latent_list = preprocessed_data['latent_features']
        if isinstance(latent_list[0], torch.Tensor):
            latent_tensor = torch.stack(latent_list).to(env.device)
        else:
            latent_tensor = torch.tensor(latent_list, device=env.device, dtype=torch.float32)
        
        env._preprocessed_latents_gpu = latent_tensor  # Shape: (T, latent_dim)
        env._preprocessed_metadata = preprocessed_data['metadata']
        
        print(f"[INFO] Loaded and optimized preprocessed latent features")
        print(f"  Motion: {env._preprocessed_metadata['filename']}")
        print(f"  Shape: {latent_tensor.shape}")
        print(f"  Device: {latent_tensor.device}")
    
    # Get the motion command to determine current timestep
    command: MotionCommand = env.command_manager.get_term(command_name)
    
    # OPTIMIZATION: Get all timesteps at once (vectorized)
    if hasattr(command, 'time_steps'):
        timesteps = command.time_steps  # Shape: (num_envs,)
    elif hasattr(command, 'time_step'):
        if hasattr(command.time_step, '__getitem__'):
            timesteps = command.time_step
        else:
            timesteps = command.time_step.expand(env.num_envs)
    else:
        # Fallback: assume all environments at timestep 0
        timesteps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    
    # OPTIMIZATION: Clamp timesteps to valid range to avoid out-of-bounds errors
    max_timestep = env._preprocessed_latents_gpu.shape[0] - 1
    timesteps = torch.clamp(timesteps, 0, max_timestep)
    
    # OPTIMIZATION: Batch indexing - single GPU operation instead of 8192 loops!
    latent_features = env._preprocessed_latents_gpu[timesteps]  # Shape: (num_envs, latent_dim)
    
    return latent_features.view(env.num_envs, -1)


# def vqvae_latent_codes(env: ManagerBasedEnv, 
#                       command_name: str,
#                       vqvae_data_dir: str = "/home/yuxin/Projects/VQVAE/VAE/58_concat_32dim",
#                       dim: int = 32) -> torch.Tensor:
#     """
#     Load VQ-VAE latent codes based on current timestep (OPTIMIZED VERSION).
    
#     每4步对应一个latent code，所以用current_step//4来索引。
#     从VQ-VAE推理结果文件中加载32维特征，每4个仿真步骤使用一个代码。
#     兼容单轨迹和多轨迹命令。优化版本：使用向量化操作避免循环，显著提高性能。
    
#     Args:
#         env: The environment instance
#         command_name: Name of the motion command to use  
#         vqvae_data_dir: Directory containing VQ-VAE latent feature files (.pkl format)
#         dim: Dimension of latent codes (should be 32)
        
#     Returns:
#         Latent codes for current timestep
#         Shape: (num_envs, latent_dim)
#     """
#     import pickle
#     import os
#     import glob
    
#     # Get the motion command (compatible with both single and multi-trajectory commands)
#     command = env.command_manager.get_term(command_name)
    
#     # Initialize VQ-VAE latent data cache if not already loaded (ONLY ONCE)
#     if not hasattr(env, '_vqvae_latent_gpu_cache'):
#         env._vqvae_latent_gpu_cache = {}
        
#         # 获取所有可能的运动名称（兼容多轨迹）
#         motion_names = []
        
#         # 检查是否为多轨迹命令
#         if hasattr(command, 'motions') and hasattr(command, 'motion_files'):
#             # MultiTrajectoryMotionCommand: 从motion_files获取所有运动名称
#             for motion_file in command.cfg.motion_files:
#                 # 从文件路径提取运动名称
#                 motion_name = os.path.splitext(os.path.basename(motion_file))[0]
#                 # 确保格式一致（添加:v0后缀如果没有的话）
#                 if ':v0' not in motion_name:
#                     motion_name = motion_name + ':v0'
#                 motion_names.append(motion_name)
#         else:
#             # 单轨迹命令：使用默认运动名称
#             motion_names = ["dance1_subject2:v0"]
        
#         print(f"[INFO] Loading VQ-VAE latent codes for motions: {motion_names}")
        
#         # 为每个运动加载VQ-VAE文件
#         for motion_name in motion_names:
#             if motion_name in env._vqvae_latent_gpu_cache:
#                 continue  # 已加载，跳过
                
#             # 尝试找到匹配的VQ-VAE文件
#             pkl_files = glob.glob(os.path.join(vqvae_data_dir, f"{motion_name}*_motion_vqvae_58d_concat_concat_w100_d32.pkl"))
            
#             if not pkl_files:
#                 # 尝试不带:v0后缀的名称
#                 base_name = motion_name.replace(':v0', '')
#                 pkl_files = glob.glob(os.path.join(vqvae_data_dir, f"{base_name}*_motion_vqvae_58d_concat_concat_w100_d32.pkl"))
            
#             if not pkl_files:
#                 print(f"[WARNING] No VQ-VAE latent files found for {motion_name} in directory: {vqvae_data_dir}")
#                 # 创建零特征作为备用
#                 env._vqvae_latent_gpu_cache[motion_name] = torch.zeros(1000, dim, device=env.device)  # 假设1000帧
#                 continue
                
#             vqvae_file = pkl_files[0]
#             print(f"[INFO] Loading and optimizing VQ-VAE file: {os.path.basename(vqvae_file)}")
                
#             try:
#                 with open(vqvae_file, 'rb') as f:
#                     data = pickle.load(f)
                
#                 # Extract latent features from the VQ-VAE inference result
#                 if 'latent_features' in data:
#                     latent_features = data['latent_features']
                    
#                     # Handle different data formats
#                     if isinstance(latent_features, list) and len(latent_features) > 0:
#                         latent_features = latent_features[0]
                    
#                     if not isinstance(latent_features, torch.Tensor):
#                         latent_features = torch.tensor(latent_features, dtype=torch.float32)
                    
#                     # 确保形状正确并直接加载到GPU
#                     if hasattr(latent_features, 'shape') and len(latent_features.shape) == 2 and latent_features.shape[1] == dim:
#                         env._vqvae_latent_gpu_cache[motion_name] = latent_features.to(env.device)
#                         print(f"[INFO] Optimized VQ-VAE latent features loaded to GPU for {motion_name}: {latent_features.shape}")
#                     else:
#                         shape_info = getattr(latent_features, 'shape', 'unknown')
#                         print(f"[ERROR] Unexpected latent features shape for {motion_name}: {shape_info}, expected (N, {dim})")
#                         # 创建零特征作为备用
#                         env._vqvae_latent_gpu_cache[motion_name] = torch.zeros(1000, dim, device=env.device)
#                 else:
#                     print(f"[ERROR] No 'latent_features' key found in {vqvae_file}")
#                     # 创建零特征作为备用
#                     env._vqvae_latent_gpu_cache[motion_name] = torch.zeros(1000, dim, device=env.device)
                    
#             except Exception as e:
#                 print(f"[ERROR] Failed to load VQ-VAE latent file {vqvae_file}: {e}")
#                 # 创建零特征作为备用
#                 env._vqvae_latent_gpu_cache[motion_name] = torch.zeros(1000, dim, device=env.device)
    
#     # 向量化处理：获取所有环境的时间步和运动索引
#     if hasattr(command, 'time_steps'):
#         timesteps = command.time_steps  # Shape: (num_envs,)
#     else:
#         timesteps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    
#     # 处理运动索引（兼容单轨迹和多轨迹）
#     if hasattr(command, 'current_motion_indices'):
#         # 多轨迹命令
#         motion_indices = command.current_motion_indices  # Shape: (num_envs,)
        
#         # 为每个环境获取对应的潜在代码
#         latent_codes = torch.zeros(env.num_envs, dim, device=env.device)
        
#         for env_idx in range(env.num_envs):
#             motion_idx = motion_indices[env_idx].item()
#             timestep = timesteps[env_idx].item()
            
#             # 获取运动名称
#             if hasattr(command, 'cfg') and hasattr(command.cfg, 'motion_files') and motion_idx < len(command.cfg.motion_files):
#                 motion_file = command.cfg.motion_files[motion_idx]
#                 motion_name = os.path.splitext(os.path.basename(motion_file))[0]
#                 if ':v0' not in motion_name:
#                     motion_name = motion_name + ':v0'
#             else:
#                 motion_name = "dance1_subject2:v0"  # 默认备用
            
#             # 获取缓存的潜在特征
#             if motion_name in env._vqvae_latent_gpu_cache:
#                 cached_latents = env._vqvae_latent_gpu_cache[motion_name]  # Shape: (T, dim)
                
#                 # 每4步对应一个latent code
#                 latent_idx = timestep // 4
#                 max_latent_idx = cached_latents.shape[0] - 1
#                 latent_idx = min(max_latent_idx, max(0, latent_idx))
                
#                 latent_codes[env_idx] = cached_latents[latent_idx]
#     else:
#         # 单轨迹命令
#         motion_name = "dance1_subject2:v0"  # 默认运动名称
        
#         if motion_name not in env._vqvae_latent_gpu_cache:
#             return torch.zeros(env.num_envs, dim, device=env.device)
        
#         cached_latents = env._vqvae_latent_gpu_cache[motion_name]  # Shape: (T, dim)
        
#         # 每4步对应一个latent code（向量化）
#         latent_indices = timesteps // 4
        
#         # 限制到有效范围（向量化）
#         max_latent_idx = cached_latents.shape[0] - 1
#         latent_indices = torch.clamp(latent_indices, 0, max_latent_idx)
        
#         # 批量索引：单次GPU操作
#         latent_codes = cached_latents[latent_indices]  # Shape: (num_envs, dim)
    
#     return latent_codes.view(env.num_envs, -1)


def predicted_quaternions_rotation_6d(env: ManagerBasedEnv, 
                                     command_name: str,
                                     quat_inference_dir: str = "/home/yuxin/Projects/VQVAE/VAE/code_to_quat_checkpoints/code_to_quat_20250927_013708/quat_inference") -> torch.Tensor:
    """
    Load predicted quaternions and convert to 6D rotation representation (first 2 columns of rotation matrix).
    
    每个时间步获取一个4维四元数，然后转换为旋转矩阵的前两列(6维)。
    VQ-VAE编码是每4步一个，但四元数预测是每步一个(因为K=4)。
    
    Args:
        env: The environment instance
        command_name: Name of the motion command to use  
        quat_inference_dir: Directory containing quaternion inference files (.pkl format)
        
    Returns:
        6D rotation features for current timestep (first 2 columns of rotation matrix)
        Shape: (num_envs, 6)
    """
    import pickle
    import os
    import glob
    
    # Get the motion command
    command: MotionCommand = env.command_manager.get_term(command_name)
    
    # Initialize quaternion data cache if not already loaded
    if not hasattr(env, '_quaternion_cache'):
        env._quaternion_cache = {}
    
    batch_size = env.num_envs
    rotation_6d = torch.zeros(batch_size, 6, device=env.device)  # 6D rotation representation
    
    for env_idx in range(batch_size):
        # Determine motion index and current timestep
        if hasattr(command, 'current_motion_indices'):
            motion_idx = int(command.current_motion_indices[env_idx].item())
        else:
            motion_idx = 0
            
        current_timestep = int(command.time_steps[env_idx].item())
        
        # Get motion name - try to map from motion file if available
        motion_name = None
        try:
            if hasattr(command, 'motion_loader') and hasattr(command.motion_loader, 'motions'):
                # Multi-motion case: try to get motion filename
                if motion_idx < len(command.motion_loader.motions):
                    motion_data = command.motion_loader.motions[motion_idx]
                    if isinstance(motion_data, dict) and 'filename' in motion_data:
                        filename = motion_data['filename']
                        # Extract motion name from filename (remove extension and path)
                        motion_name = os.path.splitext(os.path.basename(filename))[0]
                        # Remove common suffixes
                        if motion_name.endswith(':v0'):
                            motion_name = motion_name
                        elif ':' in motion_name:
                            motion_name = motion_name.split(':')[0] + ':v0'
        except Exception:
            pass
        
        # Fallback to default motion name if not found
        if motion_name is None:
            motion_name = "dance1_subject1:v0"  # Default fallback
        
        # Load quaternions if not cached
        cache_key = motion_name
        if cache_key not in env._quaternion_cache:
            # 查找四元数推理文件
            quat_files = glob.glob(os.path.join(quat_inference_dir, f"{motion_name}*_quaternions.pkl"))
            
            if not quat_files:
                # Try without :v0 suffix
                base_name = motion_name.replace(':v0', '')
                quat_files = glob.glob(os.path.join(quat_inference_dir, f"{base_name}*_quaternions.pkl"))
            
            if not quat_files:
                print(f"[WARNING] No quaternion files found for {motion_name} in directory: {quat_inference_dir}")
                # Return zeros for this environment instead of skipping
                rotation_6d[env_idx] = torch.zeros(6, device=env.device)
                continue
                
            quat_file = quat_files[0]  # 使用找到的第一个文件
            print(f"[INFO] Loading quaternion file: {os.path.basename(quat_file)}")
                
            try:
                with open(quat_file, 'rb') as f:
                    data = pickle.load(f)
                
                # Extract quaternions from the inference result
                if 'quaternions' in data:
                    quaternions = data['quaternions']  # Shape: (codes_num * K, 4)
                    
                    if not isinstance(quaternions, torch.Tensor):
                        quaternions = torch.tensor(quaternions, dtype=torch.float32)
                    
                    # Ensure we have the correct shape: (total_quats, 4)
                    if len(quaternions.shape) == 2 and quaternions.shape[1] == 4:
                        env._quaternion_cache[cache_key] = quaternions.to(env.device)
                        
                        # 获取元数据用于调试
                        metadata = data['metadata']
                        codes_num = metadata['codes_num']
                        K = metadata['K']
                        print(f"[INFO] Loaded quaternions for {motion_name}: {quaternions.shape}")
                        print(f"  - Original codes: {codes_num}")
                        print(f"  - K (quats per code): {K}")
                        print(f"  - Total quaternions: {quaternions.shape[0]}")
                    else:
                        print(f"[ERROR] Unexpected quaternion shape: {quaternions.shape}, expected (N, 4)")
                        continue
                else:
                    print(f"[ERROR] No 'quaternions' key found in {quat_file}")
                    continue
                    
            except Exception as e:
                print(f"[ERROR] Failed to load quaternion file {quat_file}: {e}")
                continue
        
        # Get quaternions from cache
        if cache_key in env._quaternion_cache:
            cached_quaternions = env._quaternion_cache[cache_key]
            
            # 每个时间步对应一个四元数
            # 因为VQ-VAE编码是每4步一个，但我们有K=4个四元数每个编码
            # 所以总共有 codes_num * K 个四元数，每个时间步一个
            quat_idx = current_timestep
            
            # Clamp to valid range
            max_quat_idx = cached_quaternions.shape[0] - 1
            quat_idx = min(max_quat_idx, max(0, quat_idx))
            
            # Get the quaternion for this timestep
            quaternion = cached_quaternions[quat_idx]  # Shape: (4,)
            
            # Convert quaternion to rotation matrix
            rotation_matrix = quaternion_to_rotation_matrix_single(quaternion)  # Shape: (3, 3)
            
            # Extract first 2 columns and flatten to 6D representation
            rotation_6d_single = rotation_matrix[:, :2].flatten()  # Shape: (6,)
            
            rotation_6d[env_idx] = rotation_6d_single
    
    return rotation_6d.view(env.num_envs, -1)


def quaternion_to_rotation_matrix_single(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert a single quaternion to rotation matrix.
    
    Args:
        quat: (4,) quaternion in (w, x, y, z) or (x, y, z, w) format
        
    Returns:
        rotation matrix of shape (3, 3)
    """
    # Normalize quaternion
    quat = quat / (quat.norm().clamp(min=1e-8))
    
    # Assume quaternion is in (w, x, y, z) format based on the data we saw
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    
    # Compute rotation matrix elements
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    
    # Build rotation matrix
    R = torch.zeros(3, 3, device=quat.device, dtype=quat.dtype)
    
    R[0, 0] = 1 - 2*(yy + zz)
    R[0, 1] = 2*(xy - wz)
    R[0, 2] = 2*(xz + wy)
    
    R[1, 0] = 2*(xy + wz)
    R[1, 1] = 1 - 2*(xx + zz)
    R[1, 2] = 2*(yz - wx)
    
    R[2, 0] = 2*(xz - wy)
    R[2, 1] = 2*(yz + wx)
    R[2, 2] = 1 - 2*(xx + yy)
    
    return R


def predicted_anchor_ori_b(env: ManagerBasedEnv, 
                          command_name: str,
                          quat_inference_dir: str = "/home/yuxin/Projects/VQVAE/VAE/code_to_quat_checkpoints/code_to_quat_20250927_013708/quat_inference") -> torch.Tensor:
    """
    Calculate orientation using predicted quaternions instead of command.anchor_quat_w (OPTIMIZED VERSION).
    
    This function replaces the anchor quaternion with our predicted quaternion and computes
    the relative orientation in robot body frame, similar to motion_anchor_ori_b but using
    predicted quaternions. 优化版本：使用向量化操作避免循环，显著提高性能。
    
    Args:
        env: The environment instance
        command_name: Name of the motion command to use
        quat_inference_dir: Directory containing quaternion inference files (.pkl format)
        
    Returns:
        Relative orientation (first 2 columns of rotation matrix)
        Shape: (num_envs, 6)
    """
    import pickle
    import os
    import glob
    
    # Get the motion command
    command: MotionCommand = env.command_manager.get_term(command_name)
    
    # Initialize quaternion data cache if not already loaded (ONLY ONCE)
    if not hasattr(env, '_predicted_quaternion_gpu_cache'):
        env._predicted_quaternion_gpu_cache = {}
        
        # 预加载所有需要的四元数文件到GPU
        motion_name = "dance1_subject2:v0"  # 简化为单一motion
        quat_files = glob.glob(os.path.join(quat_inference_dir, f"{motion_name}*_quaternions.pkl"))
        
        if not quat_files:
            # Try without :v0 suffix
            base_name = motion_name.replace(':v0', '')
            quat_files = glob.glob(os.path.join(quat_inference_dir, f"{base_name}*_quaternions.pkl"))
        
        if not quat_files:
            print(f"[ERROR] No quaternion files found for {motion_name} in directory: {quat_inference_dir}")
            return torch.zeros(env.num_envs, 6, device=env.device)
            
        quat_file = quat_files[0]
        print(f"[INFO] Loading and optimizing predicted quaternion file: {os.path.basename(quat_file)}")
            
        try:
            with open(quat_file, 'rb') as f:
                data = pickle.load(f)
            
            # Extract quaternions from the inference result
            if 'quaternions' in data:
                quaternions = data['quaternions']  # Shape: (codes_num * K, 4)
                
                if not isinstance(quaternions, torch.Tensor):
                    quaternions = torch.tensor(quaternions, dtype=torch.float32)
                
                # 确保形状正确并直接加载到GPU
                if len(quaternions.shape) == 2 and quaternions.shape[1] == 4:
                    env._predicted_quaternion_gpu_cache[motion_name] = quaternions.to(env.device)
                    
                    # 获取元数据用于调试
                    metadata = data['metadata']
                    codes_num = metadata['codes_num']
                    K = metadata['K']
                    print(f"[INFO] Optimized predicted quaternions loaded to GPU: {quaternions.shape}")
                    print(f"  - Original codes: {codes_num}, K: {K}, Total quaternions: {quaternions.shape[0]}")
                else:
                    print(f"[ERROR] Unexpected quaternion shape: {quaternions.shape}, expected (N, 4)")
                    return torch.zeros(env.num_envs, 6, device=env.device)
            else:
                print(f"[ERROR] No 'quaternions' key found in {quat_file}")
                return torch.zeros(env.num_envs, 6, device=env.device)
                
        except Exception as e:
            print(f"[ERROR] Failed to load quaternion file {quat_file}: {e}")
            return torch.zeros(env.num_envs, 6, device=env.device)
    
    # 向量化处理：获取所有环境的时间步
    motion_name = "dance1_subject2:v0"
    
    if motion_name not in env._predicted_quaternion_gpu_cache:
        return torch.zeros(env.num_envs, 6, device=env.device)
    
    cached_quaternions = env._predicted_quaternion_gpu_cache[motion_name]  # Shape: (T, 4)
    
    # 获取所有环境的时间步（向量化）
    if hasattr(command, 'time_steps'):
        timesteps = command.time_steps  # Shape: (num_envs,)
    else:
        timesteps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    
    # 限制到有效范围（向量化）
    max_quat_idx = cached_quaternions.shape[0] - 1
    quat_indices = torch.clamp(timesteps, 0, max_quat_idx)
    
    # 批量获取预测四元数：单次GPU操作替代4096次循环！
    predicted_quats = cached_quaternions[quat_indices]  # Shape: (num_envs, 4)
    
    # 获取机器人锚点四元数
    robot_anchor_quats = command.robot_anchor_quat_w  # Shape: (num_envs, 4)
    
    # 向量化计算相对方向：批量四元数运算
    robot_anchor_quats_inv = quaternion_inverse_batch(robot_anchor_quats)
    relative_quats = quaternion_multiply_batch(robot_anchor_quats_inv, predicted_quats)
    
    # 向量化转换为旋转矩阵并提取前两列
    rotation_matrices = quaternion_to_rotation_matrix_batch(relative_quats)  # Shape: (num_envs, 3, 3)
    ori_6d = rotation_matrices[:, :, :2].reshape(env.num_envs, 6)  # Shape: (num_envs, 6)
    
    return ori_6d.view(env.num_envs, -1)


def quaternion_inverse_single(q: torch.Tensor) -> torch.Tensor:
    """
    Quaternion inverse (conjugate for unit quaternions).
    
    Args:
        q: (4,) quaternion in (w, x, y, z) format
        
    Returns:
        Inverse quaternion
    """
    q_inv = q.clone()
    q_inv[1:] *= -1  # Negate x, y, z components
    return q_inv


def quaternion_multiply_single(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Quaternion multiplication: q1 * q2.
    
    Args:
        q1, q2: (4,) quaternions in (w, x, y, z) format
        
    Returns:
        Product quaternion
    """
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return torch.stack([w, x, y, z])


def quaternion_inverse_batch(q: torch.Tensor) -> torch.Tensor:
    """
    Batch quaternion inverse (conjugate for unit quaternions).
    
    Args:
        q: (batch_size, 4) quaternions in (w, x, y, z) format
        
    Returns:
        Inverse quaternions
    """
    q_inv = q.clone()
    q_inv[:, 1:] *= -1  # Negate x, y, z components for all quaternions
    return q_inv


def quaternion_multiply_batch(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Batch quaternion multiplication: q1 * q2.
    
    Args:
        q1, q2: (batch_size, 4) quaternions in (w, x, y, z) format
        
    Returns:
        Product quaternions
    """
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return torch.stack([w, x, y, z], dim=1)


def quaternion_to_rotation_matrix_batch(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert batch of quaternions to rotation matrices.
    
    Args:
        quat: (batch_size, 4) quaternions in (w, x, y, z) format
        
    Returns:
        Rotation matrices of shape (batch_size, 3, 3)
    """
    batch_size = quat.shape[0]
    
    # Normalize quaternions
    quat = quat / (quat.norm(dim=1, keepdim=True).clamp(min=1e-8))
    
    # Assume quaternion is in (w, x, y, z) format
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    
    # Compute rotation matrix elements
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    
    # Build rotation matrices
    R = torch.zeros(batch_size, 3, 3, device=quat.device, dtype=quat.dtype)
    
    R[:, 0, 0] = 1 - 2*(yy + zz)
    R[:, 0, 1] = 2*(xy - wz)
    R[:, 0, 2] = 2*(xz + wy)
    
    R[:, 1, 0] = 2*(xy + wz)
    R[:, 1, 1] = 1 - 2*(xx + zz)
    R[:, 1, 2] = 2*(yz - wx)
    
    R[:, 2, 0] = 2*(xz - wy)
    R[:, 2, 1] = 2*(yz + wx)
    R[:, 2, 2] = 1 - 2*(xx + yy)
    
    return R


def vqvae_latent_codes(env: ManagerBasedEnv, 
                      command_name: str,
                   ) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.latent_feature
def motion_anchor_ori_b_infer(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.global_root_pos,
        command.global_root_quat,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)
    
def motion_anchor_pos_b_infer(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.global_root_pos,
        command.global_root_quat,
    )

    return pos.view(env.num_envs, -1)