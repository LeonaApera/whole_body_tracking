"""
Multi-motion loader for random trajectory sampling
"""

import os
import glob
import numpy as np
import torch
from typing import Sequence, Union, List
import wandb


class MultiMotionLoader:
    """加载多个 motion 文件，支持随机采样 trajectory"""
    
    def __init__(self, motion_source: Union[str, List[str]], body_indexes: Sequence[int], device: str = "cpu"):
        """
        Args:
            motion_source: 单个 motion 文件路径，或包含多个文件的目录，或文件路径列表
            body_indexes: body indices
            device: torch device
        """
        self.device = device
        self.body_indexes = body_indexes
        self.motions = []
        self.motion_names = []
        
        if isinstance(motion_source, str):
            if os.path.isfile(motion_source):
                # 单个文件
                self._load_motion_file(motion_source)
            elif os.path.isdir(motion_source):
                # 目录，加载所有 .npz 文件
                motion_files = glob.glob(os.path.join(motion_source, "*.npz"))
                for motion_file in sorted(motion_files):
                    self._load_motion_file(motion_file)
            else:
                # 可能是 wandb registry pattern
                self._load_from_wandb_registry(motion_source)
        elif isinstance(motion_source, list):
            # 文件路径列表
            for motion_file in motion_source:
                self._load_motion_file(motion_file)
        else:
            raise ValueError(f"Invalid motion_source type: {type(motion_source)}")
            
        if not self.motions:
            raise ValueError("No motion files loaded")
            
        print(f"[INFO] Loaded {len(self.motions)} motion files: {self.motion_names}")
        
    def _load_motion_file(self, motion_file: str):
        """加载单个 motion 文件"""
        try:
            data = np.load(motion_file)
            motion_data = {
                'fps': data['fps'],
                'joint_pos': torch.tensor(data['joint_pos'], dtype=torch.float32, device=self.device),
                'joint_vel': torch.tensor(data['joint_vel'], dtype=torch.float32, device=self.device),
                'body_pos_w': torch.tensor(data['body_pos_w'], dtype=torch.float32, device=self.device),
                'body_quat_w': torch.tensor(data['body_quat_w'], dtype=torch.float32, device=self.device),
                'body_lin_vel_w': torch.tensor(data['body_lin_vel_w'], dtype=torch.float32, device=self.device),
                'body_ang_vel_w': torch.tensor(data['body_ang_vel_w'], dtype=torch.float32, device=self.device),
                'time_step_total': data['joint_pos'].shape[0]
            }
            
            self.motions.append(motion_data)
            self.motion_names.append(os.path.basename(motion_file).replace('.npz', ''))
            
        except Exception as e:
            print(f"[WARNING] Failed to load motion file {motion_file}: {e}")
    
    def _load_from_wandb_registry(self, registry_pattern: str):
        """从 wandb registry 加载多个 motion"""
        try:
            # 解析 registry pattern，例如 "org/collection/*"
            parts = registry_pattern.split('/')
            if len(parts) < 2 or '*' not in registry_pattern:
                raise ValueError("Registry pattern should be like 'org/collection/*'")
                
            api = wandb.Api()
            
            # 获取 collection 中的所有 artifacts
            if '*' in parts[-1]:
                collection_path = '/'.join(parts[:-1])
                artifacts = api.artifacts(collection_path)
                
                for artifact in artifacts:
                    try:
                        # 下载并加载 artifact
                        artifact_path = artifact.download()
                        motion_file = os.path.join(artifact_path, "motion.npz")
                        if os.path.exists(motion_file):
                            self._load_motion_file(motion_file)
                    except Exception as e:
                        print(f"[WARNING] Failed to load artifact {artifact.name}: {e}")
            else:
                # 单个 artifact
                artifact = api.artifact(registry_pattern)
                artifact_path = artifact.download()
                motion_file = os.path.join(artifact_path, "motion.npz")
                if os.path.exists(motion_file):
                    self._load_motion_file(motion_file)
                    
        except Exception as e:
            print(f"[ERROR] Failed to load from wandb registry {registry_pattern}: {e}")
    
    def sample_motion_and_timestep(self, num_samples: int = 1):
        """
        随机采样 motion 和 timestep
        
        Args:
            num_samples: 采样数量
            
        Returns:
            tuple: (motion_indices, timesteps)
        """
        # 随机选择 motion
        motion_indices = torch.randint(0, len(self.motions), (num_samples,), device=self.device)
        
        # 为每个选择的 motion 随机选择起始时间步
        timesteps = torch.zeros(num_samples, dtype=torch.long, device=self.device)
        for i, motion_idx in enumerate(motion_indices):
            max_timestep = self.motions[motion_idx.item()]['time_step_total'] - 1
            timesteps[i] = torch.randint(0, max_timestep + 1, (1,), device=self.device)
        
        return motion_indices, timesteps
    
    def get_motion_data(self, motion_idx: int, timestep: int):
        """
        获取指定 motion 和 timestep 的数据
        
        Args:
            motion_idx: motion index
            timestep: time step
            
        Returns:
            dict: motion data at timestep
        """
        motion = self.motions[motion_idx]
        timestep = min(timestep, motion['time_step_total'] - 1)
        
        return {
            'joint_pos': motion['joint_pos'][timestep],
            'joint_vel': motion['joint_vel'][timestep],
            'body_pos_w': motion['body_pos_w'][timestep, self.body_indexes],
            'body_quat_w': motion['body_quat_w'][timestep, self.body_indexes],
            'body_lin_vel_w': motion['body_lin_vel_w'][timestep, self.body_indexes],
            'body_ang_vel_w': motion['body_ang_vel_w'][timestep, self.body_indexes],
            'max_timestep': motion['time_step_total'] - 1
        }
    
    @property
    def num_motions(self) -> int:
        return len(self.motions)
    
    @property
    def max_time_steps(self) -> int:
        return max(motion['time_step_total'] for motion in self.motions)
