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
                # 也查找子目录中的 motion.npz 文件
                subdirs = [d for d in os.listdir(motion_source) if os.path.isdir(os.path.join(motion_source, d))]
                for subdir in subdirs:
                    subdir_path = os.path.join(motion_source, subdir)
                    motion_file = os.path.join(subdir_path, "motion.npz")
                    if os.path.exists(motion_file):
                        motion_files.append(motion_file)
                        
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
        """从 WandB Registry 加载多个 motion 文件"""
        try:
            import wandb
            
            # 解析 registry pattern，例如 "entity/project/*"
            parts = registry_pattern.split('/')
            if len(parts) < 3:
                raise ValueError("Registry pattern should be like 'entity/project/*'")
            
            entity, project_name, artifact_type = parts
            
            print(f"[INFO] Querying artifacts from WandB Registry: entity={entity}, project={project_name}, type={artifact_type}")
            
            # 基于你的上传代码，artifacts 被链接到 wandb-registry-motions
            # 我们需要使用正确的 registry 路径
            registry_name = f"wandb-registry-{artifact_type.replace('*', 'motions')}"
            
            api = wandb.Api()
            
            try:
                # 尝试获取 registry
                print(f"[INFO] Accessing registry: {registry_name}")
                
                # 方法1：尝试通过 runs 获取 artifacts
                # 获取 csv_to_npz 项目的所有 runs
                runs = api.runs(f"{entity}/{project_name}")
                print(f"[INFO] Found {len(runs)} runs in project")
                
                artifacts_found = []
                
                for run in runs:
                    try:
                        # 获取这个 run 记录的 artifacts
                        for artifact in run.logged_artifacts():
                            if artifact.type == 'motions':  # 匹配类型
                                artifacts_found.append(artifact)
                                print(f"[INFO] Found artifact: {artifact.name}")
                        
                        # 也检查链接的 artifacts
                        for artifact in run.used_artifacts():
                            if artifact.type == 'motions':
                                artifacts_found.append(artifact)
                                print(f"[INFO] Found linked artifact: {artifact.name}")
                                
                    except Exception as e:
                        print(f"[WARNING] Error processing run {run.name}: {e}")
                        continue
                
                # 去重 artifacts
                unique_artifacts = {}
                for artifact in artifacts_found:
                    unique_artifacts[artifact.name] = artifact
                
                print(f"[INFO] Found {len(unique_artifacts)} unique artifacts")
                
                # 下载并加载每个 artifact
                for artifact_name, artifact in unique_artifacts.items():
                    try:
                        print(f"[INFO] Downloading artifact: {artifact_name}")
                        artifact_path = artifact.download()
                        motion_file = os.path.join(artifact_path, "motion.npz")
                        if os.path.exists(motion_file):
                            self._load_motion_file(motion_file)
                            print(f"[INFO] Successfully loaded motion: {artifact_name}")
                        else:
                            print(f"[WARNING] Motion file not found in artifact: {artifact_name}")
                    except Exception as e:
                        print(f"[WARNING] Failed to load artifact {artifact_name}: {e}")
                        continue
                
            except Exception as e:
                print(f"[WARNING] Failed to access registry via runs: {e}")
                
                # 方法2：尝试直接通过名称访问 artifacts
                print("[INFO] Trying alternative approach...")
                try:
                    # 尝试获取已知的一些 artifact 名称
                    known_names = [
                        'dance1_subject1', 'dance2_subject1', 'dance3_subject1', 'dance4_subject1',
                        'walk1_subject1', 'walk2_subject1', 'run1_subject1'
                    ]
                    
                    for name in known_names:
                        try:
                            artifact = api.artifact(f"{entity}/{project_name}/{name}:latest", type='motions')
                            print(f"[INFO] Downloading artifact: {name}")
                            artifact_path = artifact.download()
                            motion_file = os.path.join(artifact_path, "motion.npz")
                            if os.path.exists(motion_file):
                                self._load_motion_file(motion_file)
                                print(f"[INFO] Successfully loaded motion: {name}")
                        except Exception as e:
                            print(f"[WARNING] Could not load {name}: {e}")
                            continue
                            
                except Exception as e:
                    print(f"[WARNING] Alternative approach also failed: {e}")
            
            if not self.motions:
                raise ValueError("No motion files loaded from WandB Registry")
            
            print(f"[INFO] Successfully loaded {len(self.motions)} motion files from WandB Registry")
        
        except Exception as e:
            print(f"[ERROR] Failed to load from WandB Registry: {e}")
            import traceback
            traceback.print_exc()
    
    def _load_from_entity_collection(self, entity: str, collection_name: str, api):
        """从指定 entity 和 collection 加载所有 artifacts"""
        try:
            # 尝试通过 entity 的 artifacts 来查找
            entity_obj = api.entity(entity)
            
            # 获取所有项目
            for project in entity_obj.projects():
                try:
                    artifacts = api.artifacts(f"{entity}/{project.name}/{collection_name}")
                    for artifact in artifacts:
                        try:
                            print(f"[INFO] Downloading artifact: {artifact.name}")
                            artifact_path = artifact.download()
                            motion_file = os.path.join(artifact_path, "motion.npz")
                            if os.path.exists(motion_file):
                                self._load_motion_file(motion_file)
                                print(f"[INFO] Successfully loaded: {artifact.name}")
                        except Exception as e:
                            print(f"[WARNING] Failed to load artifact {artifact.name}: {e}")
                except Exception as e:
                    # 某个项目可能没有指定类型的 artifacts
                    continue
                    
        except Exception as e:
            print(f"[ERROR] Failed alternative loading approach: {e}")
    
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
