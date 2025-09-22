"""
LAFAN1 Sliding Window Dataset for VQ-VAE Training

This dataset loads LAFAN1 motion data from wandb artifacts and creates sliding windows
with the format (batch_size, window_size=100, 67) where:
- 29 dimensions: joint positions/angles
- 29 dimensions: joint velocities  
- 3 dimensions: anchor global position (x, y, z)
- 6 dimensions: anchor global rotation (first 2 columns of rotation matrix)

"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import wandb
import pathlib
import tempfile
import os
from typing import List, Optional, Tuple, Dict, Any, Union
import warnings


class LAFAN1MotionData:
    """Single motion sequence data loader from npz file."""
    
    def __init__(self, motion_file: str, device: str = "cpu"):
        """
        Load motion data from npz file.
        
        Args:
            motion_file: Path to .npz file containing motion data
            device: Device to load tensors to (recommend "cpu" for memory efficiency)
        """
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        
        data = np.load(motion_file)
        self.fps = data["fps"].item() if "fps" in data else 50  # Default fps
        
        # Load joint data
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        
        # Load body data (anchor body info)
        self.body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self.body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self.body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self.body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        
        self.total_frames = self.joint_pos.shape[0]
        
        # Extract anchor body (assume first body is anchor, typically pelvis/root)
        self.anchor_pos_w = self.body_pos_w[:, 0, :]  # (T, 3)
        self.anchor_quat_w = self.body_quat_w[:, 0, :]  # (T, 4) - quaternion
        
        # Convert quaternion to rotation matrix and extract first 2 columns
        self.anchor_rot_mat = self._quat_to_rotation_matrix(self.anchor_quat_w)  # (T, 3, 3)
        self.anchor_rot_6d = self.anchor_rot_mat[:, :, :2].reshape(self.total_frames, 6)  # (T, 6)
        
        print(f"Loaded motion: {motion_file}")
        print(f"  Total frames: {self.total_frames}")
        print(f"  FPS: {self.fps}")
        print(f"  Duration: {self.total_frames / self.fps:.2f}s")
        print(f"  Joint pos shape: {self.joint_pos.shape}")
        print(f"  Joint vel shape: {self.joint_vel.shape}")
        print(f"  Anchor pos shape: {self.anchor_pos_w.shape}")
        print(f"  Anchor rot 6D shape: {self.anchor_rot_6d.shape}")
    
    def _quat_to_rotation_matrix(self, quat: torch.Tensor) -> torch.Tensor:
        """
        Convert quaternion to rotation matrix.
        
        Args:
            quat: (T, 4) quaternion in (w, x, y, z) format
            
        Returns:
            rotation matrix of shape (T, 3, 3)
        """
        # Normalize quaternion
        quat = quat / torch.norm(quat, dim=-1, keepdim=True)
        
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        
        # Compute rotation matrix elements
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z
        
        # Build rotation matrix
        R = torch.zeros(quat.shape[0], 3, 3, device=quat.device, dtype=quat.dtype)
        
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
    
    def get_feature_vector(self, frame_idx: int) -> torch.Tensor:
        """
        Get the 67-dimensional feature vector for a single frame.
        
        Args:
            frame_idx: Frame index
            
        Returns:
            Feature vector of shape (67,) containing:
            - joint_pos (29)
            - joint_vel (29) 
            - anchor_pos_w (3)
            - anchor_rot_6d (6)
        """
        return torch.cat([
            self.joint_pos[frame_idx],      # 29 dims
            self.joint_vel[frame_idx],      # 29 dims  
            self.anchor_pos_w[frame_idx],   # 3 dims
            self.anchor_rot_6d[frame_idx]   # 6 dims
        ], dim=0)  # Total: 67 dims


class LAFAN1SlidingWindowDataset(Dataset):
    """
    LAFAN1 Dataset with sliding windows for VQ-VAE training.
    
    Creates sliding windows of motion data from multiple motion sequences.
    Each sample is a window of size 100 with 67-dimensional features per frame.
    """
    
    def __init__(
        self,
        motion_files: Optional[List[str]] = None,
        wandb_registry_names: Optional[List[str]] = None,
        window_size: int = 100,
        stride: int = 1,
        min_sequence_length: int = 100,
        device: str = "cpu",
        normalize: bool = True,
        normalization_stats_file: Optional[str] = None,
        download_dir: Optional[str] = None
    ):
        """
        Initialize LAFAN1 sliding window dataset.
        
        Args:
            motion_files: List of local npz files or directories to load. 
                         If a directory is provided, will recursively search for all .npz files.
            wandb_registry_names: List of wandb artifact names to download
            window_size: Size of sliding window (default 100)
            stride: Stride for sliding window (default 1)
            min_sequence_length: Minimum sequence length to include
            device: Device to load data to (recommend "cpu")
            normalize: Whether to normalize the data
            normalization_stats_file: Path to pre-computed normalization stats file
            download_dir: Directory to download wandb artifacts to
        """
        self.window_size = window_size
        self.stride = stride
        self.min_sequence_length = min_sequence_length
        self.device = device
        self.normalize = normalize
        
        # Data storage
        self.motion_data_list: List[LAFAN1MotionData] = []
        self.windows_info: List[Tuple[int, int]] = []  # (motion_idx, start_frame)
        
        # Load from local files
        if motion_files:
            self._load_from_files(motion_files)
            
        # Load from wandb
        if wandb_registry_names and any(name is not None for name in wandb_registry_names):
            print("---------------------------------------------------")
            print("Wandb registry names:", wandb_registry_names)
            self._load_from_wandb(wandb_registry_names, download_dir)
            
        # Create sliding windows
        self._create_sliding_windows()
        
        # Load normalization statistics (required if normalize=True)
        if self.normalize:
            if normalization_stats_file is None:
                raise ValueError(
                    "normalization_stats_file must be provided when normalize=True. "
                    "Use compute_lafan1_stats.py to generate the stats file first."
                )
            self.load_normalization_stats(normalization_stats_file)
            print(f"Normalization: enabled (loaded from {normalization_stats_file})")
        else:
            print(f"Normalization: disabled")
        print(f"Dataset created with {len(self.windows_info)} windows")
        print(f"Window size: {self.window_size}, Stride: {self.stride}")

        
    def _load_from_files(self, motion_files: List[str]):
        """Load motion data from local npz files or directories (recursively search for .npz files)."""
        npz_files = []
        
        # Collect all .npz files from the provided paths
        for path in motion_files:
            if os.path.isfile(path) and path.endswith('.npz'):
                # Direct .npz file
                npz_files.append(path)
            elif os.path.isdir(path):
                # Directory - recursively find all .npz files
                for root, dirs, files in os.walk(path):
                    for file in files:
                        if file.endswith('.npz'):
                            npz_files.append(os.path.join(root, file))
                print(f"Found {len(npz_files)} .npz files in directory: {path}")
            else:
                warnings.warn(f"Path not found or not a .npz file: {path}")
        
        print(f"Total .npz files to load: {len(npz_files)}")
        
        # Load each .npz file
        for motion_file in npz_files:
            try:
                motion_data = LAFAN1MotionData(motion_file, self.device)
                if motion_data.total_frames >= self.min_sequence_length:
                    self.motion_data_list.append(motion_data)
                    print(f"  ✓ Loaded: {os.path.basename(motion_file)} ({motion_data.total_frames} frames)")
                else:
                    warnings.warn(f"  ✗ Skipping {os.path.basename(motion_file)}: too short ({motion_data.total_frames} < {self.min_sequence_length})")
            except Exception as e:
                warnings.warn(f"  ✗ Failed to load {os.path.basename(motion_file)}: {str(e)}")
                
    def _load_from_wandb(self, wandb_registry_names: List[str], download_dir: Optional[str]):
        """Load motion data from wandb artifacts."""
        api = wandb.Api()
        
        if download_dir is None:
            download_dir = tempfile.mkdtemp()
            
        for registry_name in wandb_registry_names:
            try:
                registry_filters = {
                    "name": {"$regex": registry_name}  # 使用正则匹配 Registry 名称（精确匹配可直接用名称）
            }
                collections = api.registries(filter=registry_filters).collections(filter={})  
                for collection in collections:    
                    print(f"Processing Collection: {collection.name}")
                    collection_filters = {
                        "name": {"$regex": collection.name}
                    }
                    versions = api.registries().collections(filter=collection_filters).versions(filter={})
                    for version in versions:
                        print(f"Version Name: {version.name}")
                    artifact = api.artifact(name=f"{version.collection.entity}/{version.collection.project}/{version.name}")  # 使用别名或版本
                    download_dir_specific = f"{download_dir}/{version.name}"
                    artifact_dir = artifact.download(download_dir_specific)
                # Find all .npz files in the artifact directory (supports multiple collections)
                npz_files = list(pathlib.Path(download_dir).rglob("*.npz"))
                
                print(f"Found {len(npz_files)} motion files in {registry_name}")
                
                # Load each .npz file as a separate motion sequence
                for npz_file in npz_files:
                    try:
                        motion_data = LAFAN1MotionData(str(npz_file), self.device)
                        if motion_data.total_frames >= self.min_sequence_length:
                            self.motion_data_list.append(motion_data)
                            print(f"  ✓ Loaded: {npz_file.name} ({motion_data.total_frames} frames)")
                        else:
                            warnings.warn(f"  ✗ Skipping {npz_file.name}: too short ({motion_data.total_frames} < {self.min_sequence_length})")
                    except Exception as e:
                        warnings.warn(f"  ✗ Failed to load {npz_file.name}: {str(e)}")
                        
            except Exception as e:
                warnings.warn(f"Failed to download {registry_name}: {str(e)}")
        
    def _create_sliding_windows(self):
        """Create sliding windows from all motion sequences."""
        self.windows_info = []
        
        for motion_idx, motion_data in enumerate(self.motion_data_list):
            # Calculate number of windows for this motion
            num_windows = (motion_data.total_frames - self.window_size) // self.stride + 1
            
            for i in range(num_windows):
                start_frame = i * self.stride
                self.windows_info.append((motion_idx, start_frame))
        
    def __len__(self) -> int:
        """Return number of windows in dataset."""
        return len(self.windows_info)
        
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a sliding window sample.
        
        Args:
            idx: Window index
            
        Returns:
            Window of shape (window_size, 67)
        """
        motion_idx, start_frame = self.windows_info[idx]
        motion_data = self.motion_data_list[motion_idx]
        
        # Extract window
        window_features = []
        for frame_offset in range(self.window_size):
            frame_idx = start_frame + frame_offset
            feature_vec = motion_data.get_feature_vector(frame_idx)
            window_features.append(feature_vec)
            
        window = torch.stack(window_features, dim=0)  # (window_size, 67)
        
        # Normalize if required
        if self.normalize and hasattr(self, 'mean'):
            window = (window - self.mean) / self.std
            
        return window
        
    def get_feature_names(self) -> List[str]:
        """Get names of all features in the 67-dimensional vector."""
        names = []
        
        # Joint positions (29)
        for i in range(29):
            names.append(f"joint_pos_{i}")
            
        # Joint velocities (29)  
        for i in range(29):
            names.append(f"joint_vel_{i}")
            
        # Anchor global position (3)
        names.extend(["anchor_pos_x", "anchor_pos_y", "anchor_pos_z"])
        
        # Anchor global rotation 6D (6)
        for i in range(6):
            names.append(f"anchor_rot_6d_{i}")
            
        return names
        
    def save_normalization_stats(self, file_path: str):
        """Save normalization statistics to file."""
        if hasattr(self, 'mean') and hasattr(self, 'std'):
            torch.save({
                'mean': self.mean.cpu(),
                'std': self.std.cpu(),
                'feature_names': self.get_feature_names()
            }, file_path)
            print(f"Saved normalization stats to {file_path}")
        else:
            print("No normalization stats to save")
            
    def load_normalization_stats(self, file_path: str):
        """Load normalization statistics from file."""
        if os.path.exists(file_path):
            stats = torch.load(file_path, map_location=self.device)
            self.mean = stats['mean'].to(self.device)
            self.std = stats['std'].to(self.device)
            print(f"Loaded normalization stats from {file_path}")
        else:
            print(f"Stats file not found: {file_path}")


def create_lafan1_dataloader(
    batch_size: int = 32,
    window_size: int = 60, 
    wandb_registry_names: Optional[List[str]] = None,
    motion_files: Optional[List[str]] = None,
    num_workers: int = 0,  # Set to 0 to avoid shared memory issues
    shuffle: bool = True,
    device: str = "cpu",
    normalization_stats_file: Optional[str] = None,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader for LAFAN1 sliding window dataset.
    
    Args:
        batch_size: Batch size
        window_size: Sliding window size
        wandb_registry_names: List of wandb artifact names
        motion_files: List of local npz files
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        device: Device to load data to (recommend "cpu")
        normalization_stats_file: Path to normalization stats file (required if normalize=True)
        **kwargs: Additional arguments for dataset
        
    Returns:
        DataLoader instance
    """
    dataset = LAFAN1SlidingWindowDataset(
        motion_files=motion_files,
        wandb_registry_names=wandb_registry_names,
        window_size=window_size,
        device=device,
        normalization_stats_file=normalization_stats_file,
        download_dir="/home/yuxin/Projects/VQVAE/VAE/lanfan1_npz2",
        **kwargs
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True
    )
    
    return dataloader


# Example usage and testing
if __name__ == "__main__":
    # Example: Load some LAFAN1 data from wandb
    registry_names = [
        "dance1_subject1:latest",
        "dance1_subject2:latest", 
        "walk1_subject1:latest"
    ]
    
    print("Creating LAFAN1 sliding window dataset...")
    print("Note: You need to run compute_lafan1_stats.py first to generate normalization stats!")
    
    # Try to use pre-computed stats (recommended workflow)
    stats_file = "lafan1_stats.pt"
    if os.path.exists(stats_file):
        print(f"Using pre-computed stats: {stats_file}")
        
        # Create dataset with normalization
        dataset = LAFAN1SlidingWindowDataset(
            wandb_registry_names=registry_names,
            window_size=100,
            stride=1,
            normalize=True,
            normalization_stats_file=stats_file,
            device="cpu"
        )
    else:
        print(f"Stats file {stats_file} not found. Creating dataset without normalization.")
        print("To enable normalization, run:")
        print(f"  python compute_lafan1_stats.py --wandb-artifacts {' '.join(registry_names)} -o {stats_file}")
        
        # Create dataset without normalization
        dataset = LAFAN1SlidingWindowDataset(
            wandb_registry_names=registry_names,
            window_size=60,
            stride=1,
            normalize=False,
            device="cpu"
        )
    
    # Test dataset
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample shape: {sample.shape}")  # Should be (60, 67)

        # Create dataloader
        dataloader = create_lafan1_dataloader(
            batch_size=8,
            window_size=60,
            wandb_registry_names=registry_names,
            normalization_stats_file=stats_file if os.path.exists(stats_file) else None,
            normalize=os.path.exists(stats_file),
            shuffle=True
        )
        
        # Test dataloader
        for batch in dataloader:
            print(f"Batch shape: {batch.shape}")  # Should be (8, 60, 67)
            break
            
    print("Dataset test completed!")