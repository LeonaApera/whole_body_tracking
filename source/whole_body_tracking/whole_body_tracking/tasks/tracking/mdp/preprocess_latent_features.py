#!/usr/bin/env python3
"""
Preprocess LAFAN1 motion data to extract VQ-VAE latent features.

This script:
1. Downloads motion data from wandb using registry names
2. Applies sliding windows of 100 future frames on the motion sequence 
3. Passes each window through the VQ-VAE encoder and quantizer
4. Extracts x_quantized[:, :, 0] tensor for each timestep
5. Saves the processed sequence as a list of latent spaces for reuse

The preprocessing is consistent with the latent_space_67 function in observations.py.
"""

import os
import sys
import argparse
import pickle
import pathlib
import tempfile
from typing import List, Dict, Any, Optional, Tuple
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import wandb


def add_vqvae_path():
    """Add VQ-VAE module path to sys.path."""
    vqvae_path = '/workspace/projects/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/vqvae'
    if vqvae_path not in sys.path:
        sys.path.append(vqvae_path)


class MotionDataLoader:
    """Load motion data from wandb registry or local files."""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        
    def load_from_wandb(self, registry_name: str, download_dir: Optional[str] = None) -> List[str]:
        """
        Download motion data from wandb registry and return list of npz file paths.
        
        Args:
            registry_name: wandb registry name (e.g., 'jianuocao0105-nanjing-university-org/wandb-registry-motions/dance1_subject1')
            download_dir: Directory to download files to
            
        Returns:
            List of paths to downloaded .npz files
        """
        api = wandb.Api()
        
        if download_dir is None:
            download_dir = tempfile.mkdtemp()
        
        npz_files = []
        
        try:
            print(f"Downloading from wandb registry: {registry_name}")
            
            # Parse registry name to extract parts
            registry_parts = registry_name.split('/')
            if len(registry_parts) >= 3:
                entity = registry_parts[0]
                project = registry_parts[1] 
                artifact_name = '/'.join(registry_parts[2:])
            else:
                raise ValueError(f"Invalid registry name format: {registry_name}")
                
            # Handle different artifact name formats
            if ':' not in artifact_name:
                artifact_name += ':latest'
                
            full_artifact_name = f"{entity}/{project}/{artifact_name}"
            print(f"Downloading artifact: {full_artifact_name}")
            
            # Download artifact
            artifact = api.artifact(full_artifact_name)
            artifact_dir = artifact.download(download_dir)
            
            # Find all .npz files in the downloaded directory
            npz_files = list(pathlib.Path(artifact_dir).rglob("*.npz"))
            print(f"Found {len(npz_files)} .npz files in downloaded artifact")
            
            return [str(f) for f in npz_files]
            
        except Exception as e:
            warnings.warn(f"Failed to download {registry_name}: {str(e)}")
            return []
    
    def load_motion_from_npz(self, npz_file: str) -> Dict[str, torch.Tensor]:
        """
        Load motion data from npz file and convert to format consistent with observations.py.
        
        Args:
            npz_file: Path to .npz file
            
        Returns:
            Dictionary containing motion data tensors
        """
        assert os.path.isfile(npz_file), f"Invalid file path: {npz_file}"
        
        data = np.load(npz_file)
        
        # Load basic info
        fps = data["fps"].item() if "fps" in data else 50
        
        # Load joint data (29 joints)
        joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=self.device)
        joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=self.device)
        
        # Load body data (anchor body info)
        body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=self.device)
        body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=self.device)
        
        total_frames = joint_pos.shape[0]
        
        # Extract anchor body (first body is anchor, typically pelvis/root)
        anchor_pos_w = body_pos_w[:, 0, :]  # (T, 3)
        anchor_quat_w = body_quat_w[:, 0, :]  # (T, 4)
        
        # Convert quaternion to rotation matrix and extract first 2 columns (6D representation)
        anchor_rot_mat = self._quat_to_rotation_matrix(anchor_quat_w)  # (T, 3, 3)
        anchor_rot_6d = anchor_rot_mat[:, :, :2].reshape(total_frames, 6)  # (T, 6)
        
        print(f"Loaded motion: {os.path.basename(npz_file)}")
        print(f"  Total frames: {total_frames}")
        print(f"  FPS: {fps}")
        print(f"  Duration: {total_frames / fps:.2f}s")
        print(f"  Joint pos shape: {joint_pos.shape}")
        print(f"  Joint vel shape: {joint_vel.shape}")
        print(f"  Anchor pos shape: {anchor_pos_w.shape}")
        print(f"  Anchor rot 6D shape: {anchor_rot_6d.shape}")
        
        return {
            'joint_pos': joint_pos,
            'joint_vel': joint_vel,
            'anchor_pos_w': anchor_pos_w,
            'anchor_rot_6d': anchor_rot_6d,
            'total_frames': total_frames,
            'fps': fps,
            'filename': os.path.basename(npz_file)
        }
    
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


class VQVAELatentProcessor:
    """Process motion data through VQ-VAE to extract latent features."""
    
    def __init__(
        self,
        model_path: str,
        window_size: int = 100,
        dim: int = 32,
        device: str = "cpu"
    ):
        """
        Initialize VQ-VAE processor.
        
        Args:
            model_path: Path to pre-trained VQ-VAE model
            window_size: Size of sliding window (default: 100)
            dim: Latent dimension (default: 32)
            device: Device to run inference on
        """
        self.model_path = model_path
        self.window_size = window_size
        self.dim = dim
        self.device = device
        
        # Add VQ-VAE module to path
        add_vqvae_path()
        
        # Load VQ-VAE model
        self._load_model()
        
    def _load_model(self):
        """Load the pre-trained VQ-VAE model."""
        # Import VQ-VAE modules
        try:
            from vqvae_dim_experiment import VQVae
        except ImportError as e:
            raise ImportError(f"Failed to import VQVae: {e}. Make sure the vqvae module is in the correct path.")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Create model instance with correct parameters for 67-dimensional features
        self.model = VQVae(
            nfeats=67,  # 29 joint_pos + 29 joint_vel + 3 anchor_pos + 6 anchor_rot
            quantizer='ema_reset',
            code_num=512,
            code_dim=self.dim,
            output_emb_width=self.dim,
            down_t=2,
            stride_t=2,
            width=512,
            depth=3,
            dilation_growth_rate=3,
            norm=None,
            activation='relu'
        )
        
        # Load pre-trained weights
        checkpoint = torch.load(self.model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"[INFO] Loaded VQ-VAE model from {self.model_path}")
        print(f"  Window size: {self.window_size}")
        print(f"  Latent dimension: {self.dim}")
        print(f"  Device: {self.device}")
    
    def create_67d_feature_vector(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        anchor_pos_w: torch.Tensor,
        anchor_rot_6d: torch.Tensor,
        frame_idx: int
    ) -> torch.Tensor:
        """
        Create 67-dimensional feature vector for a single frame.
        
        Args:
            joint_pos: Joint positions (T, 29)
            joint_vel: Joint velocities (T, 29)
            anchor_pos_w: Anchor position (T, 3)
            anchor_rot_6d: Anchor rotation in 6D representation (T, 6)
            frame_idx: Frame index
            
        Returns:
            67-dimensional feature vector
        """
        return torch.cat([
            joint_pos[frame_idx],      # 29 dims
            joint_vel[frame_idx],      # 29 dims  
            anchor_pos_w[frame_idx],   # 3 dims
            anchor_rot_6d[frame_idx]   # 6 dims
        ], dim=0)  # Total: 67 dims
    
    def process_motion_sequence(
        self,
        motion_data: Dict[str, torch.Tensor],
        normalize_stats: Optional[Dict[str, torch.Tensor]] = None
    ) -> List[torch.Tensor]:
        """
        Process a complete motion sequence to extract latent features.
        
        Args:
            motion_data: Motion data dictionary from load_motion_from_npz
            normalize_stats: Optional normalization statistics
            
        Returns:
            List of latent tensors, one for each timestep where a full window is available
        """
        joint_pos = motion_data['joint_pos']
        joint_vel = motion_data['joint_vel']
        anchor_pos_w = motion_data['anchor_pos_w']
        anchor_rot_6d = motion_data['anchor_rot_6d']
        total_frames = motion_data['total_frames']
        
        latent_features_list = []
        
        # Process each timestep with a future window
        max_start_frame = total_frames - self.window_size
        
        if max_start_frame < 0:
            warnings.warn(f"Motion sequence too short ({total_frames} frames) for window size {self.window_size}")
            return latent_features_list
        
        print(f"Processing {max_start_frame + 1} timesteps with {self.window_size}-frame windows...")
        
        for current_timestep in range(max_start_frame + 1):
            # Create window starting from current_timestep
            window_features = []
            
            for frame_offset in range(self.window_size):
                frame_idx = current_timestep + frame_offset
                feature_vec = self.create_67d_feature_vector(
                    joint_pos, joint_vel, anchor_pos_w, anchor_rot_6d, frame_idx
                )
                window_features.append(feature_vec)
            
            # Stack into window: (window_size, 67)
            motion_window = torch.stack(window_features, dim=0)
            
            # Apply normalization if provided
            if normalize_stats is not None:
                mean = normalize_stats['mean'].to(self.device)
                std = normalize_stats['std'].to(self.device)
                motion_window = (motion_window - mean) / std
            
            # Add batch dimension: (1, window_size, 67)
            motion_window = motion_window.unsqueeze(0)
            
            # Pass through VQ-VAE encoder and quantizer
            with torch.no_grad():
                # Encode using VQ-VAE
                x_in = motion_window.permute(0, 2, 1)  # (1, 67, window_size)
                x_encoder = self.model.encoder(x_in)   # Get encoder output
                
                # Quantize
                x_quantized, _, _ = self.model.quantizer(x_encoder)
                
                # Extract x_quantized[:, :, 0] as specified
                latent_features = x_quantized[:, :, 0]  # (1, latent_dim)
                
                # Remove batch dimension and store
                latent_features_list.append(latent_features.squeeze(0))  # (latent_dim,)
        
        print(f"Extracted latent features for {len(latent_features_list)} timesteps")
        return latent_features_list


def main():
    """Main preprocessing function."""
    parser = argparse.ArgumentParser(description="Preprocess LAFAN1 motion data to extract VQ-VAE latent features")
    
    parser.add_argument(
        "--registry_name",
        type=str,
        required=True,
        help="wandb registry name (e.g., jianuocao0105-nanjing-university-org/wandb-registry-motions/dance1_subject1)"
    )
    parser.add_argument(
        "--model_path", 
        type=str,
        default="/workspace/projects/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/vqvae/best_model_128.pt",
        help="Path to pre-trained VQ-VAE model"
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=100,
        help="Sliding window size (default: 100)"
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=32,
        help="Latent dimension (default: 32)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./preprocessed_latents",
        help="Output directory to save preprocessed features"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run on (cpu/cuda)"
    )
    parser.add_argument(
        "--download_dir",
        type=str,
        default=None,
        help="Directory to download wandb artifacts to"
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=False,
        help="Apply normalization (requires --normalize_stats_file)"
    )
    parser.add_argument(
        "--normalize_stats_file",
        type=str,
        default=None,
        help="Path to normalization statistics file (generated by compute_lafan1_stats.py)"
    )
    
    args = parser.parse_args()
    
    # Setup
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize data loader
    data_loader = MotionDataLoader(device=device)
    
    # Load normalization stats if required
    normalize_stats = None
    if args.normalize:
        if args.normalize_stats_file is None:
            raise ValueError("--normalize_stats_file must be provided when --normalize is used")
        if not os.path.exists(args.normalize_stats_file):
            raise FileNotFoundError(f"Normalization stats file not found: {args.normalize_stats_file}")
        
        normalize_stats = torch.load(args.normalize_stats_file, map_location=device)
        print(f"Loaded normalization stats from {args.normalize_stats_file}")
    
    # Initialize VQ-VAE processor
    processor = VQVAELatentProcessor(
        model_path=args.model_path,
        window_size=args.window_size,
        dim=args.dim,
        device=device
    )
    
    # Download and process motion data
    print(f"Downloading motion data from: {args.registry_name}")
    npz_files = data_loader.load_from_wandb(args.registry_name, args.download_dir)
    
    if not npz_files:
        print("No motion files found!")
        return
    
    # Process each motion file
    for npz_file in npz_files:
        print(f"\nProcessing: {npz_file}")
        
        # Load motion data
        motion_data = data_loader.load_motion_from_npz(npz_file)
        
        # Extract latent features
        latent_features_list = processor.process_motion_sequence(motion_data, normalize_stats)
        
        if not latent_features_list:
            print(f"No latent features extracted for {npz_file}")
            continue
        
        # Prepare output data
        output_data = {
            'latent_features': latent_features_list,  # List of tensors, each of shape (latent_dim,)
            'metadata': {
                'filename': motion_data['filename'],
                'total_frames': motion_data['total_frames'],
                'fps': motion_data['fps'],
                'window_size': args.window_size,
                'latent_dim': args.dim,
                'num_timesteps': len(latent_features_list),
                'model_path': args.model_path,
                'registry_name': args.registry_name,
                'normalized': args.normalize
            }
        }
        
        # Save preprocessed features
        # Extract the last part of registry_name after the last '/'
        registry_suffix = args.registry_name.split('/')[-1] if '/' in args.registry_name else args.registry_name
        # Remove version tag if present (e.g., :latest, :v0)
        if ':' in registry_suffix:
            registry_suffix = registry_suffix.split(':')[0]
            
        output_filename = f"{registry_suffix}_{os.path.splitext(motion_data['filename'])[0]}_latents_w{args.window_size}_d{args.dim}.pkl"
        output_path = os.path.join(args.output_dir, output_filename)
        
        with open(output_path, 'wb') as f:
            pickle.dump(output_data, f)
        
        print(f"Saved preprocessed latents to: {output_path}")
        print(f"  Timesteps processed: {len(latent_features_list)}")
        print(f"  Latent dimension: {latent_features_list[0].shape[0] if latent_features_list else 0}")
    
    print("\n✓ Preprocessing completed!")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()