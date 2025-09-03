# VQ-VAE with Sliding Window and Conv1D Encoder/Decoder (Similar to Traditional VQ-VAE)
# This implementation uses traditional conv1d downsampling instead of attention pooling
# and maintains separate upper/lower body encoding

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import gc
from datetime import datetime
from torch.nn.utils.rnn import pad_sequence
import joblib
import wandb


# Utility modules (from traditional VQ-VAE)
class Resnet1D(nn.Module):
    def __init__(self, n_in, n_depth, dilation_growth_rate=1, reverse_dilation=False, activation='relu', norm=None):
        super().__init__()
        
        blocks = []
        for i in range(n_depth):
            dilation = dilation_growth_rate ** i
            if reverse_dilation:
                dilation = dilation_growth_rate ** (n_depth - 1 - i)
            
            block = ResnetBlock1D(
                n_in, n_in, dilation=dilation, activation=activation, norm=norm
            )
            blocks.append(block)
        
        self.model = nn.Sequential(*blocks)
    
    def forward(self, x):
        return self.model(x)


class ResnetBlock1D(nn.Module):
    def __init__(self, n_in, n_out, dilation=1, activation='relu', norm=None):
        super().__init__()
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Normalization
        if norm == 'batch':
            self.norm1 = nn.BatchNorm1d(n_in)
            self.norm2 = nn.BatchNorm1d(n_out)
        elif norm == 'group':
            self.norm1 = nn.GroupNorm(8, n_in)
            self.norm2 = nn.GroupNorm(8, n_out)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        
        # Convolution layers
        self.conv1 = nn.Conv1d(n_in, n_out, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(n_out, n_out, kernel_size=1)
        
        # Skip connection
        if n_in != n_out:
            self.skip = nn.Conv1d(n_in, n_out, kernel_size=1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x):
        # First convolution block
        h = self.norm1(x)
        h = self.activation(h)
        h = self.conv1(h)
        
        # Second convolution block
        h = self.norm2(h)
        h = self.activation(h)
        h = self.conv2(h)
        
        # Skip connection
        return h + self.skip(x)


# Quantization modules (from traditional VQ-VAE)
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
    
    def forward(self, inputs):
        # Convert inputs from (B, C, T) -> (B, T, C)
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # Convert back to (B, C, T)
        return quantized.permute(0, 2, 1).contiguous(), loss, perplexity
    
    def quantize(self, inputs):
        # Convert inputs from (B, C, T) -> (B, T, C)
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1)
        return encoding_indices.view(input_shape[0], input_shape[1])
    
    def dequantize(self, indices):
        # indices: (B, T)
        quantized = self.embedding(indices)  # (B, T, C)
        return quantized.permute(0, 2, 1).contiguous()  # (B, C, T)


class QuantizeEMAReset(VectorQuantizer):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super().__init__(num_embeddings, embedding_dim, commitment_cost)
        
        self.decay = decay
        self.epsilon = epsilon
        
        # Use optimized data types for memory efficiency
        self.register_buffer('cluster_size', torch.zeros(num_embeddings, dtype=torch.float32))
        self.register_buffer('embed_avg', self.embedding.weight.data.clone())
        
        # Add reset counter for periodic reset to prevent memory leak
        self.register_buffer('update_count', torch.tensor(0, dtype=torch.long))
        self.reset_interval = 1000  # Reset every 1000 updates
    
    def forward(self, inputs):
        # Convert inputs from (B, C, T) -> (B, T, C)
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)
        
        # Use EMA to update the embedding vectors
        if self.training:
            # Periodic reset to prevent memory leak
            self.update_count += 1
            if self.update_count % self.reset_interval == 0:
                self.cluster_size.fill_(1.0)
                self.embed_avg.copy_(self.embedding.weight.data)
            
            # Use detach() to avoid gradient accumulation
            encodings_detached = encodings.detach()
            flat_input_detached = flat_input.detach()
            
            self.cluster_size = self.cluster_size * self.decay + \
                               (1 - self.decay) * torch.sum(encodings_detached, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self.cluster_size.data)
            self.cluster_size = (
                (self.cluster_size + self.epsilon)
                / (n + self.num_embeddings * self.epsilon) * n)
            
            dw = torch.matmul(encodings_detached.t(), flat_input_detached)
            self.embed_avg = self.embed_avg * self.decay + (1 - self.decay) * dw
            
            self.embedding.weight = nn.Parameter(self.embed_avg / self.cluster_size.unsqueeze(1))
        
        # Loss (only commitment loss in traditional style)
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self.commitment_cost * e_latent_loss
        
        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # Convert back to (B, C, T)
        return quantized.permute(0, 2, 1).contiguous(), loss, perplexity


# Traditional VQ-VAE style Encoder/Decoder
class Conv1dEncoder(nn.Module):
    """
    Traditional VQ-VAE style 1D Convolutional Encoder with downsampling
    Input:  (batch, input_dim, seq_len)
    Output: (batch, output_dim, reduced_seq_len)
    """
    def __init__(self,
                 input_emb_width=30,  # input feature dimension per timestep
                 output_emb_width=128,  # output latent dimension
                 down_t=2,  # number of downsampling layers
                 stride_t=2,  # stride for downsampling
                 width=128,  # hidden dimension
                 depth=3,  # ResNet block depth
                 dilation_growth_rate=3,  # dilation rate (changed to 3 as requested)
                 activation='relu',
                 norm=None):
        super().__init__()

        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        
        # Initial conv layer
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        # Downsampling layers with ResNet blocks
        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width,
                         depth,
                         dilation_growth_rate,
                         activation=activation,
                         norm=norm),
            )
            blocks.append(block)
        
        # Final output layer
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class Conv1dDecoder(nn.Module):
    """
    Traditional VQ-VAE style 1D Convolutional Decoder with upsampling
    Input:  (batch, input_dim, reduced_seq_len)
    Output: (batch, output_dim, seq_len)
    """
    def __init__(self,
                 input_emb_width=30,  # final output feature dimension
                 output_emb_width=128,  # input latent dimension
                 down_t=2,  # number of upsampling layers (should match encoder's down_t)
                 stride_t=2,  # stride for upsampling
                 width=128,  # hidden dimension
                 depth=3,  # ResNet block depth
                 dilation_growth_rate=2,  # dilation rate (changed to 2 as requested)
                 activation='relu',
                 norm=None):
        super().__init__()
        blocks = []

        filter_t, pad_t = stride_t * 2, stride_t // 2
        
        # Initial layer
        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        
        # Upsampling layers with ResNet blocks
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width,
                         depth,
                         dilation_growth_rate,
                         reverse_dilation=True,
                         activation=activation,
                         norm=norm), 
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1))
            blocks.append(block)
        
        # Final output layers
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class SlidingWindowVQVAEConv(nn.Module):
    """
    VQ-VAE with sliding window input and separate upper/lower body encoding
    Uses traditional VQ-VAE conv1d architecture instead of attention pooling
    
    Input: 60 frames (2 seconds at 30fps) of motion data
    Each body part has its own conv1d encoder/decoder and codebook
    """
    def __init__(self, 
                 window_size=60,  # 2 seconds at 30fps
                 n_joints=29,     # G1 has 29 joints
                 joint_dim=2,     # G1 joints: angle + velocity
                 code_num=512,    # Size of each codebook
                 code_dim=128,    # Dimension of each code vector
                 commitment_cost=0.25,
                 width=128,       # Hidden dimension for conv layers
                 depth=3,         # ResNet block depth
                 down_t=2,        # Number of downsampling layers
                 stride_t=2,      # Stride for downsampling
                 dilation_growth_rate=2):  # Dilation rate (as requested)
        super().__init__()
        
        self.window_size = window_size
        self.n_joints = n_joints
        self.joint_dim = joint_dim
        self.code_num = code_num
        self.code_dim = code_dim
        
        # Joint indices for lower and upper body
        self.lower_body_joints = list(range(15))    # joints 0-14: lower body
        self.upper_body_joints = list(range(15, 29)) # joints 15-28: upper body
        
        # Calculate input dimensions
        self.lower_body_dim = len(self.lower_body_joints) * joint_dim  # 15 * 2 = 30
        self.upper_body_dim = len(self.upper_body_joints) * joint_dim  # 14 * 2 = 28
        
        # Encoders for each body part (traditional VQ-VAE style)
        self.lower_encoder = Conv1dEncoder(
            input_emb_width=self.lower_body_dim,
            output_emb_width=code_dim,
            down_t=down_t,
            stride_t=stride_t,
            width=width,
            depth=depth,
            dilation_growth_rate=dilation_growth_rate,
            activation='relu',
            norm=None
        )
        self.upper_encoder = Conv1dEncoder(
            input_emb_width=self.upper_body_dim,
            output_emb_width=code_dim,
            down_t=down_t,
            stride_t=stride_t,
            width=width,
            depth=depth,
            dilation_growth_rate=dilation_growth_rate,
            activation='relu',
            norm=None
        )
        
        # Quantizers for each body part (using traditional VQ-VAE style)
        self.lower_quantizer = QuantizeEMAReset(
            num_embeddings=code_num,
            embedding_dim=code_dim,
            commitment_cost=commitment_cost,
            decay=0.99,
            epsilon=1e-5
        )
        self.upper_quantizer = QuantizeEMAReset(
            num_embeddings=code_num,
            embedding_dim=code_dim,
            commitment_cost=commitment_cost,
            decay=0.99,
            epsilon=1e-5
        )
        
        # Decoders for each body part (traditional VQ-VAE style)
        self.lower_decoder = Conv1dDecoder(
            input_emb_width=self.lower_body_dim,
            output_emb_width=code_dim,
            down_t=down_t,
            stride_t=stride_t,
            width=width,
            depth=depth,
            dilation_growth_rate=dilation_growth_rate,
            activation='relu',
            norm=None
        )
        self.upper_decoder = Conv1dDecoder(
            input_emb_width=self.upper_body_dim,
            output_emb_width=code_dim,
            down_t=down_t,
            stride_t=stride_t,
            width=width,
            depth=depth,
            dilation_growth_rate=dilation_growth_rate,
            activation='relu',
            norm=None
        )
    
    def forward(self, x):
        """
        Forward pass through the model
        
        Args:
            x: Input tensor of shape (B, window_size, n_joints, joint_dim)
               where B is batch size
        
        Returns:
            x_recon: Reconstructed input of same shape as x
            total_loss: Combined quantization loss
            perplexity: Average perplexity across both quantizers
            codes: Dictionary with 'lower' and 'upper' discrete codes
        """
        batch_size = x.shape[0]
        
        # Split into lower and upper body
        lower_body = x[:, :, self.lower_body_joints, :]  # (B, T, 15, 2)
        upper_body = x[:, :, self.upper_body_joints, :]  # (B, T, 14, 2)
        
        # Reshape for Conv1D input: (B, T, n_joints, joint_dim) -> (B, n_joints*joint_dim, T)
        lower_body_flat = lower_body.reshape(batch_size, self.window_size, -1)  # (B, T, 30)
        upper_body_flat = upper_body.reshape(batch_size, self.window_size, -1)  # (B, T, 28)
        
        # Transpose for Conv1D: (B, T, features) -> (B, features, T)
        lower_body_conv = lower_body_flat.permute(0, 2, 1)  # (B, 30, T)
        upper_body_conv = upper_body_flat.permute(0, 2, 1)  # (B, 28, T)

        # Encode each body part
        lower_encoded = self.lower_encoder(lower_body_conv)   # (B, code_dim, T_reduced)
        upper_encoded = self.upper_encoder(upper_body_conv)   # (B, code_dim, T_reduced)
        
        # Quantize each body part
        lower_quantized, lower_loss, lower_perplexity = self.lower_quantizer(lower_encoded)
        upper_quantized, upper_loss, upper_perplexity = self.upper_quantizer(upper_encoded)
        
        # Get discrete codes
        lower_codes = self.lower_quantizer.quantize(lower_encoded)  # (B, T_reduced)
        upper_codes = self.upper_quantizer.quantize(upper_encoded)  # (B, T_reduced)
        
        # Decode each body part
        lower_decoded = self.lower_decoder(lower_quantized)  # (B, 30, T)
        upper_decoded = self.upper_decoder(upper_quantized)  # (B, 28, T)
        
        # Transpose back: (B, features, T) -> (B, T, features)
        lower_decoded = lower_decoded.permute(0, 2, 1)  # (B, T, 30)
        upper_decoded = upper_decoded.permute(0, 2, 1)  # (B, T, 28)
        
        # Handle sequence length mismatch due to downsampling/upsampling
        target_len = self.window_size
        if lower_decoded.shape[1] != target_len:
            lower_decoded = F.interpolate(
                lower_decoded.permute(0, 2, 1), 
                size=target_len, 
                mode='linear', 
                align_corners=True
            ).permute(0, 2, 1)
        
        if upper_decoded.shape[1] != target_len:
            upper_decoded = F.interpolate(
                upper_decoded.permute(0, 2, 1), 
                size=target_len, 
                mode='linear', 
                align_corners=True
            ).permute(0, 2, 1)
        
        # Reshape back to original format
        lower_recon = lower_decoded.reshape(batch_size, target_len, len(self.lower_body_joints), self.joint_dim)
        upper_recon = upper_decoded.reshape(batch_size, target_len, len(self.upper_body_joints), self.joint_dim)
        
        # Combine lower and upper body reconstructions
        x_recon = torch.zeros_like(x)
        x_recon[:, :, self.lower_body_joints, :] = lower_recon
        x_recon[:, :, self.upper_body_joints, :] = upper_recon
        
        # Combine losses and perplexities
        total_loss = lower_loss + upper_loss
        avg_perplexity = (lower_perplexity + upper_perplexity) / 2
        
        codes = {
            'lower': lower_codes,
            'upper': upper_codes
        }
        
        return x_recon, total_loss, avg_perplexity, codes
    
    def encode(self, x):
        """
        Encode input to discrete codes
        
        Args:
            x: Input tensor of shape (B, window_size, n_joints, joint_dim)
        
        Returns:
            codes: Dictionary with 'lower' and 'upper' discrete codes
            latents: Dictionary with 'lower' and 'upper' continuous latent codes
        """
        batch_size = x.shape[0]
        
        # Split into lower and upper body
        lower_body = x[:, :, self.lower_body_joints, :]
        upper_body = x[:, :, self.upper_body_joints, :]
        
        # Reshape for Conv1D input
        lower_body_flat = lower_body.reshape(batch_size, self.window_size, -1)
        upper_body_flat = upper_body.reshape(batch_size, self.window_size, -1)
        
        # Transpose for Conv1D
        lower_body_conv = lower_body_flat.permute(0, 2, 1)
        upper_body_conv = upper_body_flat.permute(0, 2, 1)
        
        # Encode each body part
        lower_encoded = self.lower_encoder(lower_body_conv)
        upper_encoded = self.upper_encoder(upper_body_conv)
        
        # Get discrete codes
        lower_codes = self.lower_quantizer.quantize(lower_encoded)
        upper_codes = self.upper_quantizer.quantize(upper_encoded)
        
        codes = {
            'lower': lower_codes,
            'upper': upper_codes
        }
        
        latents = {
            'lower': lower_encoded,
            'upper': upper_encoded
        }
        
        return codes, latents
    
    def decode(self, codes):
        """
        Decode discrete codes to motion data
        
        Args:
            codes: Dictionary with 'lower' and 'upper' discrete codes
        
        Returns:
            x_recon: Reconstructed motion data
        """
        lower_codes = codes['lower']
        upper_codes = codes['upper']
        batch_size = lower_codes.shape[0]
        
        # Dequantize
        lower_quantized = self.lower_quantizer.dequantize(lower_codes)
        upper_quantized = self.upper_quantizer.dequantize(upper_codes)
        
        # Decode
        lower_decoded = self.lower_decoder(lower_quantized)
        upper_decoded = self.upper_decoder(upper_quantized)
        
        # Transpose back
        lower_decoded = lower_decoded.permute(0, 2, 1)
        upper_decoded = upper_decoded.permute(0, 2, 1)
        
        # Handle sequence length mismatch
        target_len = self.window_size
        if lower_decoded.shape[1] != target_len:
            lower_decoded = F.interpolate(
                lower_decoded.permute(0, 2, 1), 
                size=target_len, 
                mode='linear', 
                align_corners=True
            ).permute(0, 2, 1)
        
        if upper_decoded.shape[1] != target_len:
            upper_decoded = F.interpolate(
                upper_decoded.permute(0, 2, 1), 
                size=target_len, 
                mode='linear', 
                align_corners=True
            ).permute(0, 2, 1)
        
        # Reshape back to original format
        lower_recon = lower_decoded.reshape(batch_size, target_len, len(self.lower_body_joints), self.joint_dim)
        upper_recon = upper_decoded.reshape(batch_size, target_len, len(self.upper_body_joints), self.joint_dim)
        
        # Combine reconstructions
        x_recon = torch.zeros(batch_size, target_len, self.n_joints, self.joint_dim, device=lower_codes.device)
        x_recon[:, :, self.lower_body_joints, :] = lower_recon
        x_recon[:, :, self.upper_body_joints, :] = upper_recon
        
        return x_recon


# Utility functions for data processing
def create_sliding_windows(trajectory, window_size=60, target_fps=30):
    """
    Create sliding windows from a trajectory
    
    Args:
        trajectory: Motion data of shape (T, n_joints, joint_dim)
        window_size: Size of each window (default 60 for 2 seconds at 30fps)
        target_fps: Target FPS for resampling (default 30)
    
    Returns:
        windows: List of windows, each of shape (window_size, n_joints, joint_dim)
    """
    # Resample to target FPS if needed
    original_fps = 30  # Assuming original data is at 30fps
    if original_fps != target_fps:
        T, n_joints, joint_dim = trajectory.shape
        duration = T / original_fps
        new_T = int(round(duration * target_fps))
        
        # Reshape for interpolation: (T, n_joints*joint_dim) -> (1, n_joints*joint_dim, T)
        traj_flat = trajectory.reshape(T, -1).T.unsqueeze(0)
        traj_resampled = F.interpolate(traj_flat, size=new_T, mode='linear', align_corners=True)
        trajectory = traj_resampled.squeeze(0).T.reshape(new_T, n_joints, joint_dim)
    
    T = trajectory.shape[0]
    windows = []
    
    # Create sliding windows
    for i in range(T - window_size + 1):
        window = trajectory[i:i + window_size]
        windows.append(window)
    
    return windows


def preprocess_trajectories(trajs, target_fps=30):
    """
    Preprocess trajectories for memory-efficient training
    
    Args:
        trajs: List of trajectory dictionaries
        target_fps: Target FPS for resampling
        
    Returns:
        processed_trajs: List of processed trajectory tensors
        window_info: List of tuples (traj_idx, max_windows) for valid windows
    """
    processed_trajs = []
    window_info = []
    
    print("Preprocessing trajectories for memory-efficient training...")
    
    for traj_idx, traj in enumerate(trajs):
        motion_data = torch.tensor(traj['joint_data'], dtype=torch.float32)  # (T, 29, 2)
        
        # Resample to target FPS if needed
        original_fps = traj.get('fps', 30)
        if original_fps != target_fps:
            T, n_joints, joint_dim = motion_data.shape
            duration = T / original_fps
            new_T = int(round(duration * target_fps))
            
            # Reshape for interpolation
            motion_flat = motion_data.reshape(T, -1).T.unsqueeze(0)
            motion_resampled = F.interpolate(motion_flat, size=new_T, mode='linear', align_corners=True)
            motion_data = motion_resampled.squeeze(0).T.reshape(new_T, n_joints, joint_dim)
        
        processed_trajs.append(motion_data)
        
        # Calculate valid window positions
        T = motion_data.shape[0]
        max_windows = max(0, T - 60 + 1)  # 60 is the window size
        if max_windows > 0:
            window_info.append((traj_idx, max_windows))
        
        if (traj_idx + 1) % 100 == 0:
            print(f"Processed {traj_idx + 1}/{len(trajs)} trajectories")
    
    total_windows = sum(info[1] for info in window_info)
    print(f"Preprocessing complete: {len(processed_trajs)} trajectories, {total_windows} possible windows")
    return processed_trajs, window_info


def train_sliding_window_vqvae_conv(trajs, 
                                   window_size=60,
                                   code_num=512,
                                   code_dim=128,
                                   num_epochs=1000,
                                   lr=1e-4,
                                   batch_size=32,
                                   save_interval=100,
                                   batches_per_epoch=None,
                                   project_name="sliding-window-vqvae-conv",
                                   experiment_name=None,
                                   use_wandb=True):
    """
    Train the sliding window VQ-VAE model with conv1d architecture
    
    Args:
        trajs: List of trajectory dictionaries
        window_size: Size of sliding window (default 60 for 2 seconds at 30fps)
        code_num: Size of codebook for each body part
        code_dim: Dimension of each code vector
        num_epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size
        save_interval: Save model every N epochs
        batches_per_epoch: Number of batches per epoch
        project_name: Project name for logging
        experiment_name: Experiment name (if None, uses timestamp)
        use_wandb: Whether to use Weights & Biases for logging
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create experiment name and save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if experiment_name is None:
        experiment_name = f"sliding_vqvae_conv_{timestamp}_{num_epochs}_alpha_2_down_2_dilation_2"

    # Initialize wandb if requested
    if use_wandb:
        config = {
            "window_size": window_size,
            "code_num": code_num,
            "code_dim": code_dim,
            "num_epochs": num_epochs,
            "lr": lr,
            "batch_size": batch_size,
            "n_joints": 29,
            "joint_dim": 2,
            "width": 128,
            "depth": 2,
            "down_t": 2,
            "stride_t": 2,
            "dilation_growth_rate": 2,
            "commitment_cost": 0.05,
            "architecture": "conv1d",
            "device": str(device),
            "experiment_name": experiment_name
        }
        
        wandb.init(
            project=project_name,
            name=experiment_name,
            config=config,
            tags=["vqvae", "conv1d", "sliding-window", "g1-robot"],
            notes=f"Training sliding window VQ-VAE with conv1d architecture on G1 robot data"
        )
        print(f"🔗 Wandb initialized: {project_name}/{experiment_name}")
    
    # Initialize model
    model = SlidingWindowVQVAEConv(
        window_size=window_size,
        n_joints=29,
        joint_dim=2,
        code_num=code_num,
        code_dim=code_dim,
        commitment_cost=0.05,
        width=128,
        depth=3,
        down_t=2,
        stride_t=2,
        dilation_growth_rate=2  # As requested
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Log model architecture to wandb
    if use_wandb:
        wandb.log({"model/total_parameters": sum(p.numel() for p in model.parameters())})
        wandb.watch(model, log="all", log_freq=100)  # Log model gradients and parameters
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Create save directory
    save_dir = os.path.join("ckpts", experiment_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Preprocess trajectories
    processed_trajs, window_info = preprocess_trajectories(trajs, target_fps=30)
    
    # Calculate batches per epoch
    total_possible_windows = sum(info[1] for info in window_info)
    if batches_per_epoch is None:
        batches_per_epoch = min(500, total_possible_windows // batch_size)
    
    print(f"Training configuration:")
    print(f"  Total possible windows: {total_possible_windows:,}")
    print(f"  Batches per epoch: {batches_per_epoch}")
    print(f"  Windows per epoch: {batches_per_epoch * batch_size:,}")
    
    # Log dataset info to wandb
    if use_wandb:
        wandb.log({
            "dataset/total_trajectories": len(trajs),
            "dataset/total_possible_windows": total_possible_windows,
            "dataset/windows_per_epoch": batches_per_epoch * batch_size,
            "training/batches_per_epoch": batches_per_epoch
        })
    
    # Clear memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"\nStarting conv1d architecture training...")
    
    # Training loop
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_vq_loss = 0.0
        epoch_perplexity = 0.0
        epoch_lower_perplexity = 0.0
        epoch_upper_perplexity = 0.0
        
        # Random sampling for each epoch
        for batch_idx in range(batches_per_epoch):
            # Generate random batch
            batch_windows = []
            for _ in range(batch_size):
                # Randomly select a trajectory and window position
                traj_idx, max_windows = window_info[torch.randint(len(window_info), (1,)).item()]
                start_frame = torch.randint(max_windows, (1,)).item()
                
                # Extract window
                window = processed_trajs[traj_idx][start_frame:start_frame + window_size]
                batch_windows.append(window)
            
            # Stack into batch
            x_batch = torch.stack(batch_windows).to(device, non_blocking=True)
            
            # Forward pass
            x_recon, vq_loss, perplexity, codes = model(x_batch)
            
            # Reconstruction loss
            recon_loss = F.mse_loss(x_recon, x_batch, reduction='mean')
            
            # Total loss
            loss =2.0 * recon_loss + vq_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Calculate individual perplexities (for detailed logging)
            with torch.no_grad():
                # Get individual quantizer losses and perplexities
                lower_body = x_batch[:, :, model.lower_body_joints, :]
                upper_body = x_batch[:, :, model.upper_body_joints, :]
                
                batch_size_curr = x_batch.shape[0]
                lower_body_flat = lower_body.reshape(batch_size_curr, window_size, -1)
                upper_body_flat = upper_body.reshape(batch_size_curr, window_size, -1)
                
                lower_body_conv = lower_body_flat.permute(0, 2, 1)
                upper_body_conv = upper_body_flat.permute(0, 2, 1)
                
                lower_encoded = model.lower_encoder(lower_body_conv)
                upper_encoded = model.upper_encoder(upper_body_conv)
                
                _, _, lower_perp = model.lower_quantizer(lower_encoded)
                _, _, upper_perp = model.upper_quantizer(upper_encoded)
            
            # Accumulate stats
            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_vq_loss += vq_loss.item()
            epoch_perplexity += perplexity.item()
            epoch_lower_perplexity += lower_perp.item()
            epoch_upper_perplexity += upper_perp.item()
            
            # Log batch-level metrics to wandb (every 50 batches to avoid spam)
            if use_wandb and (batch_idx + 1) % 50 == 0:
                wandb.log({
                    "batch/loss": loss.item(),
                    "batch/reconstruction_loss": recon_loss.item(),
                    "batch/vq_loss": vq_loss.item(),
                    "batch/perplexity": perplexity.item(),
                    "batch/lower_perplexity": lower_perp.item(),
                    "batch/upper_perplexity": upper_perp.item(),
                    "batch/epoch": epoch + 1,
                    "batch/batch_idx": batch_idx + 1
                })
            
            # Clear batch
            del x_batch, x_recon, loss, recon_loss, vq_loss, perplexity, codes
            del lower_perp, upper_perp
            
            # Memory cleanup
            if (batch_idx + 1) % 50 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Calculate averages
        avg_loss = epoch_loss / batches_per_epoch
        avg_recon_loss = epoch_recon_loss / batches_per_epoch
        avg_vq_loss = epoch_vq_loss / batches_per_epoch
        avg_perplexity = epoch_perplexity / batches_per_epoch
        avg_lower_perplexity = epoch_lower_perplexity / batches_per_epoch
        avg_upper_perplexity = epoch_upper_perplexity / batches_per_epoch
        
        # Log epoch-level metrics to wandb
        if use_wandb:
            log_dict = {
                "epoch/loss": avg_loss,
                "epoch/reconstruction_loss": avg_recon_loss,
                "epoch/vq_loss": avg_vq_loss,
                "epoch/perplexity": avg_perplexity,
                "epoch/lower_perplexity": avg_lower_perplexity,
                "epoch/upper_perplexity": avg_upper_perplexity,
                "epoch/epoch": epoch + 1,
                "epoch/learning_rate": optimizer.param_groups[0]['lr']
            }
            
            # Add GPU memory usage if available
            if torch.cuda.is_available():
                gpu_memory_mb = torch.cuda.memory_allocated(device) / 1024**2
                gpu_memory_max_mb = torch.cuda.max_memory_allocated(device) / 1024**2
                log_dict.update({
                    "system/gpu_memory_mb": gpu_memory_mb,
                    "system/gpu_memory_max_mb": gpu_memory_max_mb
                })
            
            wandb.log(log_dict)
        
        # Print statistics
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated(device) / 1024**2  # MB
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Loss: {avg_loss:.4f} | Recon: {avg_recon_loss:.4f} | VQ: {avg_vq_loss:.4f}")
            print(f"  Perplexity: {avg_perplexity:.2f} (Lower: {avg_lower_perplexity:.2f}, Upper: {avg_upper_perplexity:.2f})")
            print(f"  GPU Memory: {gpu_memory:.1f}MB")
        else:
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Loss: {avg_loss:.4f} | Recon: {avg_recon_loss:.4f} | VQ: {avg_vq_loss:.4f}")
            print(f"  Perplexity: {avg_perplexity:.2f} (Lower: {avg_lower_perplexity:.2f}, Upper: {avg_upper_perplexity:.2f})")
        
        # Save model
        if (epoch + 1) % save_interval == 0:
            save_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pt")
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': {
                    'window_size': window_size,
                    'code_num': code_num,
                    'code_dim': code_dim,
                    'n_joints': 29,
                    'joint_dim': 2
                }
            }
            torch.save(checkpoint, save_path)
            print(f"  Model saved to {save_path}")
            
            # Log model checkpoint to wandb
            if use_wandb:
                wandb.save(save_path)  # Upload checkpoint to wandb
                wandb.log({"system/checkpoint_saved": epoch + 1})
        
        # Memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Finish wandb run
    if use_wandb:
        # Log final model
        final_model_path = os.path.join(save_dir, "final_model.pt")
        torch.save({
            'epoch': num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'config': {
                'window_size': window_size,
                'code_num': code_num,
                'code_dim': code_dim,
                'n_joints': 29,
                'joint_dim': 2
            }
        }, final_model_path)
        wandb.save(final_model_path)
        print(f"✅ Final model saved to {final_model_path}")
        
        # Add summary metrics
        wandb.run.summary["final_loss"] = avg_loss
        wandb.run.summary["final_reconstruction_loss"] = avg_recon_loss
        wandb.run.summary["final_vq_loss"] = avg_vq_loss
        wandb.run.summary["final_perplexity"] = avg_perplexity
        wandb.run.summary["total_epochs"] = num_epochs
        wandb.run.summary["total_parameters"] = sum(p.numel() for p in model.parameters())
        
        print("🎉 Training completed! Check your wandb dashboard for detailed logs.")
        wandb.finish()
    
    return model


# G1 joint names and data loading functions (same as original)
G1_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"
]

def compute_joint_velocities(joint_angles, fps=30):
    """Compute joint angular velocities from joint angles"""
    T, n_joints = joint_angles.shape
    dt = 1.0 / fps
    joint_velocities = np.zeros_like(joint_angles)
    
    for i in range(T):
        if i == 0:
            joint_velocities[i] = (joint_angles[i+1] - joint_angles[i]) / dt
        elif i == T-1:
            joint_velocities[i] = (joint_angles[i] - joint_angles[i-1]) / dt
        else:
            joint_velocities[i] = (joint_angles[i+1] - joint_angles[i-1]) / (2 * dt)
    
    return joint_velocities


def load_g1_dataset(data_dir):
    """Load G1 dataset from CSV files"""
    dataset = {}
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".csv"))
    for fname in files:
        fpath = os.path.join(data_dir, fname)
        data = np.loadtxt(fpath, delimiter=',', skiprows=1)
        
        joint_angles = data[:, 1:]  # Skip time column
        joint_velocities = compute_joint_velocities(joint_angles, fps=30)
        
        # Combine angles and velocities
        joint_data = np.stack([joint_angles, joint_velocities], axis=-1)  # (T, n_joints, 2)
        
        dataset[fname.replace('.csv', '')] = {
            'joint_data': joint_data,
            'joint_names': G1_JOINTS,
            'fps': 30
        }
    
    return dataset


# Main execution
if __name__ == "__main__":
    print("Loading G1 dataset...")
    g1_data = load_g1_dataset("LAFAN1_Retargeting_Dataset/g1")
    print(f"Loaded {len(g1_data)} sequences")
    
    trajs = [g1_data[traj_name] for traj_name in g1_data.keys()]
    print(f"Found {len(trajs)} trajectories")
    print(f"Sample trajectory keys: {trajs[0].keys()}")
    print(f"Sample trajectory shape: {trajs[0]['joint_data'].shape}")
    print(f"Sample trajectory FPS: {trajs[0]['fps']}")
    
    # Train sliding window VQ-VAE with conv1d architecture
    print("\nStarting Sliding Window VQ-VAE training with Conv1D architecture...")
    trained_model = train_sliding_window_vqvae_conv(
        trajs=trajs,
        window_size=60,
        code_num=256,
        code_dim=128,
        num_epochs=8000,
        lr=3e-5,
        batch_size=16,
        save_interval=100,
        batches_per_epoch=200,
        project_name="sliding-window-vqvae-conv",
        experiment_name=None,
        use_wandb=True  # Enable wandb logging
    )
    
    # Test encoding/decoding
    print("\nTesting encoding and decoding...")
    test_traj = torch.tensor(trajs[0]['joint_data'], dtype=torch.float32)  # (T, 29, 2)
    
    # Create a test window
    if test_traj.shape[0] >= 60:
        test_window = test_traj[:60].unsqueeze(0)  # (1, 60, 29, 2)
        
        with torch.no_grad():
            # Test forward pass
            x_recon, vq_loss, perplexity, codes = trained_model(test_window)
            
            # Test encode/decode
            codes_only, latents = trained_model.encode(test_window)
            x_decoded = trained_model.decode(codes_only)
            
            print(f"Original shape: {test_window.shape}")
            print(f"Reconstructed shape: {x_recon.shape}")
            print(f"VQ Loss: {vq_loss.item():.4f}")
            print(f"Perplexity: {perplexity.item():.2f}")
            print(f"Lower codes shape: {codes['lower'].shape}")
            print(f"Upper codes shape: {codes['upper'].shape}")
            print(f"Reconstruction error: {F.mse_loss(x_recon, test_window).item():.6f}")
    
    print("Training and testing completed!")
