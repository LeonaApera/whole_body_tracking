"""
Configuration for multi-trajectory motion command
"""

from dataclasses import MISSING
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass

from .multi_trajectory_commands import MultiTrajectoryMotionCommand


@configclass
class MultiTrajectoryMotionCommandCfg(CommandTermCfg):
    """Configuration for multi-trajectory motion command."""
    
    class_type: type = MultiTrajectoryMotionCommand
    
    asset_name: str = MISSING
    """Name of the robot asset in the scene."""
    
    motion_file: str = MISSING
    """Path to motion file, directory containing multiple motion files, or wandb registry pattern."""
    
    anchor_body_name: str = MISSING
    """Name of the anchor body."""
    
    body_names: list[str] = MISSING
    """Names of bodies to track."""
    
    # Command resampling (继承自原始 MotionCommandCfg)
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    """Range for resampling time (not used in multi-trajectory mode but required for compatibility)."""
    
    debug_vis: bool = False
    """Enable debug visualization."""
    
    pose_range: dict[str, tuple[float, float]] = {}
    """Range for pose perturbations. Keys: x, y, z, roll, pitch, yaw."""
    
    velocity_range: dict[str, tuple[float, float]] = {}
    """Range for velocity perturbations. Keys: x, y, z, roll, pitch, yaw."""
    
    joint_position_range: tuple[float, float] = (-0.52, 0.52)
    """Range for joint position perturbations."""
    
    # Visualization
    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
    
    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
