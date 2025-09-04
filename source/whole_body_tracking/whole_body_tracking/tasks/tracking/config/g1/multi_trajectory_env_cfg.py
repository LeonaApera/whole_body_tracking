"""
G1 robot configuration for multi-trajectory motion tracking
"""

from isaaclab.utils import configclass
from whole_body_tracking.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg
from whole_body_tracking.tasks.tracking.mdp.multi_trajectory_commands_cfg import MultiTrajectoryMotionCommandCfg
import pdb
@configclass
class G1MultiTrajectoryEnvCfg(G1FlatEnvCfg):
    """G1 configuration for multi-trajectory training."""
    
    def __post_init__(self):
        super().__post_init__()
        
        # 保存原来的 body_names
        original_body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link", 
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]
        # 替换为多轨迹命令
        self.commands.motion = MultiTrajectoryMotionCommandCfg(
            asset_name="robot",
            motion_file="jianuocao0105-nanjing-university/csv_to_npz/*",  # WandB Registry pattern
            anchor_body_name="torso_link",
            body_names=original_body_names,
            pose_range={
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1), 
                "z": (-0.05, 0.05),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.2, 0.2),
            },
            velocity_range={
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.1, 0.1),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-0.3, 0.3),
            },
            joint_position_range=(-0.1, 0.1),
            debug_vis=False,
        )
