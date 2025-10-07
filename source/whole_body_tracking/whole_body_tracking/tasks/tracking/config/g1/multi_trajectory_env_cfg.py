"""
G1 robot configuration for multi-trajectory motion tracking
"""

from isaaclab.utils import configclass
from whole_body_tracking.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg
from whole_body_tracking.tasks.tracking.mdp.multi_trajectory_commands import MultiTrajectoryMotionCommandCfg
import pdb
@configclass
class G1MultiTrajectoryEnvCfg(G1FlatEnvCfg):
    """G1 configuration for multi-trajectory training."""
    
    def __post_init__(self):
        super().__post_init__()

        # 替换为多轨迹命令
        self.commands.motion = MultiTrajectoryMotionCommandCfg(
            asset_name="robot",
            # motion_files="jianuocao0105-nanjing-university/csv_to_npz/*",  # WandB Registry pattern
            motion_files=["/home/yuxin/Projects/VQVAE/VAE/lanfan1_npz/dance1_subject1:v0/motion.npz",
            "/home/yuxin/Projects/VQVAE/VAE/lanfan1_npz/dance1_subject2:v0/motion.npz",
            "/home/yuxin/Projects/VQVAE/VAE/lanfan1_npz/dance1_subject3:v0/motion.npz",
            "/home/yuxin/Projects/VQVAE/VAE/lanfan1_npz/dance2_subject1:v0/motion.npz",
            "/home/yuxin/Projects/VQVAE/VAE/lanfan1_npz/dance2_subject2:v0/motion.npz",
            "/home/yuxin/Projects/VQVAE/VAE/lanfan1_npz/dance2_subject3:v0/motion.npz",
            "/home/yuxin/Projects/VQVAE/VAE/lanfan1_npz/dance2_subject4:v0/motion.npz",
            # "/home/yuxin/Projects/VQVAE/VAE/lanfan1_npz/dance2_subject5:v0/motion.npz",
            "/home/yuxin/Projects/VQVAE/VAE/lanfan1_npz/fallAndGetUp1_subject1:v0/motion.npz",
            "/home/yuxin/Projects/VQVAE/VAE/lanfan1_npz/fallAndGetUp1_subject4:v0/motion.npz",
            "/home/yuxin/Projects/VQVAE/VAE/lanfan1_npz/fallAndGetUp1_subject5:v0/motion.npz",
            "/home/yuxin/Projects/VQVAE/VAE/lanfan1_npz/fallAndGetUp2_subject2:v0/motion.npz",
            "/home/yuxin/Projects/VQVAE/VAE/lanfan1_npz/fallAndGetUp2_subject3:v0/motion.npz",
            "/home/yuxin/Projects/VQVAE/VAE/lanfan1_npz/fallAndGetUp3_subject1:v0/motion.npz",
            "/home/yuxin/Projects/VQVAE/VAE/lanfan1_npz/fight1_subject2:v0/motion.npz",
            "/home/yuxin/Projects/VQVAE/VAE/lanfan1_npz/fight1_subject3:v0/motion.npz",
            "/home/yuxin/Projects/VQVAE/VAE/lanfan1_npz/fight1_subject5:v0/motion.npz",
            "/home/yuxin/Projects/VQVAE/VAE/lanfan1_npz/fightAndSports1_subject1:v0/motion.npz",
            "/home/yuxin/Projects/VQVAE/VAE/lanfan1_npz/fightAndSports1_subject4:v0/motion.npz",
            "/home/yuxin/Projects/VQVAE/VAE/lanfan1_npz/jumps1_subject1:v0/motion.npz",
            "/home/yuxin/Projects/VQVAE/VAE/lanfan1_npz/jumps1_subject2:v0/motion.npz",
            "/home/yuxin/Projects/VQVAE/VAE/lanfan1_npz/jumps1_subject5:v0/motion.npz",
            "/home/yuxin/Projects/VQVAE/VAE/lanfan1_npz/run1_subject2:v0/motion.npz",
            "/home/yuxin/Projects/VQVAE/VAE/lanfan1_npz/run1_subject5:v0/motion.npz",
            "/home/yuxin/Projects/VQVAE/VAE/lanfan1_npz/run2_subject1:v0/motion.npz",
            "/home/yuxin/Projects/VQVAE/VAE/lanfan1_npz/run2_subject4:v0/motion.npz",
            "/home/yuxin/Projects/VQVAE/VAE/lanfan1_npz/sprint1_subject2:v0/motion.npz",
            "/home/yuxin/Projects/VQVAE/VAE/lanfan1_npz/sprint1_subject4:v0/motion.npz",
            "/home/yuxin/Projects/VQVAE/VAE/lanfan1_npz/walk1_subject1:v0/motion.npz",
            "/home/yuxin/Projects/VQVAE/VAE/lanfan1_npz/walk1_subject2:v0/motion.npz",
            "/home/yuxin/Projects/VQVAE/VAE/lanfan1_npz/walk1_subject5:v0/motion.npz",
            "/home/yuxin/Projects/VQVAE/VAE/lanfan1_npz/walk2_subject1:v0/motion.npz",
            "/home/yuxin/Projects/VQVAE/VAE/lanfan1_npz/walk2_subject3:v0/motion.npz",
            "/home/yuxin/Projects/VQVAE/VAE/lanfan1_npz/walk2_subject4:v0/motion.npz",
            "/home/yuxin/Projects/VQVAE/VAE/lanfan1_npz/walk3_subject1:v0/motion.npz",
            "/home/yuxin/Projects/VQVAE/VAE/lanfan1_npz/walk3_subject2:v0/motion.npz",
            "/home/yuxin/Projects/VQVAE/VAE/lanfan1_npz/walk3_subject3:v0/motion.npz",
            "/home/yuxin/Projects/VQVAE/VAE/lanfan1_npz/walk3_subject4:v0/motion.npz",
            # "/home/yuxin/Projects/VQVAE/VAE/lanfan1_npz/walk3_subject5:v0/motion.npz",
            "/home/yuxin/Projects/VQVAE/VAE/lanfan1_npz/walk4_subject1:v0/motion.npz",
            ],
            anchor_body_name="torso_link",
            body_names=self.commands.motion.body_names,
            pose_range={
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05), 
                "z": (-0.05, 0.05),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.2, 0.2),
            },
            velocity_range={
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.2, 0.2),
                "roll": (-0.52, 0.52),
                "pitch": (-0.52, 0.52),
                "yaw": (-0.78, 0.78),
            },
            joint_position_range=(-0.1, 0.1),
            debug_vis=False,
        )
