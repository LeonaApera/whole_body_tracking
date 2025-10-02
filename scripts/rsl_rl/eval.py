"""Script to evaluate a checkpoint and collect HOVER-style metrics."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
import pdb
# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate an RL agent with HOVER-style metrics.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=10, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")
# Evaluation-specific arguments
parser.add_argument("--num_episodes", type=int, default=100, help="Number of evaluation episodes.")
parser.add_argument("--max_steps", type=int, default=500, help="Max steps per episode.")
parser.add_argument("--metrics_save_path", type=str, default="./metrics", help="Directory to save evaluation metrics.")
parser.add_argument("--command_name", type=str, default="motion", help="Name of the motion command term.")
parser.add_argument("--disable_timeout", action="store_true", default=False, help="Disable time_out termination during evaluation.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import pathlib
import torch
import numpy as np
import json
import time
from collections import defaultdict
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


def run_evaluation(env, policy, args_cli):
    """Run HOVER-style evaluation with metrics collection."""
    dt = env.unwrapped.cfg.sim.dt * env.unwrapped.cfg.decimation  # action step time for survival calc
    
    # Buffers for data collection
    all_episode_data = []
    episode_lengths = []
    
    print(f"[INFO] Starting evaluation with {args_cli.num_episodes} episodes...")
    print(f"[INFO] Max steps per episode: {args_cli.max_steps}")
    print(f"[INFO] Time step (dt): {dt:.4f} seconds")
    
    # Debug: Check available command terms
    print("[DEBUG] Available command terms:")
    for name, term in env.unwrapped.command_manager._terms.items():
        print(f"  - {name}: {type(term).__name__}")
    successful_envs = 0
    for episode in range(args_cli.num_episodes):
        print(f"[INFO] Running episode {episode + 1}/{args_cli.num_episodes}")
        
        # Reset environment
        obs, _ = env.reset()
        done = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        
        episode_data = {
            'joint_pos_robot': [],
            'joint_pos_target': [],
            'body_pos_robot': [],
            'body_pos_target': [],
            'body_vel_robot': [],
            'body_vel_target': [],
            'root_pos_robot': [],
            'root_pos_target': [],
            'root_vel_robot': [],
            'root_vel_target': [],
            'root_quat_robot': [],
            'root_quat_target': [],
        }
        
        ep_steps = 0
        episode_terminated = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        
        while ep_steps < args_cli.max_steps:
            # Run policy inference
            with torch.inference_mode():
                actions = policy(obs)
            
            # Step environment (outside inference mode to allow in-place operations)
            next_obs, _, done, info = env.step(actions)
            # pdb.set_trace()
            print(f"\r[INFO] done={done.sum().item()}/{env.num_envs}", info)
            if(ep_steps>=1):
                episode_terminated = episode_terminated | done
                print(f"\r[INFO] episode_terminated={episode_terminated.sum().item()}/{env.num_envs}")
            # Collect motion tracking data from the command term
            command = env.unwrapped.command_manager.get_term(args_cli.command_name)
            
            # Robot states (mean across environments)
            joint_pos_robot = command.robot_joint_pos
            body_pos_robot = command.robot_body_pos_w
            body_vel_robot = command.robot_body_lin_vel_w
            root_pos_robot = command.robot_anchor_pos_w
            root_vel_robot = command.robot_anchor_lin_vel_w
            root_quat_robot = command.robot_anchor_quat_w
            env_origins = getattr(env.unwrapped.scene, "env_origins", None)
            # Target states from motion
            joint_pos_target = command.joint_pos
            body_pos_target = command.body_pos_w 
        
            body_vel_target = command.body_lin_vel_w
            root_pos_target = command.anchor_pos_w 
            root_vel_target = command.anchor_lin_vel_w
            root_quat_target = command.anchor_quat_w
            # Additional diagnostics for global alignment (print only for first episode/steps)
            if episode < 3 and ep_steps < 3:
                print(f"[DEBUG] root_pos_robot (first 3): {root_pos_robot[:3]}")
                print(f"[DEBUG] root_pos_target (first 3): {root_pos_target[:3]}")
                print(f"[DEBUG] body_pos_robot (first 3): {body_pos_robot[:3]}")
                print(f"[DEBUG] body_pos_target (first 3): {body_pos_target[:3]}")
                
                # 检查数值范围和量级
                jr = joint_pos_robot.detach().cpu()
                jt = joint_pos_target.detach().cpu()
                print(f"[DEBUG] joint_pos_robot (first 6 values): {jr.flatten()[:6].numpy()}")
                print(f"[DEBUG] joint_pos_target (first 6 values): {jt.flatten()[:6].numpy()}")
                print(f"[DEBUG] joint_pos_robot (last 6 values): {jr.flatten()[-6:].numpy()}")
                print(f"[DEBUG] joint_pos_target (last 6 values): {jt.flatten()[-6:].numpy()}")
                
                # 检查根部位置
                root_r = root_pos_robot.detach().cpu()
                root_t = root_pos_target.detach().cpu()
                print(f"[DEBUG] root_pos_robot: {root_r.numpy()}")
                print(f"[DEBUG] root_pos_target: {root_t.numpy()}")
                print(f"[DEBUG] root position error (m): {torch.abs(root_r - root_t).mean().item():.4f}")                    # 检查身体位置
                body_r = body_pos_robot.detach().cpu()
                body_t = body_pos_target.detach().cpu()
                if body_r.numel() == body_t.numel():
                    body_diff = torch.norm(body_r - body_t, dim=-1) if body_r.ndim > 1 else torch.norm(body_r - body_t)
                    print(f"[DEBUG] body position errors (m): mean={body_diff.mean():.4f}, max={body_diff.max():.4f}")

            
            # Store data
            episode_data['joint_pos_robot'].append(joint_pos_robot.cpu()) # [timestep,num_envs, num_joints]
            episode_data['joint_pos_target'].append(joint_pos_target.cpu())
            episode_data['body_pos_robot'].append(body_pos_robot.cpu())
            episode_data['body_pos_target'].append(body_pos_target.cpu())
            episode_data['body_vel_robot'].append(body_vel_robot.cpu())
            episode_data['body_vel_target'].append(body_vel_target.cpu())
            episode_data['root_pos_robot'].append(root_pos_robot.cpu())
            episode_data['root_pos_target'].append(root_pos_target.cpu())
            episode_data['root_vel_robot'].append(root_vel_robot.cpu())
            episode_data['root_vel_target'].append(root_vel_target.cpu())
            episode_data['root_quat_robot'].append(root_quat_robot.cpu())
            episode_data['root_quat_target'].append(root_quat_target.cpu())

            obs = next_obs
            ep_steps += 1
            print(f"\r[INFO] Episode {episode + 1}, Step {ep_steps}/{args_cli.max_steps}")
        episode_lengths.append(ep_steps)
        # 详细的成功/失败分析
        min_steps_required = args_cli.max_steps // 2
        has_data = len(episode_data['joint_pos_robot']) > 0
        successful_envs += (episode_terminated == 0).sum().item()
        print(f"\n[DEBUG] Episode {episode + 1} summary:")
        print(f"")
        print(f"  - Successful envs: {len(episode_terminated[episode_terminated == 0])} envs")
        print(f"  - Data points collected: {len(episode_data['joint_pos_robot'])}")
        print(f"  - Has data: {has_data}")
        
        # 更宽松的成功条件：只要收集到数据就算成功
        if has_data and ep_steps >= min_steps_required:
            print(f"[INFO] Episode {episode + 1}: ✓ SUCCESS (relaxed criteria)")
            all_episode_data.append(episode_data)
        else:
            print(f"[INFO] Episode {episode + 1}: ✗ FAILED")
            if not has_data:
                print(f"  - Failure reason: No motion data collected")
        
    print(f"\n[INFO] Evaluation completed:")
    print(f"  - Total episodes: {args_cli.num_episodes}")
    print(f"  - Successful envs: {successful_envs}")
    print(f"  - Success rate: {successful_envs / (args_cli.num_episodes * args_cli.num_envs):.1f}%")

    # Compute and save metrics
    if len(all_episode_data) > 0:
        metrics = compute_hover_metrics(all_episode_data, successful_envs, dt, args_cli)
        save_metrics(metrics, args_cli.metrics_save_path)
        print_metrics_table(metrics)
    else:
        print("[WARN] No successful episodes for metric computation.")


def compute_hover_metrics(all_episode_data, successful_envs, dt, args_cli):
    """Compute HOVER-style evaluation metrics."""
    print("[INFO] Computing evaluation metrics...")
    
    # Aggregate all successful episode data
    all_joint_pos_robot = []
    all_joint_pos_target = []
    all_body_pos_robot = []
    all_body_pos_target = []
    all_body_vel_robot = []
    all_body_vel_target = []
    all_root_pos_robot = []
    all_root_pos_target = []
    all_root_vel_robot = []
    all_root_vel_target = []
    
    for i, episode_data in enumerate(all_episode_data):
        if len(episode_data['joint_pos_robot']) > 0:
            episode_steps = len(episode_data['joint_pos_robot'])
            print(f"[DEBUG] Episode {i+1}: {episode_steps} timesteps")
            
            all_joint_pos_robot.extend(episode_data['joint_pos_robot'])
            all_joint_pos_target.extend(episode_data['joint_pos_target'])
            all_body_pos_robot.extend(episode_data['body_pos_robot'])
            all_body_pos_target.extend(episode_data['body_pos_target'])
            all_body_vel_robot.extend(episode_data['body_vel_robot'])
            all_body_vel_target.extend(episode_data['body_vel_target'])
            all_root_pos_robot.extend(episode_data['root_pos_robot'])
            all_root_pos_target.extend(episode_data['root_pos_target'])
            all_root_vel_robot.extend(episode_data['root_vel_robot'])
            all_root_vel_target.extend(episode_data['root_vel_target'])
    
    if len(all_joint_pos_robot) == 0:
        print("[ERROR] No data collected from any episode!")
        return {}
    
    print(f"[DEBUG] Total timesteps aggregated: {len(all_joint_pos_robot)}")
    
    # Convert to tensors
    all_joint_pos_robot = torch.stack(all_joint_pos_robot)
    all_joint_pos_target = torch.stack(all_joint_pos_target)
    all_body_pos_robot = torch.stack(all_body_pos_robot)
    all_body_pos_target = torch.stack(all_body_pos_target)
    all_body_vel_robot = torch.stack(all_body_vel_robot)
    all_body_vel_target = torch.stack(all_body_vel_target)
    all_root_pos_robot = torch.stack(all_root_pos_robot)
    all_root_pos_target = torch.stack(all_root_pos_target)
    all_root_vel_robot = torch.stack(all_root_vel_robot)
    all_root_vel_target = torch.stack(all_root_vel_target)
    
    print(f"[DEBUG] Tensor shapes after stacking:")
    print(f"  - all_joint_pos_robot: {all_joint_pos_robot.shape}")
    print(f"  - all_joint_pos_target: {all_joint_pos_target.shape}")
    print(f"  - all_body_pos_robot: {all_body_pos_robot.shape}")
    print(f"  - all_root_pos_robot: {all_root_pos_robot.shape}")
    
    # Calculate metrics
    success_rate = (successful_envs / (args_cli.num_episodes * args_cli.num_envs)) * 100
    print(f"[DEBUG] Success rate: {success_rate:.2f}%")

    # Joint position error (MPJPE)
    joint_diff =  torch.abs(all_joint_pos_robot - all_joint_pos_target) # [timesteps, num_envs,num_joints]
    mpjpe = torch.mean(joint_diff)  

    print(f"[DEBUG] Joint diff shape: {joint_diff.shape}")
    print(f"[DEBUG] Corrected MPJPE (rad): {mpjpe:.4f}")

    # Global body position error (mm)
    body_pos_errors = torch.norm(all_body_pos_robot - all_body_pos_target, dim=-1)
    print(f"[DEBUG] Body pos errors shape: {body_pos_errors.shape}")
    print(f"[DEBUG] Body pos errors (m): mean={body_pos_errors.mean():.4f}, max={body_pos_errors.max():.4f}")
    g_mpkpe = torch.mean(body_pos_errors) * 1000  # Convert to mm

    # Local body position error (relative to root)
    # Subtract root positions to get local coordinates
    print(f"[DEBUG] Computing local body errors...")
    print(f"[DEBUG] Root pos shapes - robot: {all_root_pos_robot.shape}, target: {all_root_pos_target.shape}")
    local_body_robot = all_body_pos_robot - all_root_pos_robot.unsqueeze(2)
    local_body_target = all_body_pos_target - all_root_pos_target.unsqueeze(2)
    local_body_errors = torch.norm(local_body_robot - local_body_target, dim=-1)
    print(f"[DEBUG] Local body errors (m): mean={local_body_errors.mean():.4f}, max={local_body_errors.max():.4f}")
    l_mpkpe = torch.mean(local_body_errors) * 1000  # Convert to mm
    
    # Velocity errors (mm/frame)
    body_vel_errors = torch.norm(all_body_vel_robot - all_body_vel_target, dim=-1)
    print(f"[DEBUG] Body vel errors (m/s): mean={body_vel_errors.mean():.4f}, max={body_vel_errors.max():.4f}")
    e_vel = torch.mean(body_vel_errors) * 1000  # Convert to mm/frame
    
    # Root velocity errors
    root_vel_errors = torch.norm(all_root_vel_robot - all_root_vel_target, dim=-1)
    root_vel_error = torch.mean(root_vel_errors)
    
    # Root position errors
    root_pos_errors = torch.norm(all_root_pos_robot - all_root_pos_target, dim=-1)
    root_pos_error = torch.mean(root_pos_errors)
    print(f"[DEBUG] Root pos errors (m): mean={root_pos_error:.4f}")
    print(f"[DEBUG] Root vel errors (m/s): mean={root_vel_error:.4f}")
    
    print(f"\n[DEBUG] Final metrics before formatting:")
    print(f"  - MPJPE: {mpjpe:.4f} rad")
    print(f"  - G-mpkpe: {g_mpkpe:.2f} mm") 
    print(f"  - L-mpkpe: {l_mpkpe:.2f} mm")
    print(f"  - Velocity error: {e_vel:.2f} mm/frame")
    # pdb.set_trace()
    metrics = {
        "success_rate_percent": float(success_rate),
        "mpjpe_rad": float(mpjpe),
        "g_mpkpe_mm": float(g_mpkpe),
        "l_mpkpe_mm": float(l_mpkpe),
        "e_vel_mm_per_frame": float(e_vel),
        "root_vel_error_m_per_s": float(root_vel_error),
        "root_pos_error_m": float(root_pos_error),
        "total_episodes": args_cli.num_episodes,
        "total_timesteps": len(all_joint_pos_robot),
        "evaluation_timestamp": datetime.now().isoformat(),
    }
    
    return metrics


def save_metrics(metrics, save_path):
    """Save metrics to JSON file."""
    os.makedirs(save_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"evaluation_metrics_{timestamp}.json"
    filepath = os.path.join(save_path, filename)
    
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"[INFO] Metrics saved to: {filepath}")


def print_metrics_table(metrics):
    """Print metrics in a formatted table."""
    print("\n" + "="*60)
    print("HOVER-Style Evaluation Metrics")
    print("="*60)
    print(f"| {'Metric':<25} | {'Value':<30} |")
    print("|" + "-"*27 + "|" + "-"*32 + "|")
    
    metric_format = {
        "success_rate_percent": "Success Rate (%)",
        "mpjpe_rad": "MPJPE (rad)",                        # 修正键名
        "g_mpkpe_mm": "Global MPKPE (mm)",
        "l_mpkpe_mm": "Local MPKPE (mm)",
        "e_vel_mm_per_frame": "Velocity Error (mm/frame)",
        "root_vel_error_m_per_s": "Root Vel Error (m/s)",
        "root_pos_error_m": "Root Pos Error (m)",
        "total_episodes": "Total Episodes",
        "total_timesteps": "Total Timesteps",              # 添加缺失的键
    }
    
    for key, display_name in metric_format.items():
        if key in metrics:
            value = metrics[key]
            if isinstance(value, float):
                if "percent" in key or "mm" in key or "error" in key:
                    formatted_value = f"{value:.3f}"
                else:
                    formatted_value = f"{value:.2f}"
            else:
                formatted_value = str(value)
            print(f"| {display_name:<25} | {formatted_value:<30} |")
    
    print("="*60)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Evaluate with RSL-RL agent and collect HOVER-style metrics."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    if args_cli.wandb_path:
        import wandb

        run_path = args_cli.wandb_path

        api = wandb.Api()
        if "model" in args_cli.wandb_path:
            run_path = "/".join(args_cli.wandb_path.split("/")[:-1])
        wandb_run = api.run(run_path)
        # loop over files in the run
        files = [file.name for file in wandb_run.files() if "model" in file.name]
        # files are all model_xxx.pt find the largest filename
        if "model" in args_cli.wandb_path:
            file = args_cli.wandb_path.split("/")[-1]
        else:
            file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))

        wandb_file = wandb_run.file(str(file))
        wandb_file.download("./logs/rsl_rl/temp", replace=True)

        print(f"[INFO]: Loading model checkpoint from: {run_path}/{file}")
        resume_path = f"./logs/rsl_rl/temp/{file}"

        if args_cli.motion_file is not None:
            print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
            env_cfg.commands.motion.motion_file = args_cli.motion_file

        art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
        if art is None:
            print("[WARN] No model artifact found in the run.")
        else:
            env_cfg.commands.motion.motion_file = str(pathlib.Path(art.download()) / "motion.npz")

    else:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # Disable time_out termination for evaluation if requested
    if args_cli.disable_timeout:
        if hasattr(env_cfg, 'terminations') and hasattr(env_cfg.terminations, 'time_out'):
            print("[INFO] Disabling time_out termination for evaluation")
            delattr(env_cfg.terminations, 'time_out')
        # Alternative: Use evaluation-specific termination config
        # from whole_body_tracking.tasks.tracking.tracking_env_cfg import TerminationsCfgEval
        # env_cfg.terminations = TerminationsCfgEval()
    
    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    log_dir = os.path.dirname(resume_path)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "eval"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during evaluation.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit (optional - comment out if not needed)
    # export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    # export_motion_policy_as_onnx(
    #     env.unwrapped,
    #     ppo_runner.alg.policy,
    #     normalizer=ppo_runner.obs_normalizer,
    #     path=export_model_dir,
    #     filename="policy.onnx",
    # )
    # attach_onnx_metadata(env.unwrapped, args_cli.wandb_path if args_cli.wandb_path else "none", export_model_dir)

    # Run evaluation
    print("[INFO] Running HOVER-style evaluation with metrics.")
    run_evaluation(env, policy, args_cli)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
