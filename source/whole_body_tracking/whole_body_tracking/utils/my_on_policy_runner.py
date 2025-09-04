import os

from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from isaaclab_rl.rsl_rl import export_policy_as_onnx

import wandb
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


class MyOnPolicyRunner(OnPolicyRunner):
    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if self.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_policy_as_onnx(self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=filename)
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", registry_name: str = None
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        
        # 在保存模型时尝试链接到 WandB Registry
        if hasattr(self, 'registry_name') and self.registry_name:
            try:
                # 确保 registry_name 格式为 'collection:alias'
                registry_name = self.registry_name
                if ':' not in registry_name:
                    registry_name = f"{registry_name}:latest"
                
                # 如果存在 registry_name，创建一个到该 artifact 的链接
                run = wandb.run
                if run:
                    run.use_artifact(registry_name)
                    print(f"[INFO] Linked model to motion artifact collection: {registry_name}")
            except Exception as e:
                print(f"[WARNING] Could not link to motion artifact: {e}")
                # 即使链接失败，也继续保存模型
        
        if self.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_motion_policy_as_onnx(
                self.env.unwrapped, self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=filename
            )
            run = wandb.run
            if run:
                attach_onnx_metadata(self.env.unwrapped, run.name, path=policy_path, filename=filename)
                wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
