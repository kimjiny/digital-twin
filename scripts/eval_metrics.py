"""Evaluate a trained HumanNav policy: success / collision / timeout rates.

Avoids play.py's skrl-2.x-incompatible set_running_mode call.
Runs headless (no video) to dodge room+fabric render warnings.
"""

import argparse
import glob
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-HumanNav-Room-Direct-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=1200)
parser.add_argument("--checkpoint", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402  (registers tasks)
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from skrl.utils.runner.torch import Runner  # noqa: E402


def find_checkpoint(task_dir_key="human_nav_direct"):
    base = os.path.expanduser(f"~/jy/isaaclab/logs/skrl/{task_dir_key}")
    runs = sorted(glob.glob(os.path.join(base, "*")))
    run = runs[-1]
    ckpts = glob.glob(os.path.join(run, "checkpoints", "agent_*.pt"))
    # pick the highest step count
    ckpts.sort(key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]))
    return ckpts[-1]


ckpt = args_cli.checkpoint or find_checkpoint()
print(f"[EVAL] task={args_cli.task} num_envs={args_cli.num_envs} ckpt={ckpt}")

env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")

env = gym.make(args_cli.task, cfg=env_cfg)
core = env.unwrapped  # underlying DirectRLEnv (HumanNavEnv)
env = SkrlVecEnvWrapper(env)

experiment_cfg["trainer"]["close_environment_at_exit"] = False
experiment_cfg["agent"]["experiment"]["write_interval"] = 0
experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

runner = Runner(env, experiment_cfg)
runner.agent.load(ckpt)
try:
    runner.agent.set_running_mode("eval")
except Exception:
    pass

reached = obs_hit = robot_hit = timeout = 0

obs, _ = env.reset()
print(f"[EVAL] obs type={type(obs)} shape={getattr(obs, 'shape', None)}", flush=True)
with torch.inference_mode():
    for step in range(args_cli.steps):
        try:
            outputs = runner.agent.act(states=obs, observations=obs, timestep=0, timesteps=0)
        except TypeError:
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)

        if isinstance(outputs, (tuple, list)):
            last = outputs[-1]
            actions = last.get("mean_actions", outputs[0]) if isinstance(last, dict) else outputs[0]
        else:
            actions = outputs
        if step == 0:
            print(f"[EVAL] action shape={getattr(actions, 'shape', None)}", flush=True)

        obs, _, terminated, truncated, _ = env.step(actions)

        term = terminated.bool().view(-1)
        trunc = truncated.bool().view(-1)
        r = core._reached
        ho = core._hit_obs
        hr = core._hit_robot

        reached += int((term & r).sum().item())
        obs_hit += int((term & ho & ~r).sum().item())
        robot_hit += int((term & hr & ~r & ~ho).sum().item())
        timeout += int((trunc & ~term).sum().item())

total = reached + obs_hit + robot_hit + timeout
print("==================== EVAL RESULTS ====================", flush=True)
print(f"episodes      : {total}", flush=True)
if total > 0:
    print(f"success (goal): {reached:5d}  ({100.0*reached/total:5.1f}%)", flush=True)
    print(f"hit obstacle  : {obs_hit:5d}  ({100.0*obs_hit/total:5.1f}%)", flush=True)
    print(f"hit robot     : {robot_hit:5d}  ({100.0*robot_hit/total:5.1f}%)", flush=True)
    print(f"timeout       : {timeout:5d}  ({100.0*timeout/total:5.1f}%)", flush=True)
print("======================================================", flush=True)

env.close()
simulation_app.close()
