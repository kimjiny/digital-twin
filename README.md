# Human-Nav-RL: RL-based Collision-Avoiding Human Agent in Isaac Lab

A reinforcement-learning **human agent** for indoor digital twins, built on
**NVIDIA Isaac Sim 5.1 / Isaac Lab 2.3**. The agent (a capsule "human") learns
to navigate to a goal while **avoiding static obstacles and a moving robot**,
optionally inside a loaded room USD (`simple_room.usd`).

This is *Phase 0* of a larger pipeline whose goal is a robot-interactable,
RL-avoiding human that can be dropped into a scanned indoor twin alongside a
humanoid robot (G1 + GR00T).

## Results (Phase 0)

| Task | Setup | Total reward (mean) |
|---|---|---|
| Static-obstacle avoidance (v1) | empty scene, 4.8k steps | −8.8 → +51 |
| + Moving robot (v2) | empty scene, 4.8k steps | −14.6 → +39 |
| In room (`simple_room.usd`) | 256 envs, 4.8k steps | −18.7 → +20 |
| In room, longer | 256 envs, 24k steps | −7.8 → +53 |
| **In room, + proximity shaping** | **256 envs, 48k steps** | **+14.7 → +50** |

**Evaluation rollout** (room policy, deterministic):

| outcome | baseline (24k) | **+ proximity shaping (48k)** |
|---|---|---|
| reached goal (success) | 69.4% | **98.2%** |
| hit static obstacle | 26.5% | **1.1%** |
| hit moving robot | 4.1% | **0.7%** |
| timeout | 0% | 0% |

The dense proximity penalty (graded cost inside a caution band around obstacles
and the robot) plus longer training cut collisions drastically — the agent
learns to keep a safety margin while still reaching the goal.

## Environment

- **Agent**: capsule rigid body, planar velocity action `(vx, vy)`.
- **Observation (14-D)**: vector-to-goal (2), own velocity (2), relative
  positions of 3 static obstacles (6), relative position of moving robot (2),
  robot velocity (2).
- **Reward**: progress to goal, +50 goal bonus, −25 obstacle collision,
  −30 robot collision, **dense proximity penalty** inside a caution band around
  obstacles/robot, small step penalty.
- **Termination**: goal reached / collision / timeout.
- **Moving robot**: a kinematic cylinder patrolling along the y-axis that the
  agent must avoid.

Two registered tasks:

- `Isaac-HumanNav-Direct-v0` — empty scene.
- `Isaac-HumanNav-Room-Direct-v0` — loads `simple_room.usd` per env.

## Layout

```
source/human_nav/            # Isaac Lab Direct RL task package
  __init__.py                #   gym registration (2 tasks)
  human_nav_env.py           #   DirectRLEnv + cfgs
  agents/skrl_ppo_cfg.yaml   #   skrl PPO config
scripts/
  launch.sh                  # train:   launch.sh <TASK> <NUM_ENVS>
  run_eval_metrics.sh        # eval:    success/collision rates
  eval_metrics.py            #   rollout evaluation script
  read_tb.py                 # read reward curve from tensorboard logs
  warp_test.py               # CUDA / Warp sanity check
```

## Install

Requires Isaac Sim 5.1 + Isaac Lab 2.3 and `skrl`.

```bash
# inside the Isaac Lab python env
./isaaclab.sh -p -m pip install skrl

# place the task package under the Isaac Lab tasks directory:
cp -r source/human_nav \
  <IsaacLab>/source/isaaclab_tasks/isaaclab_tasks/direct/human_nav
```

## Train

```bash
# from the IsaacLab root
TASK=Isaac-HumanNav-Room-Direct-v0
LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
  ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
    --task $TASK --headless --num_envs 256
```

## Evaluate

```bash
LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
  ./isaaclab.sh -p scripts/eval_metrics.py \
    --task Isaac-HumanNav-Room-Direct-v0 --num_envs 16 --steps 400
```

## Notes

- **CUDA stub fix**: prepend `/lib/x86_64-linux-gnu` to `LD_LIBRARY_PATH` so the
  real `libcuda.so.1` is loaded instead of the CUDA-toolkit stub (otherwise
  Warp/PhysX raise `CUDA error 34`).
- **skrl 2.x**: `play.py` from Isaac Lab 2.3 calls the removed
  `set_running_mode`; use `scripts/eval_metrics.py` instead. Its `act()` call
  passes both `states` and `observations` for skrl-2.x compatibility.

## Roadmap

- Reduce static-obstacle collisions (reward shaping, longer training).
- Phase 1: swap the capsule for an SMPL-X human mesh (appearance).
- Phase 2: full-body physics human (ProtoMotions / PHC) for two-way interaction.
- Integrate with a GR00T-driven G1 robot for human–robot co-existence.
