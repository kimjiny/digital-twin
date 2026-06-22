#!/usr/bin/env bash
# Usage: run_eval_metrics.sh [TASK] [NUM_ENVS] [STEPS]
TASK=${1:-Isaac-HumanNav-Room-Direct-v0}
NUM_ENVS=${2:-64}
STEPS=${3:-1200}
cd ~/jy/isaaclab
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
./isaaclab.sh -p ~/jy/kiro/eval_metrics.py --task "$TASK" --num_envs "$NUM_ENVS" --steps "$STEPS"
