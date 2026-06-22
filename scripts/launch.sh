#!/usr/bin/env bash
# Usage: launch.sh [TASK] [NUM_ENVS] [EXTRA_ARGS...]
TASK=${1:-Isaac-HumanNav-Direct-v0}
NUM_ENVS=${2:-1024}
shift 2 2>/dev/null
EXTRA="$@"

pkill -f "train.py" 2>/dev/null
sleep 2
rm -f ~/jy/human_nav_train.log
cd ~/jy/isaaclab
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
setsid ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
    --task "$TASK" \
    --headless \
    --num_envs "$NUM_ENVS" \
    $EXTRA \
    > ~/jy/human_nav_train.log 2>&1 < /dev/null &
echo "launched pid $! task=$TASK num_envs=$NUM_ENVS extra=$EXTRA"
