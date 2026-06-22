#!/usr/bin/env bash
# Usage: eval.sh [TASK] [VIDEO_LEN]
TASK=${1:-Isaac-HumanNav-Room-Direct-v0}
VLEN=${2:-600}
cd ~/jy/isaaclab
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py \
    --task "$TASK" \
    --num_envs 16 \
    --headless \
    --video \
    --video_length "$VLEN" \
    --enable_cameras
