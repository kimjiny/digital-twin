#!/usr/bin/env bash
cd ~/jy/isaaclab
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
exec ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
    --task Isaac-HumanNav-Direct-v0 \
    --headless \
    --num_envs 1024
