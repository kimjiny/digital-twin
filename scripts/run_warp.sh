#!/usr/bin/env bash
cd ~/jy/isaaclab
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
./isaaclab.sh -p ~/jy/kiro/warp_test.py
