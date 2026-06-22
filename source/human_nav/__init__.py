# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Human navigation with RL-based collision avoidance.

A capsule "human" agent learns to reach a goal while avoiding static obstacles
and a moving robot. Optionally inside a loaded room (simple_room.usd).
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-HumanNav-Direct-v0",
    entry_point=f"{__name__}.human_nav_env:HumanNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.human_nav_env:HumanNavEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-HumanNav-Room-Direct-v0",
    entry_point=f"{__name__}.human_nav_env:HumanNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.human_nav_env:HumanNavRoomEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
