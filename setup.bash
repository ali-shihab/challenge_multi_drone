#!/bin/bash

# Source WORKSPACE
export WORKSPACE="$(pwd)/../../"
source $WORKSPACE/install/setup.bash
echo "Sourced WORKSPACE at $WORKSPACE"

# Project-specific env vars required by launch_simulation.py and respawn_missing.py.
# These must be set in every tmux pane (not just in the launch_as2.bash subprocess).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export AS2_EXTRA_DRONE_MODELS=crazyflie_led_ring
export GZ_SIM_RESOURCE_PATH="$GZ_SIM_RESOURCE_PATH:${SCRIPT_DIR}/config_sim/gazebo/models:${SCRIPT_DIR}/config_sim/gazebo/worlds:${SCRIPT_DIR}/config_sim/gazebo/plugins:${SCRIPT_DIR}/config_sim/world/models"
export IGN_GAZEBO_RESOURCE_PATH="$IGN_GAZEBO_RESOURCE_PATH:${SCRIPT_DIR}/config_sim/gazebo/models:${SCRIPT_DIR}/config_sim/gazebo/worlds:${SCRIPT_DIR}/config_sim/gazebo/plugins:${SCRIPT_DIR}/config_sim/world/models"
export GZ_SIM_SYSTEM_PLUGIN_PATH="$GZ_SIM_SYSTEM_PLUGIN_PATH:${SCRIPT_DIR}/../../install/led_ring_plugin/lib:${SCRIPT_DIR}/../../install/dynamic_moving_objects/lib"
export IGN_GAZEBO_PLUGIN_PATH="$IGN_GAZEBO_PLUGIN_PATH:${SCRIPT_DIR}/../../install/led_ring_plugin/lib:${SCRIPT_DIR}/../../install/dynamic_moving_objects/lib"