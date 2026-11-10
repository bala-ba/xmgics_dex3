#!/usr/bin/env bash
set -e
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
source "/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-71}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"
exec "$PROJECT_DIR/.venv/bin/python" -u rl_control_node_sim.py "$@"
