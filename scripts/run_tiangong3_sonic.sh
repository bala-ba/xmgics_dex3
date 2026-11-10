#!/usr/bin/env bash
set -e
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
source /opt/ros/humble/setup.bash
exec "$PROJECT_DIR/.venv/bin/python" -u -m policy.sonic_track.run_tiangong3 "$@"
