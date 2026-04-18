#!/usr/bin/env bash
# respawn_wrapper.sh — runs respawn_missing.py and reports an accurate status.
#
# Purpose: decouples the respawn invocation from launch_as2.bash so we don't
# have to deal with escape-quoting hell across three levels of shell (outer
# bash → tmux new-window → inner bash -lc).
#
# Writes full output to /tmp/respawn_<namespace>.log (tee'd) and prints
# either SUCCESS or FAILED (with the real Python exit code) at the end.
# Holds the pane open with `sleep` so the status is visible until the user
# closes the window (Ctrl-b &).

set +e  # don't exit on individual errors — we want to report them

# Path to the challenge_multi_drone package root (parent of utils/).
PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NS="${1:-respawn}"
LOG="/tmp/respawn_${NS}.log"

cd "${PKG_DIR}" || { echo "[respawn-wrapper] cd ${PKG_DIR} failed"; sleep 86400; exit 1; }

# shellcheck source=/dev/null
source setup.bash

python3 utils/respawn_missing.py 2>&1 | tee "${LOG}"
ec=${PIPESTATUS[0]}

echo
echo "------------------------------------------------------------------"
if [ "${ec}" = "0" ]; then
  echo "[respawn-wrapper] SUCCESS. Log: ${LOG}"
else
  echo "[respawn-wrapper] FAILED (exit ${ec}). Log: ${LOG}"
fi
echo "[respawn-wrapper] Ctrl-b & to close this window."
echo "------------------------------------------------------------------"

sleep 86400
