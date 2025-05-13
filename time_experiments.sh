#!/usr/bin/env bash
set -euo pipefail

# --------------------------------------------
# run_experiments.sh
# Usage: ./run_experiments.sh experiments_list.txt
# --------------------------------------------

if [[ $# -ne 1 ]]; then
  echo "Usage: $(basename "$0") experiments_list.txt"
  exit 1
fi

LIST_FILE="$1"
if [[ ! -f "$LIST_FILE" ]]; then
  echo "Error: file not found: $LIST_FILE"
  exit 1
fi

# Directory where this script (and main.py) live:
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Directory where timing results are stored:
TIME_DIR_BASE="${SCRIPT_DIR}/timing_experiments"

while IFS= read -r DIR || [[ -n "$DIR" ]]; do
  # skip blank lines and lines starting with '#'
  [[ -z "${DIR// /}" ]] && continue
  [[ "${DIR#"${DIR%%[![:space:]]*}"}" =~ ^# ]] && continue

  # Check if timing results directory already exists
  EXP_TIME_DIR="${TIME_DIR_BASE}/${DIR}"
  if [[ -d "$EXP_TIME_DIR" ]]; then
    echo
    echo "=== Skipping: ${DIR} (timing results already exist at ${EXP_TIME_DIR}) ==="
    continue
  fi

  CFG_PATH="experiments/${DIR}/cfg.yaml"
  if [[ -f "$CFG_PATH" ]]; then
    echo
    echo "=== Processing: ${DIR} ==="
    echo "Copying ${CFG_PATH} → ${SCRIPT_DIR}/experiment_cfg.yaml"
    cp -f "$CFG_PATH" "${SCRIPT_DIR}/experiment_cfg.yaml"

    echo "Running main.py…"
    (
      cd "$SCRIPT_DIR"
      torchrun --master_port=29555 main.py gpu=3 timing=true
    )
  else
    echo
    echo "WARNING: ${CFG_PATH} not found; skipping ${DIR}."
  fi

done < "$LIST_FILE"

echo

echo "All done."
