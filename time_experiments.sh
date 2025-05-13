#!/usr/bin/env bash
set -euo pipefail

# --------------------------------------------
# run_experiments.sh
# Usage: ./run_experiments.sh -g GPU_ID experiments_list.txt
# Example: ./run_experiments.sh -g 3 experiments_list.txt
# --------------------------------------------

# ---------- option parsing ----------
GPU_ID=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -g|--gpu)          # GPU index to use (required)
      GPU_ID="$2"
      shift 2
      ;;
    -h|--help)         # show help and exit
      echo "Usage: $(basename "$0") -g GPU_ID experiments_list.txt"
      exit 0
      ;;
    -*)
      echo "Unknown option: $1"
      exit 1
      ;;
    *)                 # first non-option → experiments list file
      LIST_FILE="$1"
      shift
      ;;
  esac
done

# ---------- basic checks ----------
if [[ -z "${GPU_ID}" || -z "${LIST_FILE:-}" ]]; then
  echo "Usage: $(basename "$0") -g GPU_ID experiments_list.txt"
  exit 1
fi

if [[ ! -f "$LIST_FILE" ]]; then
  echo "Error: file not found: $LIST_FILE"
  exit 1
fi

# Directory where this script (and main.py) live:
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Directory where timing results are stored:
TIME_DIR_BASE="${SCRIPT_DIR}/timing_experiments"

# ---------- main loop ----------
while IFS= read -r DIR || [[ -n "$DIR" ]]; do
  # skip blank lines and comments
  [[ -z "${DIR// /}" ]] && continue
  [[ "${DIR#"${DIR%%[![:space:]]*}"}" =~ ^# ]] && continue

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

    echo "Running main.py on GPU ${GPU_ID}…"
    (
      cd "$SCRIPT_DIR"
      # Uncomment the next line if you prefer CUDA-style device selection
      # export CUDA_VISIBLE_DEVICES="${GPU_ID}"
      torchrun --master_port=29555 main.py "gpu=${GPU_ID}" timing=true
    )
  else
    echo
    echo "WARNING: ${CFG_PATH} not found; skipping ${DIR}."
  fi
done < "$LIST_FILE"

echo
echo "All done."