#!/bin/bash
#SBATCH --job-name=glm
#SBATCH --output=/home/akj290/logs/glm/glm_%A_%a.log
#SBATCH --error=/home/akj290/logs/glm/glm_%A_%a.log
#SBATCH --partition=gpu_quad,gpu
#SBATCH -q gpuquad_qos
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00

# ---------------------------------------------------------------------------
# launch_glm.slurm — Launch run_glm.py on the O2 cluster (gpu / gpu_quad).
#
# Usage (single session):
#   sbatch launch_glm.slurm YRA084 260228 /path/to/config.json
#
# Usage (array of sessions via session list file):
#   sbatch --array=0-9 launch_glm.slurm YRA084 sessions.txt /path/to/config.json
#
#   where sessions.txt has one session ID per line:
#     260224
#     260225
#     260228
#     ...
# ---------------------------------------------------------------------------

set -euo pipefail
date; hostname; pwd

# ---- Parse arguments ----
MOUSE="${1:?Usage: sbatch launch_glm.slurm MOUSE SESSION_OR_LIST CONFIG [OUTPUT_DIR]}"
SESSION_OR_LIST="${2:?Provide session ID or sessions list file}"
CONFIG="${3:?Provide path to run specification JSON}"
OUTPUT_BASE="${4:-/n/data2/hms/neurobio/harvey/akshay/data/glm}"

# Resolve session ID: if SLURM_ARRAY_TASK_ID is set, read from list file
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] && [[ -f "$SESSION_OR_LIST" ]]; then
    SESSION=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$SESSION_OR_LIST")
    if [[ -z "$SESSION" ]]; then
        echo "ERROR: No session at line $((SLURM_ARRAY_TASK_ID + 1)) in $SESSION_OR_LIST"
        exit 1
    fi
else
    SESSION="$SESSION_OR_LIST"
fi

OUTPUT_DIR="${OUTPUT_BASE}/${MOUSE}/${SESSION}"

echo "Mouse:   $MOUSE"
echo "Session: $SESSION"
echo "Config:  $CONFIG"
echo "Output:  $OUTPUT_DIR"
echo ""

# ---- Create log directory if needed ----
mkdir -p /home/akj290/logs/glm

# ---- Activate environment ----
module load conda/miniforge3
conda activate /n/data2/hms/neurobio/harvey/akshay/conda/envs/glm

# ---- Run ----
python /home/akj290/code/glm/scripts/run_glm.py \
    --mouse "$MOUSE" \
    --session "$SESSION" \
    --config "$CONFIG" \
    --output "$OUTPUT_DIR"

echo ""
echo "Finished: $(date)"
