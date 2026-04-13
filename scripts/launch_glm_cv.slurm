#!/bin/bash
#SBATCH --job-name=glm_cv
#SBATCH --output=/home/akj290/logs/glm/glm_cv_%A_%a.log
#SBATCH --error=/home/akj290/logs/glm/glm_cv_%A_%a.log
#SBATCH --partition=gpu_quad,gpu
#SBATCH -q gpuquad_qos
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=0-24:00:00

# ---------------------------------------------------------------------------
# launch_glm_cv.slurm — Launch run_glm_cv.py on the O2 cluster (gpu / gpu_quad).
#
# Cross-validation pipeline: fits GroupLasso PopulationGLM across a grid of
# lambda values, selects optimal lambda per neuron, refits, and evaluates.
#
# Usage (single session):
#   sbatch launch_glm_cv.slurm YRA084 260228 /path/to/config.json [N_LAMBDAS]
#
# Usage (array of sessions):
#   sbatch --array=0-9 launch_glm_cv.slurm YRA084 sessions.txt /path/to/config.json [N_LAMBDAS]
# ---------------------------------------------------------------------------

set -euo pipefail
date; hostname; pwd

# ---- Parse arguments ----
MOUSE="${1:?Usage: sbatch launch_glm_cv.slurm MOUSE SESSION_OR_LIST CONFIG [N_LAMBDAS]}"
SESSION_OR_LIST="${2:?Provide session ID or sessions list file}"
CONFIG="${3:?Provide path to run specification JSON}"
N_LAMBDAS="${4:-11}"

# Resolve session ID for array jobs
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] && [[ -f "$SESSION_OR_LIST" ]]; then
    SESSION=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$SESSION_OR_LIST")
    if [[ -z "$SESSION" ]]; then
        echo "ERROR: No session at line $((SLURM_ARRAY_TASK_ID + 1)) in $SESSION_OR_LIST"
        exit 1
    fi
else
    SESSION="$SESSION_OR_LIST"
fi

OUTPUT_DIR="/n/data2/hms/neurobio/harvey/akshay/data/glm/${MOUSE}/${SESSION}/cv_results"

echo "Mouse:     $MOUSE"
echo "Session:   $SESSION"
echo "Config:    $CONFIG"
echo "N_LAMBDAS: $N_LAMBDAS"
echo "Output:    $OUTPUT_DIR"
echo ""

# ---- Create log directory if needed ----
mkdir -p /home/akj290/logs/glm

# ---- Activate environment ----
module load conda/miniforge3
conda activate /n/data2/hms/neurobio/harvey/akshay/conda/envs/glm

# ---- Run ----
python /home/akj290/code/glm/scripts/run_glm_cv.py \
    --mouse "$MOUSE" \
    --session "$SESSION" \
    --config "$CONFIG" \
    --output "$OUTPUT_DIR" \
    --n-lambdas "$N_LAMBDAS"

echo ""
echo "Finished: $(date)"
