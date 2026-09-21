#!/usr/bin/env bash
#SBATCH --job-name=mlip-readfc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail

# Site-specific modules and licensed Gaussian setup belong here.
# module load gaussian/16
# source /path/to/horm/environment

: "${HORM_ROOT:?Set HORM_ROOT}"
: "${HORM_CHECKPOINT:?Set HORM_CHECKPOINT}"
export HORM_DEVICE="${HORM_DEVICE:-cpu}"

input="${1:?Usage: sbatch oneshot.sbatch INPUT_GJF OUTPUT_DIR}"
output="${2:?Usage: sbatch oneshot.sbatch INPUT_GJF OUTPUT_DIR}"

mlip-gaussian-prepare \
  --input "$input" \
  --output "$output" \
  --method 'wB97X/6-31G(d)' \
  --nproc "${SLURM_CPUS_PER_TASK}" \
  --run-sp

mlip-gaussian-run "$output/ts_freq.gjf"
mlip-gaussian-audit "$output/ts_freq.log" > "$output/opt_freq_audit.json"
