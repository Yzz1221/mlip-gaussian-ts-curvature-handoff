#!/usr/bin/env bash
#SBATCH --job-name=mlip-external-calcall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

: "${HORM_ROOT:?Set HORM_ROOT}"
: "${HORM_CHECKPOINT:?Set HORM_CHECKPOINT}"
export HORM_DEVICE="${HORM_DEVICE:-cuda}"
export MLIP_GAUSSIAN_PYTHON="${MLIP_GAUSSIAN_PYTHON:-python}"

input="${1:?Usage: sbatch external_calcall.sbatch INPUT_GJF}"
workdir="$(cd "$(dirname "$input")" && pwd)"
basename="$(basename "$input" .gjf)"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cp "$repo_dir/scripts/horm_external.sh" "$workdir/horm_external.sh"
chmod +x "$workdir/horm_external.sh"

cd "$workdir"
export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
g16 < "$basename.gjf" > "$basename.log"
