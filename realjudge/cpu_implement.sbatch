#!/bin/bash
# Real-eval step 1 (CPU only): Claude implements the idea in solution/, then run.sh --cpu-smoke.
# Exit 0 iff the CPU smoke run is valid and the harness is untouched -> the GPU job (afterok) runs.
# usage (via `python -m ar.realjudge submit`): sbatch cpu_implement.sbatch <idea_dir>
# -t: 40 min claude timeout + <=3 min CPU smoke + copy/cleanup.
#SBATCH -J rj-impl
#SBATCH -p compsci
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 00:55:00
set -u
source /home/users/kg349/AUTORESEARCHPREVIEW/env.sh
IDEA_DIR=${1:?idea dir}
echo "rj-impl node $(hostname) job ${SLURM_JOB_ID:-} $(date) dir $IDEA_DIR"
cd "$AR_REPO" && python -m ar.realjudge job-implement --dir "$IDEA_DIR"
rc=$?
echo "rj-impl done rc=$rc $(date)"
exit $rc
