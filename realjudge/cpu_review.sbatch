#!/bin/bash
# Real-eval step 3 (CPU only), runs afterany the GPU job.
#   phase 1: if the full run failed -> one repair (claude --resume) + CPU smoke -> resubmit the GPU job once
#            (and a phase-2 review job afterany it); otherwise review -> result.json.
#   phase 2: review -> result.json.
# Also copies Claude transcripts into the idea dir and removes them from ~/.claude (home file quota).
# usage: sbatch cpu_review.sbatch <idea_dir> <phase>
# -t: repair (40 min) + CPU smoke + review (15 min).
#SBATCH -J rj-review
#SBATCH -p compsci
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 01:15:00
set -u
source /home/users/kg349/AUTORESEARCHPREVIEW/env.sh
IDEA_DIR=${1:?idea dir}
PHASE=${2:-1}
echo "rj-review node $(hostname) job ${SLURM_JOB_ID:-} $(date) dir $IDEA_DIR phase $PHASE"
cd "$AR_REPO" && python -m ar.realjudge job-review --dir "$IDEA_DIR" --phase "$PHASE"
rc=$?
echo "rj-review done rc=$rc $(date)"
exit $rc
