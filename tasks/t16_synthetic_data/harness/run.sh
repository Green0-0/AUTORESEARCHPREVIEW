#!/usr/bin/env bash
# run.sh <solution_dir> <out_dir> [--smoke | --cpu-smoke]  -> <out_dir>/metric.json
# Full mode: solution synthesis capped at 15 min (hard timeout), then 3 fixed classifiers (harness cost).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ $# -lt 2 ]; then echo "usage: run.sh <solution_dir> <out_dir> [--smoke|--cpu-smoke]" >&2; exit 2; fi
[ -z "${VIRTUAL_ENV:-}" ] && [ -f /home/users/kg349/AUTORESEARCHPREVIEW/env.sh ] && source /home/users/kg349/AUTORESEARCHPREVIEW/env.sh
export AR_DATA="${AR_DATA:-/usr/xtmp/kg349/ar/data}"
export AR_SEED="${AR_SEED:-0}"
# outer safety net: synthesis budget + generous classifier allowance
case "${3:-}" in
  --cpu-smoke) OUTER=600 ;;
  --smoke) OUTER=300 ;;
  *) OUTER=1800 ;;
esac
exec timeout -k 30 "$OUTER" python "$HERE/run.py" "$@"
