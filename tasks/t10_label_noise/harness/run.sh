#!/bin/bash
# usage: run.sh <solution_dir> <out_dir> [--smoke | --cpu-smoke]   -> <out_dir>/metric.json
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f /home/users/kg349/AUTORESEARCHPREVIEW/env.sh ]; then
  source /home/users/kg349/AUTORESEARCHPREVIEW/env.sh
fi
exec python "$HERE/run.py" "$@"
