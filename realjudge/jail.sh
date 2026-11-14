#!/bin/bash
# usage: [JAIL_MODE=claude|harness] jail.sh <idea_dir> <cmd...>
#   claude  (default): for the headless Claude session (see below).
#   harness: for harness runs that execute Claude-written solution code (CPU smoke, GPU full run). Same read-only
#            filesystem, but $AR_ROOT/private stays visible (the harness eval needs the test set), GPUs are
#            visible (--dev-bind /dev), and Claude's credentials/state are hidden instead of writable.
#            No network in harness mode (no downloading pretrained weights or data).
# Runs <cmd> (the headless Claude session) in a bubblewrap jail, replacing --dangerously-skip-permissions:
#   - the whole filesystem is read-only, except the idea dir and Claude's own state (~/.claude, ~/.claude.json);
#   - the project repo (tasks/ holds held-out material), other runs/probes/anchors and ~/.ssh are covered by
#     empty tmpfs mounts (even subprocesses cannot see them);
#   - $AR_ROOT/private and the raw CIFAR-100 test pickle are replaced by random decoys with the same names,
#     shapes and dtypes (scripts/make_decoy_private.py), so Claude's own `run.sh --cpu-smoke` self-tests run
#     end to end at chance accuracy; raw CIFAR-10 batches and the .tar.gz archives are hidden;
#   - writes anywhere else fail or land in a throwaway tmpfs.
# Claude Code's built-in sandbox can't be used here: its seccomp step needs a nested user namespace, which
# this cluster's kernel refuses (tested 2026-11-08). Network stays on (Claude needs the API).
set -eu
D=$(realpath "$1"); shift
R=$(realpath "${AR_ROOT:-/usr/xtmp/kg349/ar}")
H=$(realpath "$HOME")
REPO=$(realpath "${AR_REPO:-$H/AUTORESEARCHPREVIEW}")
MODE=${JAIL_MODE:-claude}
args=(--die-with-parent --ro-bind / / --proc /proc --tmpfs "$REPO")
# Allow-list of $AR_ROOT top-level dirs; everything else (runs/ except this idea dir, *_dev probe solutions,
# harness_runs, analysis, logs, anchors, probes, tmp, ...) is hidden, so strong reference solutions can't leak.
KEEP=" data venv hf cache uv-cache uv-python sandbox private decoy_private decoy_raw "
for x in "$R"/*; do
  [ -d "$x" ] || continue
  case "$KEEP" in *" $(basename "$x") "*) ;; *) args+=(--tmpfs "$x");; esac
done
if [ "$MODE" = harness ]; then args+=(--dev-bind /dev /dev --unshare-net)
else   # claude mode: real test data replaced by random decoys (scripts/make_decoy_private.py), so self-tests run
  [ -d "$R/decoy_private" ] || { echo "jail.sh: missing $R/decoy_private (run scripts/make_decoy_private.py)" >&2; exit 3; }
  args+=(--dev /dev --ro-bind "$R/decoy_private" "$R/private" --tmpfs "$R/decoy_private")
fi
[ -d "$H/.ssh" ] && args+=(--tmpfs "$H/.ssh")
if [ "$MODE" != harness ]; then   # raw CIFAR (incl. test batches); harness evals may read it, like private/
  [ -d "$R/data/cifar-10-batches-py" ] && args+=(--tmpfs "$R/data/cifar-10-batches-py")
  [ -d "$R/data/cifar-100-python" ] && args+=(--ro-bind "$R/decoy_raw/cifar-100-python" "$R/data/cifar-100-python")
  for x in "$R"/data/*.tar.gz; do [ -f "$x" ] && args+=(--ro-bind /dev/null "$x"); done
fi
args+=(--bind "$D" "$D")
if [ "$MODE" = harness ]; then
  args+=(--tmpfs "$H/.claude"); [ -f "$H/.claude.json" ] && args+=(--ro-bind /dev/null "$H/.claude.json")
else
  args+=(--bind "$H/.claude" "$H/.claude"); [ -f "$H/.claude.json" ] && args+=(--bind "$H/.claude.json" "$H/.claude.json")
fi
exec bwrap "${args[@]}" --chdir "$D" "$@"
