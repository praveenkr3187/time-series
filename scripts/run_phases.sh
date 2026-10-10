#!/usr/bin/env bash
# Run make targets in order, one log per target in artifacts/logs/, stop at the first failure.
#   ./scripts/run_phases.sh features global_ml deep
set -uo pipefail
mkdir -p artifacts/logs
for t in "$@"; do
  echo "$(date '+%H:%M:%S') start $t"
  if ! make "$t" > "artifacts/logs/$t.log" 2>&1; then
    echo "$(date '+%H:%M:%S') FAILED $t (see artifacts/logs/$t.log)"; exit 1
  fi
  echo "$(date '+%H:%M:%S') done $t"
done
