#!/bin/sh
# Benchmark prompt prefill speed (tok/s and time to first token) for this Strata install.
# It starts the model with your settings (or uses the one already running), tests prompt prefill and changes nothing.
# Options: --long (also a 32K prompt), --runs N (runs per test, default 3),
#          --prompt-sizes 1024,4096,16384 (custom token sizes),
#          --config <file>, --url <http://...>, --compare <older results folder>.
cd "$(dirname "$0")" || exit 1
if [ -x .venv/bin/python ]; then
  echo "Prefill speed check. Testing prompt prefill tok/s and time to first token."
  echo
  exec .venv/bin/python tools/strata_bench.py --prefill-only "$@"
elif command -v python3 >/dev/null 2>&1; then
  echo "Prefill speed check. Testing prompt prefill tok/s and time to first token."
  echo
  exec python3 tools/strata_bench.py --prefill-only "$@"
fi
echo "Strata is not set up here yet: run ./setup.sh first."
exit 1
