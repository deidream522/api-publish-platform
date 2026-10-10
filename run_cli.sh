#!/usr/bin/env bash
# COMP6713 bravo11234 - Command-line tester launcher (macOS / Linux).
# Usage:
#   bash run_cli.sh                                   # default demo prompt
#   bash run_cli.sh --text "you are an idiot"
#   bash run_cli.sh --model naive-bayes --file CODE/sample_inputs.txt

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="python3"

# Prefer a shipped venv if present.
if [ -x "$HERE/.venv/bin/python" ]; then
    PY="$HERE/.venv/bin/python"
fi

if [ "$#" -eq 0 ]; then
    echo
    echo "Running a default demo prompt. Pass your own arguments to override."
    echo "  Example:  bash run_cli.sh --text \"you are an idiot\""
    echo "            bash run_cli.sh --file CODE/sample_inputs.txt"
    echo
    exec "$PY" "$HERE/CODE/cli.py" --text "you are an absolute idiot and should shut up"
else
    exec "$PY" "$HERE/CODE/cli.py" "$@"
fi
