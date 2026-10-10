#!/usr/bin/env bash
# COMP6713 bravo11234 - Gradio demo launcher (macOS / Linux).
# Opens a local web UI on http://127.0.0.1:7860 with all available models.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="python3"

if [ -x "$HERE/.venv/bin/python" ]; then
    PY="$HERE/.venv/bin/python"
fi

URL="http://127.0.0.1:7860/"

echo
echo "Starting Gradio demo on ${URL}"
echo "(First launch may take 30s while transformers/gradio import.)"
echo "Press Ctrl+C to stop the server."
echo

# Try to open the browser once the server is up (non-blocking, silent if unsupported).
(
    sleep 20
    if command -v open   >/dev/null 2>&1; then open "$URL"   >/dev/null 2>&1 || true
    elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL" >/dev/null 2>&1 || true
    fi
) &

exec "$PY" "$HERE/CODE/app.py" "$@"
