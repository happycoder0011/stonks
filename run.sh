#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# pyarrow's bundled allocator can segfault on some macOS Python builds.
export ARROW_DEFAULT_MEMORY_POOL="${ARROW_DEFAULT_MEMORY_POOL:-system}"
if [[ -x ".venv/bin/streamlit" ]]; then
  exec ".venv/bin/streamlit" run app.py
fi
exec streamlit run app.py
