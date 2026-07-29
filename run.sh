#!/bin/zsh
cd "$(dirname "$0")"
# pyarrow 25.0's bundled mimalloc segfaults on this machine (macOS, Python 3.14)
# when Streamlit serializes DataFrames; the system allocator is stable.
export ARROW_DEFAULT_MEMORY_POOL=system
.venv/bin/streamlit run app.py
