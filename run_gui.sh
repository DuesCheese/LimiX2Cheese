#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON_EXE="${VIRTUAL_ENV}/bin/python"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON_EXE="${CONDA_PREFIX}/bin/python"
else
  PYTHON_EXE="$(python -c "import sys; print(sys.executable)" 2>/dev/null || true)"
  if [[ -z "${PYTHON_EXE}" ]]; then
    PYTHON_EXE="$(python3 -c "import sys; print(sys.executable)")"
  fi
fi

exec "${PYTHON_EXE}" limix_gui.py
