#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python_bin="${PYTHON_BIN:-python3}"
venv_dir="${REPRO_VENV:-.venv-reproduction}"
requirements="reproduction/requirements-linux-x86_64-py312.txt"

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  echo "canonical reproduction environment requires Linux x86_64" >&2
  exit 1
fi
if [[ "$($python_bin -c 'import platform; print(platform.python_implementation(), platform.python_version())')" != "CPython 3.12.3" ]]; then
  echo "canonical reproduction environment requires CPython 3.12.3" >&2
  exit 1
fi
if [[ -e "$venv_dir" ]]; then
  echo "refusing to overwrite existing $venv_dir" >&2
  exit 1
fi

"$python_bin" -m venv "$venv_dir"
"$venv_dir/bin/python" -m pip install --require-hashes -r "$requirements"

env \
  PYTHONHASHSEED=0 \
  CUDA_VISIBLE_DEVICES= \
  OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  MKL_CBWR=COMPATIBLE \
  "$venv_dir/bin/python" scripts/check_reproduction_environment.py

echo "created canonical environment at $venv_dir"
