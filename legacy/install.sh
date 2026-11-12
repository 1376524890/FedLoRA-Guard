#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'torch==2.5.1' --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python -m pip check
python -m pip freeze > runs/requirements-lock.txt
printf '\nEnvironment installed. Activate: source .venv/bin/activate\n'
