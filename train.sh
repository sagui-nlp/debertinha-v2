#!/bin/bash

# install uv (if not already installed)
command -v uv &>/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

uv run debertinha/data.py -i 0
uv run debertinha/tokenization.py -i 0
uv run debertinha/train.py -i 0
