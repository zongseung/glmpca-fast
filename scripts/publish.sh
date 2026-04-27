#!/usr/bin/env bash
# Publish to PyPI via `uv publish`.
#
# Auth: set UV_PUBLISH_TOKEN to a PyPI API token (https://pypi.org/manage/account/token/)
#       or pass --token explicitly. Token must start with `pypi-`.
#
# To test on TestPyPI first (recommended):
#   UV_PUBLISH_URL=https://test.pypi.org/legacy/ \
#   UV_PUBLISH_TOKEN=pypi-... \
#       ./scripts/publish.sh

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -d dist || -z "$(ls -A dist 2>/dev/null)" ]]; then
    echo "dist/ is empty — run scripts/build.sh first." >&2
    exit 1
fi

uv publish dist/*
