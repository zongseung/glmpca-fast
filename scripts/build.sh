#!/usr/bin/env bash
# Build manylinux-tagged wheel + sdist into ./dist/
# Both artefacts are PyPI-acceptable (`twine check` passes).

set -euo pipefail
cd "$(dirname "$0")/.."

rm -rf dist/ target/wheels/

# Wheel (Rust extension) — must be built via maturin to get manylinux tag.
# `uv build` goes through PEP 517 and emits `linux_x86_64`, which PyPI rejects.
uv tool run --from maturin maturin build \
    --release \
    --interpreter python3.13 \
    --manylinux 2_34 \
    --out dist

# Source distribution — fine through `uv build`.
uv build --sdist --out-dir dist

ls -la dist/
echo
echo "Validating with twine..."
uv tool run --with twine twine check dist/*
