#!/usr/bin/env bash
set -euo pipefail

# mkdir -p build
# cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
# cmake --build build -j"$(nproc)"

cd build
make
cd ..

INPUT_FILE="${1:-PbPb_events.root}"
OUTPUT_FILE="${2:-bootstrap_roots.dat}"

./build/LYZ "$INPUT_FILE" "$OUTPUT_FILE"
