#!/usr/bin/env bash
set -euo pipefail

mkdir -p build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"

./build/LYZ "${1:-PbPb_central_4.dat}"
