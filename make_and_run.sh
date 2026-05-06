#!/usr/bin/env bash
set -euo pipefail

# mkdir -p build
# cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
# cmake --build build -j"$(nproc)"

cd build
make
cd ..

./build/LYZ "${1:-PbPb_central_4.dat}"
