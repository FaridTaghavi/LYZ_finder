#!/usr/bin/env bash
set -euo pipefail

# mkdir -p build
# cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
# cmake --build build -j"$(nproc)"

cd build
make
cd ..

INPUT_FILE="${1:-PbPb_events.root}"
OUTPUT_FOLDER="${2:-results}"

# ./build/LYZ  
./build/LYZ \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_FOLDER" \
    --do-theta-n2 true \
    --theta-k-max 200 \
    --theta-dk 0.25 \
    --theta-degree 4 \
    --re-min 0 \
    --re-max 50 \
    --dre 10 \
    --im-min 0 \
    --im-max 20 \
    --dim 10 \
    --nsub 30 \
    --nres 1000 \
    --max-events 1000 \
    --multicore-root-search true \
    --ncore 5 \
    --seed 12345 \
    --do-roots-n2 true \
    --do-roots-n3 true \
    --do-cumulants-n2 true \
    --do-cumulants-n3 true
