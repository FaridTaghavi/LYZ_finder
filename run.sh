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
    --re-min 50 \
    --re-max 80 \
    --dre 10 \
    --im-min 0 \
    --im-max 30 \
    --dim 20 \
	--root-start-point 20,0 \
	--root-start-point 40,0 \
	--root-start-point 60,0 \
    --nsub 30 \
    --nres 1000 \
    --max-events 100000 \
    --ncore 10 \
    --seed 12345 \
	--root-algorithm "hybridSJ" \
	--do-roots-n2 true \
    --do-roots-n3 true \
    --do-cumulants-n2 true \
    --do-cumulants-n3 true
