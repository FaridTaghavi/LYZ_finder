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
    --re-min 20 \
    --re-max 80 \
    --dre 20 \
    --im-min 0 \
    --im-max 20 \
    --dim 10 \
    --nsub 10 \
    --nres 10 \
    --max-events 1000 \
    --ncore 10 \
	--do-roots-n2 true \
	--do-roots-n3 true \
	--do-cumulants-n2 false \
	--do-cumulants-n3 false
