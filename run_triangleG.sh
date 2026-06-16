#!/usr/bin/env bash
set -euo pipefail

cmake -S . -B build -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
cmake --build build --target triangleG -j"${BUILD_JOBS:-$(nproc)}"

SEED=$(( $(date +%s%N) % 4294967295 ))
BOOTSTRAP_SEED=$(( (SEED + 1000000) % 4294967295 ))

./build/triangleG \
    --a 0.2 \
    --b 0.6 \
    --M-flow 30 \
    --M-nonflow 20 \
    --nonflow-q 20 \
    --nonflow-delta-phi 1.0 \
    --nonflow-delta-eta 1.0 \
    --eta-max 0.8 \
    --Neta 8 \
    --N 10000000 \
    --Nresam 1000 \
    --Nsub 300 \
    --ncore 8 \
    --q-bins 1000 \
    --generating-function all \
    --modified-gl-nodes 2 \
    --product-theta-bins 1 \
    --root-start-point 0.036722,0.0204246 \
    --root-start-point 0.0730664,0.0240179 \
    --seed "$SEED" \
    --bootstrap-seed "$BOOTSTRAP_SEED" \
    --output-folder outputs_triangleG \
    --correlations-output correlations_2_4_6_8_10.dat \
    --write-correlations 0 \
    --write-corr-lyz 0 \
    --write-vn-j0 1 \
    --nominal-output MC_generating_roots.dat \
    --bootstrap-output MC_generating_roots_bootstrap.dat \
    --root-diagnostics-output MC_generating_roots_bootstrap_diagnostics.dat \
    --coefficient-error-output MC_generating_roots_coefficient_chi2.dat \
    --vn-error-output MC_generating_roots_vn_chi2.dat \
    --vn-direct-error-source none \
    --vn-j0-error-source none \
    --coefficient-error-grid 121 \
    --coefficient-error-re-min 0.0 \
    --coefficient-error-re-max 0.95 \
    --coefficient-error-im-min 0.0 \
    --coefficient-error-im-max 0.95


#    --coefficient-error-re-min 0.0 \
#    --coefficient-error-re-max 0.95 \
#    --coefficient-error-im-min 0.0 \
#    --coefficient-error-im-max 0.95

# For full unscaled <J0(k vn)> also set --write-vn-j0 1 above.
#    --root-start-point 0.644028,0.0 \
