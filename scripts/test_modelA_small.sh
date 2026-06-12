#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT=${1:-"./history/VSSBJSCC_small.model"}
mkdir -p results

python main.py \
  --testset kodak \
  --model_size small \
  --C 32,64,96,128,192 \
  --multiple-snr 1,4,7,10,13 \
  --checkpoint "$CHECKPOINT" \
  2>&1 | tee results/test_ModelA_small_noDiff_multiC_multiSNR.log
