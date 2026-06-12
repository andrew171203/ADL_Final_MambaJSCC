#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT=${1:-"./history/VSSBJSCC_base.model"}
mkdir -p results

python main.py \
  --testset kodak \
  --model_size base \
  --C 32,64,96,128,192 \
  --multiple-snr 1,4,7,10,13 \
  --checkpoint "$CHECKPOINT" \
  2>&1 | tee results/test_ModelA_base_noDiff_multiC_multiSNR.log
