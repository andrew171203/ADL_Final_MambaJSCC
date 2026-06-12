#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT=${1:-"./history/VSSBJSCC_diff.model"}
mkdir -p results

python main.py \
  --testset kodak \
  --model_size small \
  --C 32,64,96,128,192 \
  --multiple-snr 1,4,7,10,13 \
  --use-diffusion \
  --diffusion-steps 1 \
  --diffusion-blend 0.01 \
  --diffusion-strength 0.05 \
  --diffusion-base-channels 16 \
  --checkpoint "$CHECKPOINT" \
  2>&1 | tee results/test_ModelB_diff_multiC_multiSNR.log
