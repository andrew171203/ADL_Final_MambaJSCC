#!/usr/bin/env bash
set -euo pipefail

# Train VSSBJSCC-small from scratch. Use --checkpoint instead of --no-preload to resume/fine-tune.
mkdir -p results

python main.py \
  --training \
  --trainset DIV2K \
  --testset kodak \
  --model_size small \
  --C 32,64,96,128,192 \
  --multiple-snr 1,4,7,10,13 \
  --no-preload \
  2>&1 | tee results/train_ModelA_small_noDiff.log
