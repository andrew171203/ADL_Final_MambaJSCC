#!/usr/bin/env bash
set -euo pipefail

# Start from a trained Model A small checkpoint unless another checkpoint is passed.
CHECKPOINT=${1:-"./history/VSSBJSCC_small.model"}
mkdir -p results

python main.py \
  --training \
  --trainset DIV2K \
  --testset kodak \
  --model_size small \
  --C 32,64,96,128,192 \
  --multiple-snr 1,4,7,10,13 \
  --use-diffusion \
  --finetune-decoder-diffusion \
  --diffusion-steps 1 \
  --diffusion-blend 0.01 \
  --diffusion-strength 0.05 \
  --diffusion-base-channels 16 \
  --diffusion-train-timesteps 256 \
  --checkpoint "$CHECKPOINT" \
  2>&1 | tee results/train_ModelB_finetune_decoder_diffusion.log
