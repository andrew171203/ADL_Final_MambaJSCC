# Code Modification Summary

This document summarizes how this repository differs from the original SwinJSCC codebase.

## Added components

- `net/vmamba.py`: VSSBlock/SS2D implementation.
- `net/mamba_adapter.py`: adapter from SwinJSCC token tensors `[B, L, C]` to VSSBlock image-like tensors `[B, H, W, C]` and back.
- `net/diffusion_refiner.py`: lightweight conditional diffusion refiner after the JSCC decoder.
- `net/model_names.py`: canonical VSSBJSCC names plus compatibility mapping from old SwinJSCC names.
- `selective_scan/`: local CUDA selective-scan extension source used by the VSSBlock path.
- `scripts/`: reproducible scripts for testing and fine-tuning.
- `results/`: reproduced experiment logs and summarized results.

## Modified original files

- `main.py`
  - imports `VSSBJSCC` instead of the original `SwinJSCC` class;
  - adds `--use-diffusion`, `--train-diffusion-only`, `--finetune-decoder-diffusion`, and `--diffusion-*` arguments;
  - supports comma-separated multi-C and multi-SNR evaluation;
  - adds explicit checkpoint loading with `--checkpoint`, `--run-id`, `--epoch-id`, and `--no-preload`;
  - logs total/trainable/non-trainable parameter counts.

- `net/encoder.py`
  - replaces Swin Transformer layers with `VSSBlockAdapter` layers;
  - keeps the original JSCC rate/SNR adaptation logic.

- `net/decoder.py`
  - replaces Swin Transformer decoder layers with `VSSBlockAdapter` layers;
  - keeps compatibility with original reconstruction and modulation flow.

- `net/network.py`
  - renames the main model to `VSSBJSCC`;
  - preserves the original JSCC forward path as `forward_jscc`;
  - applies diffusion refinement only during evaluation when `--use-diffusion` is enabled;
  - adds `diffusion_training_loss` for diffusion-only and decoder+diffusion fine-tuning;
  - adds freeze/unfreeze helpers for staged training.

