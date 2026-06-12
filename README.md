# VSSBJSCC-Diff

This repository contains the code submission for the final project:

**VSSBJSCC-Diff: Enhancing Mamba-based JSCC Image Reconstruction with a Lightweight Diffusion Refiner**

The project is built upon the open-source SwinJSCC codebase and adapts it for semantic image communication / deep joint source-channel coding (JSCC). The main goal is to explore a Mamba/VSSB-based JSCC architecture and a lightweight receiver-side diffusion refiner for image reconstruction under noisy wireless channels.

For a more detailed summary of the code-level changes from the original SwinJSCC repository, please refer to [`MODIFICATIONS.md`](MODIFICATIONS.md).

---

## 1. Project overview

This project studies image transmission over an AWGN channel using deep joint source-channel coding.

The implemented models are:

| Model | Description |
|---|---|
| `VSSBJSCC_small` | VSSBJSCC small model without diffusion |
| `VSSBJSCC_base` | VSSBJSCC base model without diffusion |
| `VSSBJSCC_diff` | VSSBJSCC small model with a lightweight diffusion refiner |

---

## 2. Main modifications

Compared with the original SwinJSCC codebase, this repository adds or modifies the following components:

| File / module | Purpose |
|---|---|
| `net/vmamba.py` | VSS / Mamba-related visual backbone components |
| `net/mamba_adapter.py` | Adapter utilities for using VSSB/Mamba blocks in the JSCC model |
| `net/diffusion_refiner.py` | Lightweight diffusion-based reconstruction refiner |
| `net/network.py` | VSSBJSCC model integration |
| `net/encoder.py` | Encoder-side architecture changes |
| `net/decoder.py` | Decoder-side architecture changes |
| `net/model_names.py` | Canonical VSSBJSCC model names and compatibility handling |
| `main.py` | Added support for VSSBJSCC, diffusion flags, checkpoint loading, and multi-C/multi-SNR evaluation |
| `selective_scan/` | Local CUDA extension source used by the VSSB / SS2D path |
| `scripts/` | Reproducible commands for training and testing |


---

## 3. Environment setup

The experiments were run with Python 3.10 and PyTorch built against CUDA 11.8.

Create the conda environment:

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate swin_mamba
```

Alternatively, install dependencies with pip:

```bash
pip install -r requirements.txt
```

---

## 4. Build the selective-scan / SSM CUDA extension

The VSSB / SS2D path requires a compiled selective-scan CUDA extension. This repository includes the local `selective_scan/` source.

Build the local extension:

```bash
cd selective_scan
```

```bash
rm -rf build dist *.egg-info
```

```bash
pip install -v --no-build-isolation .
```

```bash
cd ..
```

Verify the import:

```bash
python - <<'PY'
import selective_scan_cuda_core
print("selective_scan_cuda_core import: OK")
PY
```

If the build fails, the most common cause is a CUDA/PyTorch/compiler mismatch. Recheck `torch.version.cuda`, `nvcc -V`, and rebuild inside the same conda environment used to run `main.py`.

---

## 5. Dataset preparation

The experiments use:

| Dataset | Usage |
|---|---|
| DIV2K train HR | Training |
| Kodak | Testing |

Recommended directory structure:

```text
datasets/
├── DIV2K/
│   └── DIV2K_train_HR/
└── Kodak/
```

Example paths used during development:

```text
/home/ippnet/Andrew/datasets/DIV2K/DIV2K_train_HR
/home/ippnet/Andrew/datasets/Kodak
```

If your paths are different, update the dataset paths in `main.py`.

Datasets are not included in this repository. Please download them from the following links:

DIV2K:
https://data.vision.ee.ethz.ch/cvl/DIV2K/

Kodak Lossless True Color Image Suite:
https://r0k.us/graphics/kodak/

---

## 6. Checkpoints

Checkpoint files are not included in this GitHub repository because they are large files.

Please download the checkpoints from the [Google Drive](https://drive.google.com/drive/folders/1yHd55HKqtY3SvqRFq3qTU0iYXCOPgQeW?usp=sharing) and place them in the `history/` directory.

Expected checkpoint structure:

```text
history/
├── VSSBJSCC_small.model
├── VSSBJSCC_base.model
└── VSSBJSCC_diff.model
```

Checkpoint descriptions:

| Checkpoint | Description |
|---|---|
| `history/VSSBJSCC_small.model` | VSSBJSCC small model without diffusion |
| `history/VSSBJSCC_base.model` | VSSBJSCC base model without diffusion |
| `history/VSSBJSCC_diff.model` | VSSBJSCC small model with the lightweight diffusion refiner |

---

## 7. Testing

### Test VSSBJSCC_small

```bash
bash scripts/test_modelA_small.sh history/VSSBJSCC_small.model
```

### Test VSSBJSCC_base

```bash
bash scripts/test_modelA_base.sh history/VSSBJSCC_base.model
```

### Test VSSBJSCC_diff

```bash
bash scripts/test_modelB_diff.sh history/VSSBJSCC_diff.model
```

The testing scripts evaluate the models on Kodak under an AWGN channel with multiple channel bandwidth ratios and SNR values.

The default evaluation grid is:

```text
C = 32, 64, 96, 128, 192
SNR = 1, 4, 7, 10, 13
```

---

## 8. Training / fine-tuning

### Train VSSBJSCC_small

```bash
bash scripts/train_modelA_small.sh
```

### Fine-tune VSSBJSCC_diff

```bash
bash scripts/train_modelB_finetune_diff.sh history/VSSBJSCC_small.model
```

The diffusion fine-tuning script uses:

```text
--use-diffusion
--finetune-decoder-diffusion
--diffusion-steps 1
--diffusion-blend 0.01
--diffusion-strength 0.05
--diffusion-base-channels 16
--diffusion-train-timesteps 256
```
