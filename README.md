Language: [English](README.en.md) | [日本語](README.ja.md)

# ESM-2 Fine-Tuning & DMS Screening Pipeline

A scalable, reproducible Nextflow pipeline for fine-tuning the **ESM-2** protein language model on Deep Mutational Scanning (DMS) data to predict protein functional activity. Designed for high-throughput screening of large mutation libraries across multiple datasets in parallel.

---

## Overview

This pipeline fine-tunes ESM-2 (Evolutionary Scale Modeling 2) on DMS substitution data to learn sequence-to-function relationships. It wraps the entire training, hyperparameter optimization, and inference workflow in **Nextflow** with **Docker** containerization, enabling consistent, reproducible, and massively parallel execution across hundreds of protein datasets.

### Key Features

- **ESM-2 Fine-Tuning** — Fine-tunes Facebook's ESM-2 transformer models (e.g., `esm2_t36_3B_UR50D`) with a custom mean-pooling + MLP regression head for DMS score prediction.
- **LoRA Parameter-Efficient Fine-Tuning** — Supports Low-Rank Adaptation (LoRA) via PEFT, dramatically reducing trainable parameters while preserving model quality. Targets attention `query`, `key`, `value`, and `dense` modules.
- **Automatic Hyperparameter Optimization (HPO)** — Integrated **Optuna** search over learning rate, weight decay, LoRA rank, and dropout, with pruning of unpromising trials.
- **Three-Stage Workflow** — (1) Rapid HPO search, (2) Final full-epoch training with best parameters, (3) Batch inference on held-out test data.
- **WandB Real-Time Tracking** — Every training run, HPO trial, and inference job is logged to **Weights & Biases**, including metrics, artifacts, model checkpoints, and prediction distributions.
- **Docker Containerization** — A CUDA 12.8 + Python 3.13 + PyTorch 2.6 Docker image ensures environment stability and reproducibility across machines.
- **Parallel Multi-Dataset Execution** — Process dozens of DMS datasets simultaneously via a metadata CSV, with configurable GPU concurrency.
- **Mixed Precision (AMP)** — Automatic BF16/FP16 mixed-precision training with gradient scaling for memory efficiency and speed.
- **Early Stopping** — Stops training when validation Spearman correlation plateaus, preventing overfitting and saving compute.
- **Comprehensive Reporting** — Auto-generates Nextflow execution reports, timelines, and DAG flow diagrams.

---

## Pipeline Architecture

```
samples_metadata.csv
        │
        ▼
┌───────────────────┐
│  SAMPLE_PARSING   │  Parse metadata CSV → channel of (id, dataset CSV)
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   HPO_SEARCH      │  Optuna hyperparameter search (short epochs)
│  (opt: skip)      │  Output: best_hpo_params.json
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   TRAIN_FINAL     │  Full training with best HPO params
│                   │  Output: model weights, scaler, test split, wandb_run_id
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│     PREDICT       │  Inference on test set → predictions CSV
└───────────────────┘
```

---

## Project Structure

```
.
├── main.nf                  # Nextflow entry point & workflow orchestration
├── nextflow.config          # Pipeline configuration (params, profiles, resources)
├── Dockerfile               # CUDA 12.8 + Python 3.13 + PyTorch 2.6 container
├── requirements.txt         # Python dependencies
├── samples_metadata.csv     # Input dataset manifest (id, sample, info, data path)
├── modules/
│   ├── hpo_search.nf        # Optuna HPO search process
│   ├── train_final.nf       # Final training process (loads HPO params)
│   ├── train.nf             # Standalone training process
│   └── predict.nf           # Inference / prediction process
└── src/
    ├── train.py             # Training script (HPO + final training)
    ├── predict.py           # Inference script
    ├── model.py             # ESM-2 model with mean-pooling regression head
    └── dataset.py           # Dataset classes & data loading utilities
```

---

## Prerequisites

- **Nextflow** ≥ 23.04 (DSL 2)
- **Docker** with NVIDIA Container Toolkit (GPU support)
- **NVIDIA GPU** with CUDA ≥ 12.8 compatible driver
- **W&B account** and API key for experiment tracking

---

## Installation

### 1. Build the Docker Image

```bash
docker build -t my-gpu-app:v1.2 .
```

The image is based on `nvidia/cuda:12.8.0-runtime-ubuntu22.04` with Python 3.13, PyTorch 2.6, and all dependencies installed via `uv`.

### 2. Configure WandB

```bash
export WANDB_API_KEY="your_wandb_api_key"
```

Nextflow will inject this as a secret into each process.

### 3. Prepare Your Data

Create a `samples_metadata.csv` with the following columns:

| Column | Description |
|--------|-------------|
| `id` | Unique dataset identifier (used for output naming) |
| `sample` | Sample / protein name |
| `info` | Additional description (e.g., study reference) |
| `data` | Absolute path to the DMS data CSV |

Each DMS data CSV must contain at minimum:
- `mutated_sequence` — the protein amino acid sequence
- `DMS_score` — the functional activity score (regression target)

---

## Usage

### Basic Run

```bash
nextflow run main.nf \
    --input samples_metadata.csv \
    --output_dir ./output \
    -profile docker
```

### Skip HPO (Direct Training)

```bash
nextflow run main.nf \
    --input samples_metadata.csv \
    --output_dir ./output \
    --skip_hpo true \
    -profile docker
```

### Custom Parameters

```bash
nextflow run main.nf \
    --input samples_metadata.csv \
    --output_dir ./output \
    --model_name facebook/esm2_t33_650M_UR50D \
    --use_lora true \
    --lora_r 16 \
    --batch_size 16 \
    --epochs 30 \
    --n_trials 10 \
    --wandb_project my-esm2-project \
    -profile docker
```

---

## Configuration

All parameters are defined in `nextflow.config` and can be overridden via command-line flags.

### Data Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--seq_column` | `mutated_sequence` | Column name for protein sequences |
| `--max_length` | `405` | Maximum sequence length (truncation) |
| `--test_size` | `0.2` | Test set split ratio |
| `--random_state` | `42` | Random seed for reproducibility |

### Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_name` | `esm2_t36_3B_UR50D` | ESM-2 pretrained model path or HuggingFace ID |
| `--num_labels` | `1` | Output dimension (1 = regression) |
| `--freeze_layers` | `0` | Number of bottom encoder layers to freeze |

### LoRA Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--use_lora` | `true` | Enable LoRA fine-tuning |
| `--lora_r` | `8` | LoRA rank |
| `--lora_alpha` | `16` | LoRA alpha (scaling factor) |
| `--lora_dropout` | `0.1` | LoRA dropout rate |
| `--target_modules` | `query,key,value,dense` | Attention modules to apply LoRA |

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--batch_size` | `8` | Training batch size |
| `--gradient_accumulation_steps` | `2` | Gradient accumulation steps |
| `--epochs` | `20` | Final training epochs |
| `--learning_rate` | `5e-5` | Initial learning rate |
| `--optimizer` | `AdamW` | Optimizer (AdamW / Adam / SGD) |
| `--scheduler` | `CosineAnnealingLR` | LR scheduler |
| `--weight_decay` | `0.01` | Weight decay |
| `--use_amp` | `true` | Enable mixed precision |
| `--patience` | `3` | Early stopping patience |
| `--min_delta` | `0.0001` | Minimum improvement threshold |

### HPO Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--skip_hpo` | `false` | Skip HPO stage |
| `--n_trials` | `5` | Number of Optuna trials |
| `--epochs_hpo` | `5` | Epochs per HPO trial |
| `--lr_search_range` | `5e-6 1e-4` | Learning rate search range |

### WandB Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--wandb_project` | `esm2-dms-HPO-...` | WandB project name |
| `--wandb_run_prefix` | `esm2` | Run name prefix |

### Resource Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--gpus` | `1` | GPUs per process |
| `--cpus` | `104` | CPUs per process |
| `--memory` | `48.GB` | Memory per process |
| `--maxForks` | `2` | Max parallel processes |

---

## Output

For each dataset, the pipeline produces:

```
output_dir/
└── {dataset_id}/
    ├── final_model_weights/
    │   ├── adapter_config.json      # LoRA adapter configuration
    │   ├── adapter_model.safetensors # LoRA adapter weights
    │   ├── scaler.joblib            # Fitted StandardScaler for target inverse-transform
    │   ├── tokenizer.json           # ESM-2 tokenizer
    │   ├── train.csv                # Training split
    │   ├── val.csv                  # Validation split
    │   ├── test.csv                 # Test split (used for prediction)
    │   ├── best_hpo_params.json     # Best hyperparameters (if HPO enabled)
    │   └── wandb_run_id.txt         # WandB run ID for resuming
    └── predictions/
        └── {dataset_id}_predictions.csv  # Predicted DMS scores
```

Additionally, Nextflow generates execution reports under `output_dir/reports/`:
- `execution_report.html` — Resource usage and task summary
- `timeline.html` — Process execution timeline
- `dag.html` — Workflow DAG visualization

---

## Model Architecture

The fine-tuning head replaces ESM-2's default pooling with a **CLS-token pooling** approach followed by a two-layer MLP:

```
ESM-2 Encoder → CLS Token Embedding → Linear(hidden, hidden/2) → ReLU → Dropout → Linear(hidden/2, 1)
```

- **Loss**: MSE (Mean Squared Error) on StandardScaler-normalized DMS scores
- **Evaluation Metric**: Spearman's rank correlation coefficient between predicted and true scores
- **Target Normalization**: `StandardScaler` fit on training set; inverse-transform applied during inference

---

## WandB Integration

Each pipeline stage logs to WandB:

- **HPO Stage**: Separate project (`{project}_HPO`), per-trial metrics, best params summary
- **Training Stage**: Loss curves, validation Spearman, learning rate schedule, gradient norms, model artifact upload
- **Inference Stage**: Prediction histograms, preview tables, input/output dataset artifacts

Run names follow the pattern: `{prefix}-{model_short}-{lora|full}-{dataset_id}`.

---

## Docker Environment

The Docker image provides:

- **Base**: `nvidia/cuda:12.8.0-runtime-ubuntu22.04`
- **Python**: 3.13
- **PyTorch**: 2.6.0 with CUDA 12.8
- **Key Libraries**: transformers 5.14, peft 0.20, accelerate 1.14, optuna 4.9, wandb 0.28, scikit-learn 1.9, pandas 3.0
- **Package Manager**: `uv` for fast, reproducible installs

---

## Troubleshooting

### CUDA Out of Memory
- Reduce `--batch_size` or increase `--gradient_accumulation_steps`
- Use a smaller ESM-2 model (e.g., `esm2_t33_650M_UR50D`)
- Ensure `--use_amp true` is set

### WandB Login Failed
- Verify `WANDB_API_KEY` environment variable is set
- Run `wandb login` inside the container to test connectivity

### HPO Trials All Pruned
- Widen `--lr_search_range`
- Increase `--epochs_hpo` to allow more learning signal
- Reduce `--patience` for HPO runs

---

## License

This project is provided for research purposes. ESM-2 is released by Facebook AI Research (FAIR) under its respective license.

---

## Citation

If you use this pipeline in your research, please cite:

- ESM-2: Lin et al., "Evolutionary-scale prediction of atomic-level protein structure with a language model," *Science*, 2023.
- ProteinGym / DMS reference datasets as appropriate.
