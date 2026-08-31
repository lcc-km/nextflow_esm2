#!/usr/bin/env python
import argparse
import copy
import gc
import json
import math
import os
import time

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TRANSFORMERS_SUPPRESS_TORCH_LOAD_WARNING"] = "1"
os.environ["WANDB_DATA_DIR"] = os.path.abspath("./wandb_data")
os.environ["WANDB_CACHE_DIR"] = os.path.abspath("./wandb_cache")

import joblib
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from peft import LoraConfig, TaskType, get_peft_model
from scipy.stats import spearmanr
import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, StepLR
from tqdm import tqdm
import wandb

from dataset import get_dataloaders
from model import build_esm2_model


def parse_args():
    parser = argparse.ArgumentParser(description="ESM-2 Fine-tuning / LoRA with Auto-HPO for DMS screening")
    # Data and directory configurations
    parser.add_argument("--data_csv", type=str, required=True, help="Absolute path to input CSV")
    parser.add_argument("--output_dir", type=str, default="./model_weights", help="Directory to save model weights and outputs")
    parser.add_argument("--dataset_name", type=str, default="DMS_Dataset", help="Dataset identifier for logging")
    # Model and preprocessing
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")
    parser.add_argument("--num_labels", type=int, default=1, help="Output dimension (1 for regression)")
    parser.add_argument("--freeze_layers", type=int, default=0, help="Number of bottom encoder layers to freeze")
    parser.add_argument("--max_length", type=int, default=512, help="Sequence truncation threshold")
    parser.add_argument("--test_size", type=float, default=0.15, help="Test set split ratio")
    parser.add_argument("--val_size", type=float, default=0.15, help="Validation set split ratio")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for splitting")
    # Training and optimization
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=8)
    # Modification 1: reduce default lr and HPO search range to avoid destroying pretrained weights and mode collapse
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_search_range", type=float, nargs=2, default=[5e-6, 1e-4], help="Search range for HPO learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--optimizer", type=str, default="AdamW", choices=["AdamW", "Adam", "SGD"])
    parser.add_argument("--scheduler", type=str, default="CosineAnnealingLR", choices=["CosineAnnealingLR", "StepLR", "ConstantLR"])
    parser.add_argument("--eta_min", type=float, default=1e-5, help="Minimum LR for CosineAnnealingLR")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping threshold")
    # Early stopping criteria
    parser.add_argument("--patience", type=int, default=3, help="Patience epochs before triggering early stopping")
    parser.add_argument("--min_delta", type=float, default=1e-4, help="Minimum delta score to qualify as improvement")
    # LoRA parameters
    parser.add_argument("--use_lora", type=lambda x: (str(x).lower() == "true"), default=True)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--target_modules", type=str, default="query,key,value,dense", help="Comma‑separated target attention modules")
    # Optuna HPO options
    parser.add_argument("--use_hpo", type=lambda x: (str(x).lower() == "true"), default=False, help="Enable Optuna hyperparameter optimization")
    parser.add_argument("--n_trials", type=int, default=5, help="Number of trials for HPO")
    parser.add_argument("--load_hpo_params", type=str, default=None, help="Path to pre‑existing best_hpo_params.json")
    # Environment and logging
    parser.add_argument("--use_amp", type=lambda x: (str(x).lower() == "true"), default=True)
    parser.add_argument("--wandb_project", type=str, default="esm2-pten-dms")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--save_best_only", type=lambda x: (str(x).lower() == "true"), default=True)
    return parser.parse_args()


class EarlyStopping:
    """Early stopping helper based on Spearman's rank correlation."""
    def __init__(self, patience=3, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = -1.0
        self.early_stop = False

    def __call__(self, val_spearman):
        if self.best_score is None or val_spearman > (self.best_score + self.min_delta):
            self.best_score = val_spearman
            self.counter = 0
        else:
            self.counter += 1
            print(f"Validation Spearman Rho flat ({val_spearman:.4f} <= best {self.best_score:.4f}) | Early stopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


def evaluate(model, val_loader, device, criterion, use_amp, amp_dtype):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = outputs.logits.view(-1).float()
                labels_f32 = labels.view(-1).float()
                loss = criterion(preds, labels_f32)
            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_f32.cpu().numpy())
    spearman_corr, _ = spearmanr(all_preds, all_labels)
    avg_loss = total_loss / len(val_loader)
    return spearman_corr, avg_loss


def train_single_run(args, train_loader, val_loader, test_loader, tokenizer, scaler, device, amp_dtype, trial=None, is_final_run=False):
    """Executes a single training run with optional Optuna pruning."""
    model = None
    optimizer = None
    scheduler = None
    try:
        # Initialize model
        model = build_esm2_model(model_name=args.model_name, num_labels=args.num_labels, freeze_encoder_layers=args.freeze_layers, cache_dir=args.cache_dir)
        # Attach LoRA adapter if enabled
        if args.use_lora:
            target_modules_list = [mod.strip() for mod in args.target_modules.split(",")]
            peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=args.lora_r, lora_alpha=args.lora_alpha,
                                     lora_dropout=args.lora_dropout, target_modules=target_modules_list, modules_to_save=["classifier"])
            model = get_peft_model(model, peft_config)
        model.to(device)

        trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
        total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable params: {len(trainable_params)} tensors, {total_trainable:,} / {total_params:,}")
        if total_trainable == 0:
            raise RuntimeError("No trainable parameters! LoRA target_modules may not match any layers.")

        should_save_weights = (not args.use_hpo) or (trial is None) or is_final_run
        if should_save_weights:
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            joblib.dump(scaler, os.path.join(args.output_dir, "scaler.joblib"))
            if hasattr(test_loader.dataset, "df"):
                test_loader.dataset.df.to_csv(os.path.join(args.output_dir, "test.csv"), index=False)

        # Build optimizer
        optim_params = filter(lambda p: p.requires_grad, model.parameters())
        if args.optimizer == "AdamW":
            optimizer = AdamW(optim_params, lr=args.learning_rate, weight_decay=args.weight_decay)
        elif args.optimizer == "Adam":
            optimizer = Adam(optim_params, lr=args.learning_rate, weight_decay=args.weight_decay)
        elif args.optimizer == "SGD":
            optimizer = SGD(optim_params, lr=args.learning_rate, weight_decay=args.weight_decay)

        # Build lr scheduler
        total_steps = (len(train_loader) // args.gradient_accumulation_steps) * args.epochs
        if args.scheduler == "CosineAnnealingLR":
            scheduler = CosineAnnealingLR(optimizer, T_max=max(1, total_steps), eta_min=args.eta_min)
        elif args.scheduler == "StepLR":
            scheduler = StepLR(optimizer, step_size=max(1, total_steps // 3), gamma=0.5)
        else:
            scheduler = ConstantLR(optimizer, factor=1.0)

        # Modification 2: enable GradScaler only for FP16; disable under BF16
        use_scaler = args.use_amp and (device.type == "cuda") and (amp_dtype == torch.float16)
        grad_scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

        criterion = nn.MSELoss()
        early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta)
        best_spearman = -2.0
        global_step = 0
        accum_steps = args.gradient_accumulation_steps

        # Main training loop
        for epoch in range(args.epochs):
            model.train()
            running_loss = 0.0
            optimizer.zero_grad()
            for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False)):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=args.use_amp):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    preds = outputs.logits.view(-1).float()
                    labels_f32 = labels.view(-1).float()
                    loss = criterion(preds, labels_f32)
                    loss = loss / accum_steps

                # Modification 3: backward pass branch for FP16 vs BF16/FP32
                if use_scaler:
                    grad_scaler.scale(loss).backward()
                else:
                    loss.backward()

                running_loss += loss.item() * accum_steps

                if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                    # Modification 4: gradient clipping & optimizer step for FP16 / BF16‑FP32
                    if use_scaler:
                        grad_scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
                        grad_scaler.step(optimizer)
                        grad_scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
                        optimizer.step()

                    # Debug gradient norm at global_step ==1
                    if global_step == 1:
                        total_grad_norm = 0.0
                        for n, p in model.named_parameters():
                            if p.requires_grad and p.grad is not None:
                                total_grad_norm += p.grad.data.norm(2).item() ** 2
                        total_grad_norm = total_grad_norm ** 0.5
                        print(f"[Step 1] Total gradient norm: {total_grad_norm:.6f}")
                        if total_grad_norm == 0:
                            print("WARNING: All gradients are ZERO! Computation graph may be broken.")

                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1

                    log_data = {"train/batch_loss": loss.item() * accum_steps, "learning_rate": scheduler.get_last_lr()[0], "global_step": global_step}
                    if trial is not None:
                        log_data["hpo/trial_number"] = trial.number
                    wandb.log(log_data)

            avg_loss = running_loss / len(train_loader)
            val_spearman, val_loss = evaluate(model, val_loader, device, criterion, args.use_amp, amp_dtype)
            if math.isnan(val_spearman):
                val_spearman = -1.0

            epoch_log = {"epoch": epoch + 1, "train/epoch_loss": avg_loss, "val/epoch_loss": val_loss, "val/spearman_rho": val_spearman}
            if trial is not None:
                epoch_log[f"trial_{trial.number}/val_spearman"] = val_spearman
            wandb.log(epoch_log)
            print(f"Epoch {epoch + 1}/{args.epochs} | Train MSE: {avg_loss:.4f} | Val MSE: {val_loss:.4f} | Val Spearman Rho: {val_spearman:.4f}")

            # Save best model
            if val_spearman > best_spearman:
                best_spearman = val_spearman
                if should_save_weights:
                    model.save_pretrained(args.output_dir)
                    tokenizer.save_pretrained(args.output_dir)
                    joblib.dump(scaler, os.path.join(args.output_dir, "scaler.joblib"))
                    if hasattr(test_loader.dataset, "df"):
                        test_loader.dataset.df.to_csv(os.path.join(args.output_dir, "test.csv"), index=False)

            # Optuna pruning
            if trial is not None:
                trial.report(val_spearman, epoch)
                if trial.should_prune():
                    print(f"Trial {trial.number} pruned by Optuna at epoch {epoch + 1}.")
                    raise optuna.exceptions.TrialPruned()

            # Early stopping check
            early_stopping(val_spearman)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break
        return best_spearman

    finally:
        # Explicit garbage collection to release GPU VRAM
        del model, optimizer, scheduler
        if 'grad_scaler' in locals():
            del grad_scaler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    args = parse_args()
    # Load pre‑searched HPO params from json file
    if args.load_hpo_params and os.path.exists(args.load_hpo_params):
        with open(args.load_hpo_params, "r") as f:
            hpo_data = json.load(f)
        best_params = hpo_data.get("best_params", {})
        for k, v in best_params.items():
            setattr(args, k, v)
        if hasattr(args, "lora_r"):
            args.lora_alpha = args.lora_r * 2
    os.makedirs(args.output_dir, exist_ok=True)

    # Hardware detection and AMP precision setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float16
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        args.use_amp = False
        amp_dtype = torch.float32
    else:
        device = torch.device("cpu")
        args.use_amp = False
        amp_dtype = torch.float32

    # Initialize wandb
    run_name = args.wandb_run_name if args.wandb_run_name else f"{args.dataset_name}_{os.path.basename(args.model_name)}_lr{args.learning_rate}"
    wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
    current_run_id = wandb.run.id if wandb.run else "unknown_run"
    run_id_path = os.path.join(args.output_dir, "wandb_run_id.txt")
    with open(run_id_path, "w") as f:
        f.write(current_run_id)
    with open("wandb_run_id.txt", "w") as f:
        f.write(current_run_id)

    # Build dataloaders
    train_loader, val_loader, test_loader, tokenizer, scaler = get_dataloaders(
        data_csv=args.data_csv,
        model_name=args.model_name,
        batch_size=args.batch_size,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
        max_length=args.max_length,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir
    )

    if args.use_hpo:
        def objective(trial):
            trial_args = copy.deepcopy(args)
            min_lr, max_lr = trial_args.lr_search_range
            trial_args.learning_rate = trial.suggest_float("learning_rate", min_lr, max_lr, log=True)
            trial_args.weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
            trial_args.lora_r = trial.suggest_categorical("lora_r", [8, 16, 32])
            trial_args.lora_alpha = trial_args.lora_r * 2
            trial_args.lora_dropout = trial.suggest_float("lora_dropout", 0.05, 0.2)
            wandb.config.update({f"trial_{trial.number}_params": trial.params}, allow_val_change=True)
            try:
                score = train_single_run(trial_args, train_loader, val_loader, test_loader, tokenizer, scaler, device, amp_dtype, trial=trial)
                return score
            except torch.cuda.OutOfMemoryError:
                gc.collect()
                torch.cuda.empty_cache()
                return -1.0

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=args.n_trials, catch=(optuna.exceptions.TrialPruned, torch.cuda.OutOfMemoryError))
        # Overwrite args with best hyper‑parameters
        for k, v in study.best_params.items():
            setattr(args, k, v)
        args.lora_alpha = args.lora_r * 2
        wandb.config.update({"best_hpo_params": study.best_params}, allow_val_change=True)
        wandb.run.summary["best_hpo_spearman"] = study.best_value
        # Final full training run using best HPO params
        best_spearman = train_single_run(args, train_loader, val_loader, test_loader, tokenizer, scaler, device, amp_dtype, is_final_run=True)
        best_params_path = os.path.join(args.output_dir, "best_hpo_params.json")
        with open(best_params_path, "w", encoding="utf-8") as f:
            save_data = {"best_value_spearman": study.best_value, "best_params": study.best_params}
            json.dump(save_data, f, indent=4)
    else:
        best_spearman = train_single_run(args, train_loader, val_loader, test_loader, tokenizer, scaler, device, amp_dtype)

    # Upload model artifact to wandb
    if os.path.exists(args.output_dir) and len(os.listdir(args.output_dir)) > 0:
        model_artifact = wandb.Artifact(name=f"esm2-model-{wandb.run.id}", type="model", description="Best ESM‑2 model with highest validation Spearman")
        model_artifact.add_dir(args.output_dir)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                wandb.log_artifact(model_artifact)
                break
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(10)

    wandb.run.summary["best_val_spearman"] = best_spearman
    run_id_path = os.path.join(args.output_dir, "wandb_run_id.txt")
    with open(run_id_path, "w") as f:
        f.write(wandb.run.id)
    wandb.finish()


if __name__ == "__main__":
    main()
