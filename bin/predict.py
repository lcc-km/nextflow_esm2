#!/usr/bin/env python
import argparse
import os
import joblib
import numpy as np
import pandas as pd
import torch
import safetensors.torch
from dataset import get_inference_dataloader
from peft import PeftConfig, PeftModel
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import wandb
from model import build_esm2_model


def parse_args():
    parser = argparse.ArgumentParser(description="ESM-2 Prediction / Inference")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to fine-tuned model checkpoint or LoRA directory")
    parser.add_argument("--base_model_name", type=str, default="facebook/esm2_t33_650M_UR50D", help="Fallback ESM-2 base model name or path")
    parser.add_argument("--data_csv", type=str, required=True, help="Path to input CSV file for prediction")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save output prediction CSV")
    parser.add_argument("--scaler_path", type=str, default=None, help="Path to target scaler file (.joblib)")
    parser.add_argument("--batch_size", type=int, default=16, help="Inference batch size")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence truncation length")
    parser.add_argument("--seq_column", type=str, default="mutated_sequence", help="Column name containing protein sequences")
    parser.add_argument("--use_wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--wandb_project", type=str, default="esm2-pten-dms")
    parser.add_argument("--wandb_run_id", type=str, default=None, help="WandB Run ID for resuming logging")
    return parser.parse_args()


def main():
    args = parse_args()
    # Track input dataset using WandB
    run = None
    if args.use_wandb:
        run = wandb.init(
            project=args.wandb_project,
            id=args.wandb_run_id,
            resume="must" if args.wandb_run_id else None,
            job_type="inference",
            config=vars(args)
        )
        artifact = wandb.Artifact(name="prediction_inputs", type="dataset")
        artifact.add_file(args.data_csv)
        run.log_artifact(artifact)

    # Hardware setup & precision policy (Disable FP16 for ESM models to prevent overflow)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        use_amp = torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_amp else torch.float32
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        use_amp = False
        amp_dtype = torch.float32
    else:
        device = torch.device("cpu")
        use_amp = False
        amp_dtype = torch.float32

    print(f"Device: {device} | AMP Enabled: {use_amp} ({amp_dtype})")

    # Load dataset
    df = pd.read_csv(args.data_csv)
    if args.seq_column not in df.columns:
        raise ValueError(f"Column '{args.seq_column}' not found in input CSV.")
    sequences = df[args.seq_column].tolist()

    # Load model and tokenizer
    is_lora = os.path.exists(os.path.join(args.model_dir, "adapter_config.json"))

    if is_lora:
        print("Loading PEFT/LoRA model...")
        peft_config = PeftConfig.from_pretrained(args.model_dir)
        base_path = peft_config.base_model_name_or_path or args.base_model_name

        base_model = build_esm2_model(
            model_name=base_path,
            num_labels=1,
            cache_dir=args.model_dir if hasattr(args, 'cache_dir') else None,
        )

        model = PeftModel.from_pretrained(base_model, args.model_dir)
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(base_path)

    model.to(device)
    model.eval()

    dataloader = get_inference_dataloader(
        sequences=sequences,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length
    )

    # Run inference loop
    all_preds = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            if use_amp:
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    preds = outputs.logits.squeeze(-1).float()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = outputs.logits.squeeze(-1).float()

            all_preds.extend(preds.cpu().numpy())

    # Apply inverse target scaling if scaler is available
    if args.scaler_path and os.path.exists(args.scaler_path):
        scaler = joblib.load(args.scaler_path)
        preds_array = np.array(all_preds).reshape(-1, 1)
        final_preds = scaler.inverse_transform(preds_array).flatten()
    else:
        final_preds = np.array(all_preds)

    # Save output predictions
    df["prediction"] = final_preds
    df.to_csv(args.output_csv, index=False)

    if args.use_wandb and run:
        res_artifact = wandb.Artifact(name="inference_predictions", type="predictions")
        res_artifact.add_file(args.output_csv)
        run.log_artifact(res_artifact)
        run.log({"prediction_dist": wandb.Histogram(final_preds)})
        run.log({"predictions_preview": wandb.Table(dataframe=df.head(100))})
        run.finish()

    print(f"Inference finished! Saved predictions to: {args.output_csv}")


if __name__ == "__main__":
    main()