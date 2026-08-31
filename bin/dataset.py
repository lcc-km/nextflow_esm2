#!/usr/bin/env python
import os
import joblib
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class ProteinDMSDataset(Dataset):
    """Dataset class for training, validation, and evaluation (requires ground-truth labels)."""

    def __init__(self, sequences, labels, tokenizer, max_length=512):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = str(self.sequences[idx])
        label = float(self.labels[idx])

        encoding = self.tokenizer(
            seq,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.float),
        }


class InferenceDataset(Dataset):
    """Dataset class for unlabelled sequence inference."""

    def __init__(self, sequences, tokenizer, max_length=512):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = str(self.sequences[idx])
        encoding = self.tokenizer(
            seq,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }


def get_dataloaders(
    data_csv: str,
    model_name: str,
    batch_size: int = 16,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    max_length: int = 512,
    seq_column: str = "mutated_sequence",
    label_column: str = "DMS_score",
    output_dir: str = "./final_model_weights",
    cache_dir: str = None,
):
    """Loads CSV, splits data (Train/Val/Test), normalizes target labels, saves preprocessed artifacts, and returns DataLoaders."""
    print(f"Loading raw dataset: {data_csv}")
    df = pd.read_csv(data_csv)

    if seq_column not in df.columns or label_column not in df.columns:
        raise ValueError(
            f"Missing required columns in CSV: '{seq_column}' or '{label_column}'."
        )

    df = df.dropna(subset=[seq_column, label_column])

    # 1. First split: separate Test set from combined Train+Val set
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )

    # 2. Second split: divide remaining data into Train and Val sets
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_size, random_state=random_state
    )

    # Create explicit copies to avoid Pandas SettingWithCopyWarning
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    scaler = StandardScaler()

    # Fit and transform target scores on Training set only
    train_df["DMS_score_scaled"] = scaler.fit_transform(train_df[[label_column]])

    # Transform Validation and Test sets using Training set statistics
    val_df["DMS_score_scaled"] = scaler.transform(val_df[[label_column]])
    test_df["DMS_score_scaled"] = scaler.transform(test_df[[label_column]])

    # Save fitted scaler for downstream inference
    os.makedirs(output_dir, exist_ok=True)
    scaler_path = os.path.join(output_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"Saved fitted scaler to: {scaler_path}")

    # Export split DataFrames with scaled labels
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    print(
        f"Data split completed | Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
    )

    # Initialize Tokenizer and build DataLoaders
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=cache_dir, trust_remote_code=True
    )

    train_dataset = ProteinDMSDataset(
        sequences=train_df[seq_column].tolist(),
        labels=train_df["DMS_score_scaled"].tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
    )

    val_dataset = ProteinDMSDataset(
        sequences=val_df[seq_column].tolist(),
        labels=val_df["DMS_score_scaled"].tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
    )

    test_dataset = ProteinDMSDataset(
        sequences=test_df[seq_column].tolist(),
        labels=test_df["DMS_score_scaled"].tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, tokenizer, scaler


def get_inference_dataloader(
    sequences: list, tokenizer, batch_size: int = 16, max_length: int = 512
):
    """Creates a DataLoader for unlabelled sequence prediction during inference."""
    dataset = InferenceDataset(
        sequences=sequences, tokenizer=tokenizer, max_length=max_length
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)