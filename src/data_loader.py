# src/data_loader.py

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Load raw CSV dataset
    """
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Data loaded successfully: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise


class EmotionDataset(Dataset):
    def __init__(self, file_path, tokenizer_name="distilbert-base-uncased", max_length=128):
        self.data = pd.read_csv(file_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = str(self.data.iloc[idx]["text"])
        label = int(self.data.iloc[idx]["label"])

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }