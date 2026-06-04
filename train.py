# train.py

import os
import torch

from src.data_loader import load_raw_data
from src.preprocess import (
    clean_data,
    encode_labels,
    reduce_dataset,
    split_data,
    save_data
)
from src.dataloader_builder import get_dataloaders
from src.train import train_model


def train_pipeline():
    print("\n🚀 TRAINING PIPELINE STARTED\n")

    # =========================
    # 📥 Load & preprocess data
    # =========================
    df = load_raw_data("data/raw/combined_emotion.csv")
    df = clean_data(df)
    df = encode_labels(df)
    df = reduce_dataset(df, sample_size=50000)

    # =========================
    # 🔀 Split
    # =========================
    train_df, test_df = split_data(df)
    save_data(train_df, test_df)

    # =========================
    # 📦 DataLoader
    # =========================
    train_loader, _ = get_dataloaders(
        "data/processed/train.csv",
        "data/processed/test.csv"
    )

    # =========================
    # 🧠 Train Model
    # =========================
    model = train_model(train_loader, epochs=1)

    # =========================
    # 💾 Save Model
    # =========================
    os.makedirs("models/baseline", exist_ok=True)
    model_path = "models/baseline/model.pth"

    torch.save(model.state_dict(), model_path)

    print(f"\n✅ Model saved at: {model_path}")
    print("\n🎉 Training completed!\n")


if __name__ == "__main__":
    train_pipeline()