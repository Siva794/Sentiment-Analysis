# main.py

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


def run_pipeline():
    print("\n🚀 Starting Efficient NLP Pipeline...\n")

    # =========================
    # 📥 1. Load Raw Data
    # =========================
    print("📂 Loading dataset...")
    df = load_raw_data("data/raw/combined_emotion.csv")

    # =========================
    # 🧹 2. Preprocessing
    # =========================
    print("\n🧹 Cleaning and preprocessing data...")
    df = clean_data(df)
    df = encode_labels(df)

    # Reduce dataset for faster training (can increase later)
    df = reduce_dataset(df, sample_size=50000)

    # =========================
    # 🔀 3. Train-Test Split
    # =========================
    print("\n🔀 Splitting dataset...")
    train_df, test_df = split_data(df)

    # Save processed data
    save_data(train_df, test_df)

    # =========================
    # 📦 4. DataLoaders
    # =========================
    print("\n📦 Creating DataLoaders...")
    train_loader, test_loader = get_dataloaders(
        "data/processed/train.csv",
        "data/processed/test.csv"
    )

    # =========================
    # 🧠 5. Model Training
    # =========================
    print("\n🧠 Starting training...")
    model = train_model(train_loader, epochs=1)

    # =========================
    # 💾 6. Save Model
    # =========================
    print("\n💾 Saving model...")
    os.makedirs("models/baseline", exist_ok=True)

    model_path = "models/baseline/model.pth"
    torch.save(model.state_dict(), model_path)

    print(f"✅ Model saved at: {model_path}")

    print("\n🎉 Pipeline completed successfully!\n")


if __name__ == "__main__":
    run_pipeline()