# evaluate.py

import torch
from transformers import AutoModelForSequenceClassification

from src.dataloader_builder import get_dataloaders
from src.evaluate import evaluate_model


def evaluation_pipeline():
    print("\n📊 EVALUATION PIPELINE STARTED\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================
    # 🧠 Load Model
    # =========================
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=6
    )

    model.load_state_dict(
        torch.load("models/baseline/model.pth", map_location=device)
    )

    model.to(device)

    print("✅ Model loaded successfully")

    # =========================
    # 📦 Load Test Data
    # =========================
    _, test_loader = get_dataloaders(
        "data/processed/train.csv",
        "data/processed/test.csv"
    )

    # =========================
    # 📊 Evaluate
    # =========================
    accuracy, f1 = evaluate_model(model, test_loader)

    print("\n🎯 FINAL RESULTS:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")


if __name__ == "__main__":
    evaluation_pipeline()