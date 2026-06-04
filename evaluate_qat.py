# evaluate_qat.py

import torch
from src.dataloader_builder import get_dataloaders
from src.evaluate import evaluate_model
from src.qat import prepare_qat_model, convert_qat_model


def run_qat_evaluation():
    print("\n🚀 Evaluating QAT Model...\n")

    # 🚨 IMPORTANT: Quantized models must run on CPU
    device = torch.device("cpu")

    # =========================
    # 📦 Load Data
    # =========================
    print("📦 Loading test data...")

    _, test_loader = get_dataloaders(
        "data/processed/train.csv",
        "data/processed/test.csv"
    )

    # =========================
    # 🧠 Rebuild QAT Model
    # =========================
    print("⚙️ Rebuilding QAT model structure...")

    model = prepare_qat_model()
    model = convert_qat_model(model)

    # =========================
    # 📥 Load QAT Weights
    # =========================
    print("📥 Loading QAT weights...")

    state_dict = torch.load(
        "models/quantized/qat_model.pth",
        map_location=device
    )

    model.load_state_dict(state_dict)

    model.to(device)

    # =========================
    # 📊 Evaluate Model
    # =========================
    print("📊 Running evaluation...\n")

    evaluate_model(model, test_loader, device)


if __name__ == "__main__":
    run_qat_evaluation()