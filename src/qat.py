# src/qat.py

import torch
from transformers import AutoModelForSequenceClassification


def prepare_qat_model():
    print("\n⚙️ Preparing model for QAT...\n")

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=6
    )

    model.train()

    # ✅ Use SAFE QAT config (per-tensor, avoids crash)
    qat_config = torch.quantization.QConfig(
        activation=torch.quantization.default_fake_quant,
        weight=torch.quantization.default_weight_fake_quant  # 👈 key fix
    )

    # Apply selectively
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.qconfig = qat_config
        else:
            module.qconfig = None  # 🚨 disable for others

    # Prepare QAT
    torch.quantization.prepare_qat(model, inplace=True)

    return model


def convert_qat_model(model):
    print("\n⚡ Converting QAT model to INT8...\n")

    model.eval()

    # Convert safely
    torch.quantization.convert(model, inplace=True)

    return model