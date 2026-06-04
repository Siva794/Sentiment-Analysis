# src/quantize.py

import torch
import os
import time


def quantize_model(model):
    print("\n⚡ Applying Dynamic Quantization...\n")

    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},  # Only quantize Linear layers
        dtype=torch.qint8
    )

    return quantized_model


def save_model_size(model, path):
    torch.save(model.state_dict(), path)
    size = os.path.getsize(path) / (1024 * 1024)
    return size


def measure_inference_time(model, test_loader, device):
    model.eval()
    model.to(device)

    start_time = time.time()

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

    end_time = time.time()

    return end_time - start_time