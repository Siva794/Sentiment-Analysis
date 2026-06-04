# qat_train.py

import torch
from src.dataloader_builder import get_dataloaders
from src.qat import prepare_qat_model, convert_qat_model
from torch.optim import AdamW
from tqdm import tqdm


def qat_training():
    print("\n🚀 QAT TRAINING STARTED\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    train_loader, _ = get_dataloaders(
        "data/processed/train.csv",
        "data/processed/test.csv"
    )

    # Prepare model
    model = prepare_qat_model()
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    model.train()

    for epoch in range(1):  # Keep small for QAT
        print(f"\n🔥 QAT Epoch {epoch+1}")
        loop = tqdm(train_loader)

        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())

    # ✅ Convert to quantized model
    quant_model = convert_qat_model(model)

    # Save model
    torch.save(
        quant_model.state_dict(),
        "models/quantized/qat_model.pth"
    )

    print("\n✅ QAT Model saved successfully!\n")


if __name__ == "__main__":
    qat_training()