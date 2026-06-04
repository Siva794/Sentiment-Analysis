# src/train.py

import torch
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from torch.amp import autocast, GradScaler   # ✅ UPDATED API
from tqdm import tqdm
import matplotlib.pyplot as plt
import os


def train_model(train_loader, num_labels=6, epochs=1, lr=5e-5):
    # =========================
    # 🔥 Device Setup
    # =========================
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        use_amp = True
    else:
        device = torch.device("cpu")
        print("⚠️ Using CPU")
        use_amp = False

    # =========================
    # 🧠 Load Model
    # =========================
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=num_labels
    )
    model.to(device)

    # =========================
    # ⚙️ Optimizer
    # =========================
    optimizer = AdamW(model.parameters(), lr=lr)

    scaler = GradScaler(device_type="cuda", enabled=use_amp)

    model.train()

    epoch_losses = []
    best_loss = float("inf")

    os.makedirs("models/improved", exist_ok=True)

    # =========================
    # 🔁 Training Loop
    # =========================
    for epoch in range(epochs):
        print(f"\n🔥 Epoch {epoch+1}/{epochs}")
        total_loss = 0

        loop = tqdm(train_loader, leave=True)

        for step, batch in enumerate(loop):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            with autocast(device_type="cuda", enabled=use_amp):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

            # =========================
            # 💾 MID-EPOCH BACKUP
            # =========================
            if step % 2000 == 0 and step != 0:
                temp_path = "models/improved/temp_backup.pth"
                torch.save(model.state_dict(), temp_path)
                print(f"💾 Mid-epoch backup saved at step {step}")

        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)

        print(f"✅ Epoch {epoch+1} Loss: {avg_loss:.4f}")

        # =========================
        # 💾 SAVE EVERY EPOCH
        # =========================
        epoch_path = f"models/improved/checkpoint_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), epoch_path)
        print(f"💾 Saved checkpoint: {epoch_path}")

        # =========================
        # 🏆 SAVE BEST MODEL
        # =========================
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = "models/improved/best_model.pth"
            torch.save(model.state_dict(), best_path)
            print("🏆 Best model updated")

    # =========================
    # 📊 Save Training Graph
    # =========================
    save_training_plot(epoch_losses)

    return model


# =========================
# 📊 Plot Function
# =========================
def save_training_plot(losses):
    os.makedirs("results/plots/improved", exist_ok=True)

    plt.figure()
    plt.plot(losses, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Improved Training Loss Curve")

    plot_path = "results/plots/improved/training_loss_improved.png"
    plt.savefig(plot_path)
    plt.close()

    print(f"📊 Training plot saved: {plot_path}")