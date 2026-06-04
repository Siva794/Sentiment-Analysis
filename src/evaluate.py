# src/evaluate.py

import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm


def evaluate_model(model, test_loader, device):
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # =========================
    # 📊 Metrics
    # =========================
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")

    print(f"\n✅ Accuracy: {accuracy:.4f}")
    print(f"✅ F1 Score: {f1:.4f}")

    print("\n📋 Classification Report:")
    print(classification_report(all_labels, all_preds))

    # =========================
    # 📊 Confusion Matrix
    # =========================
    save_confusion_matrix(all_labels, all_preds)

    return accuracy, f1


def save_confusion_matrix(labels, preds):
    os.makedirs("results/plots", exist_ok=True)

    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    path = "results/plots/confusion_matrix.png"
    plt.savefig(path)
    plt.close()

    print(f"📊 Confusion matrix saved: {path}")