# src/dataloader_builder.py

from torch.utils.data import DataLoader
from src.data_loader import EmotionDataset


def get_dataloaders(train_path, test_path, batch_size=8):
    train_dataset = EmotionDataset(train_path)
    test_dataset = EmotionDataset(test_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader