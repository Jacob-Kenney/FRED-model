from datasets import load_dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

class Model(nn.Module):
    ...

# Consume batch size, produce tuple of train and test dataloaders (train, test)
def get_dataloaders(batch_size: int = 32) -> tuple[DataLoader, DataLoader]:
    dataset = load_dataset("Ecoaetix/uFRED")
    dataset = dataset.sort("sequence_id", "frame_id")
    train = dataset["train"]
    test = dataset["test"]

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader