from datasets import load_dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

class Model(nn.Module):
    def __init__(self, Np: int, Nf: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.Np = Np
        self.Nf = Nf

        # LSTM layer
        self.lstm = nn.LSTM(input_size = 4, hidden_size = hidden_dim, num_layers = num_layers, dropout = dropout)
        # Prediction head [Nf, 4]
        self.classifier = nn.Linear(hidden_dim, Nf * 4)

# Consume processing function, batch size, produce tuple of train and test dataloaders (train, test)
def get_dataloaders(dataset_name: str, process_function: None, batch_size: int = None, shuffle: bool = False) -> tuple[DataLoader, DataLoader]:
    # Load and sort dataset
    dataset = load_dataset(dataset_name)
    dataset = dataset.sort(["sequence_id", "frame_id"])
    # Process dataset
    train = dataset["train"]
    test = dataset["test"]
    if process_function is not None:
        (train, test) = process_function(train, test)

    # Create dataloaders
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

test, train = get_dataloaders("Ecoaetix/uFRED-predict", shuffle=True)
print("Test set:")
print(next(iter(test)))
print("Train set:")
print(next(iter(train)))