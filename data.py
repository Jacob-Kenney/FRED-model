import os
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Consume batch, produce tuple of past and future tensors
def collate_fn(batch):
    past = torch.tensor([item["past"] for item in batch], dtype=torch.float32)
    future = torch.tensor([item["future"] for item in batch], dtype=torch.float32)
    return past, future

# Consume processing function, batch size, produce tuple of train and test dataloaders (train, test)
def get_dataloaders(dataset_name: str, process_function = None, batch_size: int = None, shuffle: bool = False, num_workers: int = 0) -> tuple[DataLoader, DataLoader]:
    # Load and sort dataset
    if os.getenv("HF_TOKEN") is not None:
        dataset = load_dataset(dataset_name, token=os.getenv("HF_TOKEN"))
    else:
        dataset = load_dataset(dataset_name)
    # Process dataset
    train = dataset["train"]
    test = dataset["test"]
    if process_function is not None:
        (train, test) = process_function(train, test)

    # Create dataloaders
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=num_workers)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)

    return train_loader, test_loader