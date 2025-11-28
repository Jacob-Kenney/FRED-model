import os
from datasets import load_dataset
from torch.utils.data import DataLoader

# Consume processing function, batch size, produce tuple of train and test dataloaders (train, test)
def get_dataloaders(dataset_name: str, process_function = None, batch_size: int = None, shuffle: bool = False) -> tuple[DataLoader, DataLoader]:
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
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader