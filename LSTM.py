from datasets import load_dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

class Model(nn.Module):
    ...

# Consume processing function, batch size, produce tuple of train and test dataloaders (train, test)
def get_dataloaders(process_function: function = None, batch_size: int = None, shuffle: bool = False) -> tuple[DataLoader, DataLoader]:
    # Load and sort dataset
    dataset = load_dataset("Ecoaetix/uFRED")
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

# Consume train and test datasets, produce tuple of processed train and test datasets
def prediction_data_processor(train, test) -> tuple:
    # Remove images from dataset for drone prediction
    train = train.remove_columns(["rgb_image", "event_image"])
    test = test.remove_columns(["rgb_image", "event_image"])
    # Process sequences
    train = process_sequence(train)
    test = process_sequence(test)
    # Process track_ids
    train = process_track_ids(train)
    test = process_track_ids(test)
    # Return processed datasets
    return train, test
    
# Consume dataset, produce list of sequences
def process_sequence(dataset):
        result = []
        for sequence_id in dataset["sequence_id"].unique():
            sequence = dataset.filter(lambda x: x["sequence_id"] == sequence_id)
            sequence = sequence.sort("frame_id")
            result.append(sequence)
        return result

# Consume dataset, produce list of sequences grouped by track_id
def process_track_ids(dataset):
    result = []
    for sequence in dataset:
        for track_id in sequence["track_id"].unique():
            if track_id == -1:
                continue
            track = sequence.filter(lambda x: x["track_id"] == track_id or x["track_id"] == -1)
            track = track.sort("frame_id")
            result.append(track)
    return result

test, train = get_dataloaders(prediction_data_processor, shuffle=True)
print("Test set:")
print(next(iter(test)))
print("Train set:")
print(next(iter(train)))