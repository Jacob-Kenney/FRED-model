# Process data for use in PyTorch for prediction, upload to Hugging Face
from datasets import load_dataset, Dataset, DatasetDict

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
        for sequence_id in list(dict.fromkeys(dataset["sequence_id"])):
            sequence = dataset.filter(lambda x: x["sequence_id"] == sequence_id)
            sequence = sequence.sort("frame_id")
            result.append(sequence)
        return result

# Consume dataset, produce list of sequences grouped by track_id
def process_track_ids(dataset):
    result = []
    for sequence in dataset:
        for track_id in list(dict.fromkeys(sequence["track_id"])):
            if track_id == -1:
                continue
            track = sequence.filter(lambda x: x["track_id"] == track_id or x["track_id"] == -1)
            track = track.sort("frame_id")
            result.append(track.to_dict())
    return result


dataset = load_dataset("Ecoaetix/uFRED")
dataset = dataset.sort(["sequence_id", "frame_id"])
# Process dataset
train = dataset["train"]
test = dataset["test"]
(train, test) = prediction_data_processor(train, test)
print(f"Train samples: {len(train)}")
print(f"Test samples: {len(test)}")

# Upload to Hugging Face
processed_dataset = DatasetDict({"train": Dataset.from_list(train), "test": Dataset.from_list(test)})
processed_dataset.push_to_hub("Ecoaetix/uFRED-predict")
