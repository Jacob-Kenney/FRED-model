# Process data for use in PyTorch for prediction, upload to Hugging Face
from datasets import load_dataset, Dataset, DatasetDict
import os

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

# Consume dataset, produce dataset with collapsed lists
def process_lists(dataset):
    result = []
    for sequence in dataset:
        sequence = sequence.to_dict()
        sequence["sequence_id"] = sequence["sequence_id"][0]
        sequence["track_id"] = (set(sequence["track_id"]) - {-1}).pop()
        sequence["class"] = [c for c in sequence["class"] if c != ''][0]
        result.append(sequence)
    return result

dataset = load_dataset("Ecoaetix/uFRED-predict")
# Collapse redundant lists
train = process_lists(dataset["train"])
test = process_lists(dataset["test"])


# Upload to Hugging Face
processed_dataset = DatasetDict({"train": Dataset.from_list(train), "test": Dataset.from_list(test)})
processed_dataset.push_to_hub("Ecoaetix/uFRED-predict", token=os.getenv("HF_TOKEN"))
