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

def process_positive_examples(dataset, Np: int, Nf: int, stride: int):
    # For collecting only samples with np + nf positive frames (where positive is defined as having a non-empty bounding box)
    result = []
    window_size = Np + Nf

    for track in dataset:
        bboxes = track["bounding_box"]
        # Sliding window with stride
        for start_idx in range(0, len(bboxes) - window_size + 1, stride):
            window_bboxes = bboxes[start_idx:start_idx + window_size]
            # Check if all bboxes in window are non-empty
            all_non_empty = all(
                bbox is not None
                for bbox in window_bboxes
            )
            if all_non_empty:
                # Split into past (first Np frames) and future (last Nf frames)
                past_bboxes = list(window_bboxes[:Np])
                future_bboxes = list(window_bboxes[Np:])
                result.append({
                    'past': past_bboxes,
                    'future': future_bboxes
                })
    return result
    

dataset = load_dataset("Ecoaetix/uFRED-predict")
# Get positive example samples
train = process_positive_examples(dataset["train"], 12, 12, 16)
test = process_positive_examples(dataset["test"], 12, 12, 16)


# Upload to Hugging Face
processed_dataset = DatasetDict({"train": Dataset.from_list(train), "test": Dataset.from_list(test)})
processed_dataset.push_to_hub("Ecoaetix/uFRED-predict-0.4", private=True, token=os.getenv("HF_TOKEN"))
