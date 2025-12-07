import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from models.LSTM import Model
from datasets import Dataset, Features, Value, Image as DatasetImage, Sequence
import numpy as np
from PIL import Image
import cv2
import zipfile
import os
import re
from collections import defaultdict


def generate_gif(zip_path: str, output_path: str, type: str = "rgb", weights_path: str = "weights/LSTM-0.3/drone_stalker-0.3.pth", Np: int = 12, Nf: int = 12, hidden_dim: int = 16, num_layers: int = 1, dropout: float = 0, max_frames: int = None, skip_frames: int = 0, fps: int = 10):
    # Load sequence from zip file
    dataset_features = Features({
        'sequence_id': Value('int32'),
        'frame_id': Value('int32'),
        'timestamp': Value('string'),
        'rgb_image': DatasetImage(),
        'event_image': DatasetImage(),
        'bounding_box': Sequence(Value('float32')),
        'track_id': Value('int32'),
        'class': Value('string'),
    })

    dataset = Dataset.from_generator(
        lambda: zip_generator(zip_path),
        features=dataset_features
    )
    sequence_data = sorted(dataset, key=lambda x: x["frame_id"])
    print("Sequence data loaded")

    # Load trained model
    model = Model(Np=Np, Nf=Nf, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    print("Model loaded")
    
    # Generate predictions for each track
    tracks = set([row["track_id"] for row in sequence_data])
    predictions = {}
    for track in tracks: 
        track_data = [row for row in sequence_data if row["track_id"] == track or row["track_id"] == -1]
        for current_i, frame in enumerate(track_data[Np:-Nf], start=Np):
            if frame["bounding_box"]:
                # 12 consecutive frames have coordinates?
                past_coordinates = []
                for past_i in range(current_i - (Np - 1), current_i):
                    if track_data[past_i]["bounding_box"]:
                        past_coordinates.append(track_data[past_i]["bounding_box"])
                    else:
                        break
                past_coordinates.append(frame["bounding_box"])
                # Generate predictions and denormalize
                if len(past_coordinates) == Np:
                    if frame["frame_id"] not in predictions:
                        predictions[frame["frame_id"]] = {}
                    input_tensor = torch.tensor(past_coordinates, dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        output = model(input_tensor).squeeze(0)
                        truth = track_data[current_i:current_i+Nf]
                        predictions[frame["frame_id"]][track] = ([_denormalize_bbox(box) for box in output], [row["bounding_box"] for row in truth])
    print("Predictions generated")

    # Generate GIF
    gif_frames = []
    for i, row in enumerate(sequence_data[skip_frames:], start=skip_frames):
        # Generate frame
        image = row["rgb_image"] if type == "rgb" else row["event_image"]
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        gif_frames.append(_draw_frame(frame, row["bounding_box"], predictions.get(row["frame_id"], {})))
        if max_frames and len(gif_frames) == max_frames:
            break
    print("GIF frames generated")
    
    # Save GIF
    duration = int(1000 / fps)
    gif_frames[0].save(
        output_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=duration,
        loop=0
    )
    print("GIF saved")

def _draw_frame(frame, current_bbox, predictions, img_width=1280, img_height=720):
    frame = frame.copy()
    overlay = frame.copy()

    # Draw current bounding box
    if current_bbox:
        cv2.rectangle(frame, (int(current_bbox[0]), int(current_bbox[1])), (int(current_bbox[2]), int(current_bbox[3])), (0, 215, 255), 3)
        cv2.putText(frame, "Current", (int(current_bbox[0]), int(current_bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2)

    # Draw bounding boxes
    for _, (future, past) in predictions.items():
        for box in future:
            # Future boxes
            cv2.rectangle(overlay, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
        for box in past:
            # Past boxes
            if box:
                cv2.rectangle(overlay, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

    # Blend overlay with original frame
    frame = cv2.addWeighted(overlay, 0.6, frame, 1 - 0.6, 0)

    # Add legend
    legend_y = 30
    cv2.putText(frame, "Current (Gold)", (10, legend_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2)
    cv2.putText(frame, "Predicted", (10, legend_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "Truth", (10, legend_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Return PIL Image
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)

def _denormalize_bbox(bbox, img_width=1280, img_height=720):
    x1, y1, x2, y2 = bbox
    return [
        x1 * img_width,
        y1 * img_height,
        x2 * img_width,
        y2 * img_height
    ]

def zip_generator(dir_path):
    """Generate dataset rows from an extracted directory."""
    path = Path(dir_path)
    sequence_id = int(path.name)

    # Get image paths and timestamps
    rgb_frames = {}
    event_frames = {}
    annotations = []

    # Walk through directory and find files
    for root, dirs, files in os.walk(path):
        for file in files:
            full_path = os.path.join(root, file)
            relative_path = os.path.relpath(full_path, path)
            # Normalize path separators for consistent matching
            relative_path_normalized = relative_path.replace(os.sep, '/')

            # Get all RGB frames
            if 'RGB' in relative_path_normalized and file.endswith('.jpg'):
                timestamp = parse_rgb_timestamp(relative_path_normalized)
                rgb_frames[file] = (timestamp, full_path)

            # Get all event frames
            elif 'Event/Frames' in relative_path_normalized and file.endswith('.png'):
                timestamp = parse_event_timestamp(relative_path_normalized)
                event_frames[file] = (timestamp, full_path)

            # Get coordinates
            elif file == 'coordinates.txt':
                with open(full_path, 'r') as f:
                    coordinates_text = [line for line in f.read().splitlines() if line.strip()]
                    annotations = parse_coordinates(coordinates_text)
                    del coordinates_text

    # Sort files by timestamps
    sorted_rgb_files = sorted(rgb_frames.items(), key=lambda x: x[1][0])
    if not sorted_rgb_files:
        print(f"Warning: No RGB frames found in {path.name}")
        return
    del rgb_frames
    sorted_event_files = sorted(event_frames.items(), key=lambda x: x[1][0])
    if not sorted_event_files:
        print(f"Warning: No event frames found in {path.name}")
        return
    del event_frames

    # Pair RGB and event frames by index
    num_frames = min(len(sorted_rgb_files), len(sorted_event_files))

    # Build event timestamp to frame index mapping
    event_ts_to_index = {}
    for idx in range(num_frames):
        basename, (event_ts, full_path) = sorted_event_files[idx]
        event_ts_to_index[event_ts] = idx

    # Map annotations to their closest event frame index
    annotation_map = defaultdict(list)
    for ann in annotations:
        # Convert annotation time to microseconds
        ann_us = int(ann['time'] * 1e6)

        # Find closest event frame timestamp
        if event_ts_to_index:
            closest_event_ts = min(event_ts_to_index.keys(), key=lambda x: abs(x - ann_us))
            frame_idx = event_ts_to_index[closest_event_ts]
            annotation_map[frame_idx].append(ann)

    # Process each paired frame
    for frame_id in range(num_frames):
        # Get RGB frame
        rgb_basename, (rgb_timestamp, rgb_full_path) = sorted_rgb_files[frame_id]
        with open(rgb_full_path, 'rb') as f:
            rgb_image = f.read()

        # Get event frame
        event_basename, (event_timestamp, event_full_path) = sorted_event_files[frame_id]
        with open(event_full_path, 'rb') as f:
            event_image = f.read()

        # Format timestamp string from event timestamp
        timestamp_seconds = event_timestamp / 1e6

        # Get annotations for this frame
        frame_annotations = annotation_map.get(frame_id, [])

        if frame_annotations:
            # Create one row per object in this frame
            for ann in frame_annotations:
                yield {
                    'sequence_id': sequence_id,
                    'frame_id': frame_id,
                    'timestamp': str(timestamp_seconds),
                    'rgb_image': rgb_image,
                    'event_image': event_image,
                    'bounding_box': [ann['x1'], ann['y1'], ann['x2'], ann['y2']],
                    'track_id': ann['track_id'],
                    'class': ann['class']
                }
        else:
            # Create a single row with no objects
            yield {
                'sequence_id': sequence_id,
                'frame_id': frame_id,
                'timestamp': str(timestamp_seconds),
                'rgb_image': rgb_image,
                'event_image': event_image,
                'bounding_box': [],
                'track_id': -1,
                'class': ''
            }

def parse_rgb_timestamp(filename):
    match = re.search(r'Video_\d+_(\d+)_(\d+)_([\d.]+)\.jpg', filename)
    if match:
        hours, minutes, seconds = match.groups()
        total_seconds = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        return total_seconds
    return 0.0

def parse_event_timestamp(filename):
    match = re.search(r'Video_\d+_frame_(\d+)\.png', filename)
    if match:
        return int(match.group(1))
    return 0

def parse_coordinates(coordinates_text):
    annotations = []

    for line in coordinates_text:
        # Get time
        if ':' not in line:
            print(f"Warning: No ':' separator found in line: {line}")
            continue

        time_part, rest = line.split(':', 1)
        parts = [p.strip() for p in rest.split(',')]

        if len(parts) >= 6:
            try:
                annotation = {
                    'time': float(time_part.strip()),
                    'x1': float(parts[0]),
                    'y1': float(parts[1]),
                    'x2': float(parts[2]),
                    'y2': float(parts[3]),
                    'track_id': int(parts[4]),
                    'class': parts[5]
                }
                annotations.append(annotation)
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse annotation line: {line} - {e}")
                continue

    return annotations

if __name__ == "__main__":
    generate_gif(zip_path="/Users/jacob/Downloads/8", output_path="../8.gif", type="event", weights_path="weights/LSTM-0.3/drone_stalker-0.3.pth", Np=12, Nf=12, hidden_dim=16, num_layers=1, dropout=0, max_frames=450, skip_frames=300)
    