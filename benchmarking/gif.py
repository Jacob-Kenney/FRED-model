import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from models.LSTM import Model
from datasets import load_dataset
import numpy as np
from PIL import Image
import cv2


def generate_gif(sequence: int, output_path: str, type: str = "rgb", weights_path: str = "weights/LSTM-0.3/drone_stalker-0.3.pth", Np: int = 12, Nf: int = 12, hidden_dim: int = 16, num_layers: int = 1, dropout: float = 0, max_frames: int = None, skip_frames: int = 0, fps: int = 10):
    # Load trained model
    model = Model(Np=Np, Nf=Nf, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    # Load sequence from dataset
    dataset = load_dataset("Ecoaetix/uFRED-nosplit",split="train", streaming=True)
    sequence_data = sorted([row for row in dataset if row["sequence_id"] == sequence],key=lambda x: x["frame_id"])
    
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
                    if current_i not in predictions:
                        predictions[current_i] = {}
                    input_tensor = torch.tensor(past_coordinates, dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        output = model(input_tensor).squeeze(0)
                        truth = track_data[current_i:current_i+Nf]
                        predictions[current_i][track] = ([_denormalize_bbox(box) for box in output], [row["bounding_box"] for row in truth])

    # Generate GIF
    gif_frames = []
    for i, row in enumerate(sequence_data[skip_frames:], start=skip_frames):
        # Generate frame
        image = row["rgb_image"] if type == "rgb" else row["event_image"]
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        # Ground truth rows needs to be 
        gif_frames.append(_draw_frame(frame, row["bounding_box"], predictions[i]))
        if max_frames and len(gif_frames) == max_frames:
            break
    
    # Save GIF
    duration = int(1000 / fps)
    gif_frames[0].save(
        output_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=duration,
        loop=0
    )

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

if __name__ == "__main__":
    generate_gif(sequence=17, output_path="gifs/17.gif", type="event", weights_path="weights/LSTM-0.3/drone_stalker-0.3.pth", Np=12, Nf=12, hidden_dim=16, num_layers=1, dropout=0, max_frames=100, skip_frames=0)