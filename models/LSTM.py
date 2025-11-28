import torch
import torch.nn as nn

# LSTM model
class Model(nn.Module):
    INTERVAL = 0.033333 # Seconds
    IMAGE_WIDTH = 1280 
    IMAGE_HEIGHT = 720
    
    def __init__(self, Np: int, Nf: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.Np = Np
        self.Nf = Nf

        # LSTM layer
        self.lstm = nn.LSTM(input_size = 4, hidden_size = hidden_dim, num_layers = num_layers, dropout = dropout, batch_first = True)
        # Prediction head [Nf, 4]
        self.classifier = nn.Linear(hidden_dim, Nf * 4)

    def forward(self, batch):
        batch_size = batch.shape[0]
        
        # Extract features
        features = []
        for sample in batch:
            features.append(self._extract_features(sample))
        x = torch.tensor(features, dtype=torch.float32)

        # Forward pass
        lstm_out, _ = self.lstm(x)
        out = self.classifier(lstm_out[:, -1, :])
        return out.view(batch_size, self.Nf, 4)

    def _extract_features(self, sample):
        past_boxes = sample["past"]
        features = []
        for i, box in enumerate(past_boxes):
            if i == 0:
                features.append(self._get_kinematics(box, box))
                continue
            past_box = past_boxes[i - 1]
            features.append(self._get_kinematics(past_box, box))
        return features

    def _get_kinematics(self, past_box, box):
        past_x1, past_y1, past_x2, past_y2 = (past_box[0] / self.IMAGE_WIDTH, past_box[1] / self.IMAGE_HEIGHT, past_box[2] / self.IMAGE_WIDTH, past_box[3] / self.IMAGE_HEIGHT)
        x1, y1, x2, y2 = (box[0] / self.IMAGE_WIDTH, box[1] / self.IMAGE_HEIGHT, box[2] / self.IMAGE_WIDTH, box[3] / self.IMAGE_HEIGHT)
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        past_x_center = (past_x1 + past_x2) / 2
        past_y_center = (past_y1 + past_y2) / 2
        x_velocity = (x_center - past_x_center) / (self.INTERVAL)
        y_velocity = (y_center - past_y_center) / (self.INTERVAL)
        return [x_center, y_center, x_velocity, y_velocity]
