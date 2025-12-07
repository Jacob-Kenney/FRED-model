import torch
import torch.nn as nn

# LSTM model
class DroneStalker(nn.Module):
    INTERVAL = 0.033333 # Seconds
    IMAGE_WIDTH = 1280 
    IMAGE_HEIGHT = 720
    
    def __init__(self, Np: int, Nf: int):
        super().__init__()
        self.Np = Np
        self.Nf = Nf

    def _extract_features(self, sample):
        features = []
        for i, box in enumerate(sample):
            if i == 0:
                features.append(self._get_kinematics(box, box))
                continue
            past_box = sample[i - 1]
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