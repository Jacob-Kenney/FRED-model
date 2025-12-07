from models.DroneStalker import DroneStalker
import torch
import torch.nn as nn

# LSTM model
class Model(DroneStalker):
    def __init__(self, Np: int, Nf: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.1):
        super().__init__(Np, Nf)
        self.gru = nn.GRU(input_size = 4, hidden_size = hidden_dim, num_layers = num_layers, dropout = dropout, batch_first = True)
        self.classifier = nn.Linear(hidden_dim, Nf * 4)

    def forward(self, batch):
        batch_size = batch.shape[0]

        # Extract features
        features = []
        for sample in batch:
            features.append(self._extract_features(sample))
        x = torch.tensor(features, dtype=torch.float32)

        # Forward pass
        gru_out, _ = self.gru(x)
        out = self.classifier(gru_out[:, -1, :])
        return out.view(batch_size, self.Nf, 4)