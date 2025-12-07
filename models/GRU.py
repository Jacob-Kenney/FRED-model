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
        pass