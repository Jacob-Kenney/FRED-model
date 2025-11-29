import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path and load .env
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
load_dotenv(project_root / ".env")

import torch
import torch.nn as nn
from models.LSTM import Model
from data import get_dataloaders
from training import test_loop

def evaluate_model(weights_path: str, dataset_name: str = "Ecoaetix/uFRED-predict-0.4", Np: int = 12, Nf: int = 12, hidden_dim: int = 16, num_layers: int = 1, dropout: float = 0, batch_size: int = 256, num_workers: int = 4):
    print(f"\nLoading model from: {weights_path}")
    model = Model(
        Np=Np,
        Nf=Nf,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    )

    # Load trained weights
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load test data
    print(f"\nLoading dataset: {dataset_name}")
    _, test_loader = get_dataloaders(
        dataset_name=dataset_name,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    print(f"Test samples: {len(test_loader.dataset)}")

    # Evaluate
    print("\nEvaluating on test set...")
    loss_function = nn.SmoothL1Loss()
    loss, ade, fde, miou = test_loop(model, test_loader, loss_function)

    # Final results
    print("\n" + "="*60)
    print("Final Results")
    print("="*60)
    print(f"Test Loss: {loss:.6f}")
    print(f"Average Displacement Error (ADE): {ade:.2f} pixels")
    print(f"Final Displacement Error (FDE): {fde:.2f} pixels")
    print(f"Mean IoU (mIoU): {miou:.4f}")
    print("="*60 + "\n")

    return loss, ade, fde, miou

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    evaluate_model(
        weights_path=str(project_root / "weights/LSTM-0.3/drone_stalker-0.3.pth"),
        dataset_name="Ecoaetix/uFRED-predict-0.4",
        Np=12,
        Nf=12,
        hidden_dim=16,
        num_layers=1,
        dropout=0,
        batch_size=256,
        num_workers=4
    )