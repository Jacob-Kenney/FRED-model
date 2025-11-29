import torch
import torch.nn as nn
import torch.optim as optim
from models.LSTM import Model
from data import get_dataloaders
from training import train

def main():
    torch.set_num_threads(12)
    
    # Log
    print("="*60)
    print("Drone Position Prediction Training")
    print("="*60)

    # Hyperparameters
    Np = 12  # Number of past frames
    Nf = 12  # Number of future frames to predict
    hidden_dim = 128
    num_layers = 2
    dropout = 0.1
    batch_size = 256
    epochs = 50
    learning_rate = 1e-3

    # Load dataloaders
    print("\nLoading dataset...")
    train_loader, test_loader = get_dataloaders(
        dataset_name="Ecoaetix/uFRED-predict-0.4",
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Initialize model
    print("\nInitializing model...")
    model = Model(
        Np=Np,
        Nf=Nf,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss function and optimizer
    loss_function = nn.SmoothL1Loss()  # Robust loss for bbox regression
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"\nTraining configuration:")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Loss function: Smooth L1")
    print(f"  - Optimizer: Adam")

    # Train model
    print("\nStarting training...\n")
    train(
        model=model,
        dataloaders=(train_loader, test_loader),
        loss_function=loss_function,
        optimizer=optimizer,
        epochs=epochs,
        save_path="weights/drone_stalker-0.2.pth"
    )

    print("Training finished!")

if __name__ == "__main__":
    main()
