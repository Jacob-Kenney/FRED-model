import torch

def train(model, dataloaders, loss_function, optimizer, epochs: int = 50, save_path: str = "model.pth"):
    # Initialise best loss
    best_loss = float('inf')

    # Train loop
    for epoch in range(epochs):
        # Log 
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'='*60}")

        # Training
        train_loop(model, dataloaders[0], loss_function, optimizer)
        loss, ade = test_loop(model, dataloaders[1], loss_function)

        # Save the best model each epoch
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved (loss: {best_loss:.6f} | ADE: {ade:.6f} pixels)")

    # Log final progress
    print(f"\n{'='*60}")
    print(f"Training complete! Best validation loss: {best_loss:.6f}")
    print(f"Model saved to: {save_path}")
    print(f"{'='*60}\n")

def train_loop(model, dataloader, loss_function, optimizer):
    size = len(dataloader.dataset)
    
    # Train loop
    model.train()
    for i, (X, y) in enumerate(dataloader):
        output = model(X)
        loss = loss_function(output, y)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Log progress
        if i % 100 == 0:
            loss, current = loss.item(), i * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(model, dataloader, loss_function):
    num_batches = len(dataloader)
    loss, ade, samples = 0, 0, 0

    # Image dimensions for denormalization
    IMAGE_WIDTH = 1280
    IMAGE_HEIGHT = 720

    # Test loop
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            output = model(X)
            loss += loss_function(output, y).item()

            # Calculate average displacement error (ADE) in pixels
            # Positions are normalized, so denormalize them first
            predicted_x = output[:, :, 0] * IMAGE_WIDTH   # [batch, Nf]
            predicted_y = output[:, :, 1] * IMAGE_HEIGHT  # [batch, Nf]
            true_x = y[:, :, 0] * IMAGE_WIDTH
            true_y = y[:, :, 1] * IMAGE_HEIGHT

            # Euclidean distance in pixels
            distance = torch.sqrt((predicted_x - true_x) ** 2 + (predicted_y - true_y) ** 2)
            ade += distance.sum().item()
            samples += distance.numel()

    # Log progress
    loss /= num_batches
    ade /= samples
    print(f"Test Error: \n Avg loss: {loss:>8f} | Avg ADE: {ade:>8f} pixels\n")
    return loss, ade
