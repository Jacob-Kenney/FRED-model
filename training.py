import torch

def train(model, dataloaders, loss_function, optimizer, epochs: int = 50, save_path: str = "model.pth", image_width: int = 1280, image_height: int = 720):
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
        loss, ade, fde = test_loop(model, dataloaders[1], loss_function)

        # Save the best model each epoch
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved (loss: {best_loss:.6f} | ADE: {ade:.6f} pixels | FDE: {fde:.6f} pixels)")

    # Log final progress
    print(f"\n{'='*60}")
    print(f"Training complete! Best validation loss: {best_loss:.6f}")
    print(f"Model saved to: {save_path}")
    print(f"{'='*60}\n")

def train_loop(model, dataloader, loss_function, optimizer, image_width: int = 1280, image_height: int = 720):
    size = len(dataloader.dataset)
    
    # Train loop
    model.train()
    for i, (X, y) in enumerate(dataloader):
        # Min-max normalised output
        output = model(X)
        # Denormalize predictions to pixels
        output_pixels = _denormalize_output(output, image_width, image_height)
        # Calculate loss
        loss = loss_function(output_pixels, y)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Log progress
        if i % 100 == 0:
            loss, current = loss.item(), i * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(model, dataloader, loss_function, image_width: int = 1280, image_height: int = 720):
    num_batches = len(dataloader)
    loss, ade, fde, samples, fde_samples = 0, 0, 0, 0, 0

    # Test loop
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            # Min-max normalised output
            output = model(X)
            # Denormalize predictions to pixels
            output_pixels = _denormalize_output(output, image_width, image_height)
            # Calculate loss
            loss += loss_function(output_pixels, y).item()

            # Calculate average displacement error (ADE) in pixels
            distance = torch.sqrt(((output_pixels - y) ** 2).sum(dim=2))
            ade += distance.sum().item()
            samples += distance.numel()

            # Calculate final displacement error (FDE) in pixels
            last_distance = torch.sqrt(((output_pixels[:, -1, :] - y[:, -1, :]) ** 2).sum(dim=1))
            fde += last_distance.sum().item()
            fde_samples += last_distance.numel()

    # Log progress
    loss /= num_batches
    ade /= samples
    fde /= fde_samples
    print(f"Test Error: \n Avg loss: {loss:>8f} | Avg ADE: {ade:>8f} pixels | Avg FDE: {fde:>8f} pixels\n")
    return loss, ade, fde

def _denormalize_output(output, image_width: int, image_height: int):
    output_pixels = output.clone()
    output_pixels[:, :, [0, 2]] *= image_width
    output_pixels[:, :, [1, 3]] *= image_height
    return output_pixels