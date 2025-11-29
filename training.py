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
        loss, ade, fde, miou = test_loop(model, dataloaders[1], loss_function)

        # Save the best model each epoch
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved (loss: {best_loss:.6f} | ADE: {ade:.6f} pixels | FDE: {fde:.6f} pixels | mIoU: {miou:.6f})")

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
    loss, ade, fde, miou, samples, fde_samples, iou_samples = 0, 0, 0, 0, 0, 0, 0

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

            # Calculate mean IoU (mIoU)
            iou = _calculate_iou(output_pixels, y)
            miou += iou.sum().item()
            iou_samples += iou.numel()

    # Log progress
    loss /= num_batches
    ade /= samples
    fde /= fde_samples
    miou /= iou_samples
    print(f"Test Error: \n Avg loss: {loss:>8f} | Avg ADE: {ade:>8f} pixels | Avg FDE: {fde:>8f} pixels | mIoU: {miou:>8f}\n")
    return loss, ade, fde, miou

def _denormalize_output(output, image_width: int, image_height: int):
    output_pixels = output.clone()
    output_pixels[:, :, [0, 2]] *= image_width
    output_pixels[:, :, [1, 3]] *= image_height
    return output_pixels

def _calculate_iou(boxes1, boxes2):
    """
    Calculate IoU between two sets of bounding boxes.

    Args:
        boxes1: [batch, frames, 4] tensor in format [x1, y1, x2, y2]
        boxes2: [batch, frames, 4] tensor in format [x1, y1, x2, y2]

    Returns:
        iou: [batch, frames] tensor of IoU values
    """
    # Extract coordinates
    x1_pred, y1_pred, x2_pred, y2_pred = boxes1[..., 0], boxes1[..., 1], boxes1[..., 2], boxes1[..., 3]
    x1_gt, y1_gt, x2_gt, y2_gt = boxes2[..., 0], boxes2[..., 1], boxes2[..., 2], boxes2[..., 3]

    # Calculate intersection
    x_left = torch.max(x1_pred, x1_gt)
    y_top = torch.max(y1_pred, y1_gt)
    x_right = torch.min(x2_pred, x2_gt)
    y_bottom = torch.min(y2_pred, y2_gt)

    intersection_area = torch.clamp(x_right - x_left, min=0) * torch.clamp(y_bottom - y_top, min=0)

    # Calculate union
    pred_area = (x2_pred - x1_pred) * (y2_pred - y1_pred)
    gt_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)
    union_area = pred_area + gt_area - intersection_area

    # Calculate IoU (avoid division by zero)
    iou = intersection_area / (union_area + 1e-6)

    return iou