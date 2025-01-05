import torch
torch.backends.cudnn.enabled = False

def evaluate_simple_cnn(model, dataloader, criterion, device='cpu'):
    """
    Evaluate the model on the validation/test dataset.

    Parameters:
    - model: The trained model to evaluate.
    - dataloader: DataLoader for the validation/test dataset.
    - criterion: The loss function.
    - device: The device to run the computations on.

    Returns:
    - evaluation: A dictionary containing loss, accuracy, and wrong cases.
    """
    model.eval()  # Set the model to evaluation mode

    tot_loss = 0.0
    tot_correct = 0
    tot_count = 0
    wrong_cases = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Update total loss
            tot_loss += loss.item() * images.size(0)

            # Calculate accuracy
            preds = outputs.argmax(dim=1)
            correct = preds.eq(labels).sum().item()
            tot_correct += correct
            tot_count += images.size(0)

            # Track wrong cases
            wrong_indices = ~preds.eq(labels)
            for img, label, pred in zip(images[wrong_indices], labels[wrong_indices], preds[wrong_indices]):
                wrong_cases.append((img.cpu(), label.item(), pred.item()))

    # Compute overall metrics
    avg_loss = tot_loss / tot_count
    accuracy = tot_correct / tot_count

    evaluation = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'wrong_cases': wrong_cases
    }
    return evaluation
