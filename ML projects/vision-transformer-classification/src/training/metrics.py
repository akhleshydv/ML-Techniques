def calculate_accuracy(outputs, labels):
    preds = (outputs > 0.5).float()  # Convert probabilities to binary predictions
    correct = (preds == labels).sum().item()
    accuracy = correct / labels.size(0)
    return accuracy

def calculate_loss(loss_values):
    return sum(loss_values) / len(loss_values) if loss_values else 0.0

def get_metrics(train_outputs, train_labels, val_outputs, val_labels, train_loss_values, val_loss_values):
    train_accuracy = calculate_accuracy(train_outputs, train_labels)
    val_accuracy = calculate_accuracy(val_outputs, val_labels)
    train_loss = calculate_loss(train_loss_values)
    val_loss = calculate_loss(val_loss_values)

    return {
        'train_accuracy': train_accuracy,
        'val_accuracy': val_accuracy,
        'train_loss': train_loss,
        'val_loss': val_loss
    }