import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data.dataset_preparation import prepare_datasets
from src.models.vit_model import create_vit_model
from src.training.metrics import calculate_accuracy
from src.utils.plot_curves import plot_loss_accuracy_curves

def train_vit_model(num_epochs=10, batch_size=16, learning_rate=1e-4):
    train_dataset, val_dataset = prepare_datasets()
    
    if train_dataset is None or val_dataset is None:
        print("Dataset preparation failed.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_vit_model().to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(correct / total)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                predicted = (outputs > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(correct / total)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, Train Acc: {train_accuracies[-1]:.4f}, "
              f"Val Acc: {val_accuracies[-1]:.4f}")

    plot_loss_accuracy_curves(train_losses, val_losses, train_accuracies, val_accuracies)

    torch.save(model.state_dict(), "vit_binary_classifier.pth")
    print("Model saved as vit_binary_classifier.pth")

if __name__ == "__main__":
    train_vit_model(num_epochs=10, batch_size=16, learning_rate=1e-4)