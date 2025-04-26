import torch
from src.data.dataset_preparation import prepare_datasets
from src.models.vit_model import create_vit_model
from src.training.train import train_vit_model
from src.utils.plot_curves import plot_loss_accuracy_curves

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Prepare datasets
    train_dataset, val_dataset = prepare_datasets()
    
    if train_dataset and val_dataset:
        # Create the Vision Transformer model
        model = create_vit_model()
        # Train the model and get training history
        history = train_vit_model(model, train_dataset, val_dataset, device)
        
        # Plot loss and accuracy curves
        plot_loss_accuracy_curves(history)

if __name__ == "__main__":
    main()