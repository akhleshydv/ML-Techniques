import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from pathlib import Path
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vit_b_16
from torch.utils.data import DataLoader

def is_valid_image(filepath):
    try:
        img = Image.open(filepath)
        img.verify()  # Verify it's an image
        return True
    except:
        return False

def create_transforms(image_size=224):
    """
    Create a transformation pipeline for image preprocessing
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform

def prepare_datasets():
    # Set up parameters
    image_size = 224
    dataset_path = "./PetImages"
    
    # Remove corrupted images first
    for class_dir in ['cat', 'dog']:
        dir_path = os.path.join(dataset_path, class_dir)
        for img_path in Path(dir_path).glob('*'):
            if not is_valid_image(img_path):
                print(f"Removing corrupted image: {img_path}")
                os.remove(img_path)
    
    # Create transform pipeline
    transform = create_transforms(image_size)
    
    # Load the full dataset
    try:
        full_dataset = ImageFolder(root=dataset_path, transform=transform)
        
        # Split dataset (80% train, 20% validation)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        # Print dataset information
        print("\nDataset Summary:")
        print(f"Total images: {len(full_dataset)}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"\nClasses: {full_dataset.classes}")
        
        return train_dataset, val_dataset
        
    except Exception as e:
        print(f"Error preparing datasets: {str(e)}")
        return None, None

def train_vit_model(train_dataset, val_dataset, num_epochs=10, batch_size=16, learning_rate=1e-4):
    """
    Train a Vision Transformer (ViT) model for binary classification.
    """
    # Load pre-trained ViT model
    model = vit_b_16(pretrained=True)
    model.heads = nn.Sequential(
        nn.Linear(model.heads.head.in_features, 1),  # Binary classification
        nn.Sigmoid()
    )
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "vit_binary_classifier.pth")
    print("Model saved as vit_binary_classifier.pth")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, val_dataset = prepare_datasets()
    if train_dataset and val_dataset:
        train_vit_model(
            train_dataset, 
            val_dataset,
            num_epochs=10,      # Increase for better results
            batch_size=16,      # Reduce this (default is 32)
            learning_rate=1e-4  # Adjust if training is unstable
        )